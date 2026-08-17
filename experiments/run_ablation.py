#!/usr/bin/env python
"""
Ablation study runner for CISSN Paper 1.

Runs each ablation configuration on ETTh1 (or another dataset) and collects
metrics. Reuses experiments/run_benchmark.py's Experiment class (early
stopping, LR schedule, gradient clipping, best-checkpoint restore, and
sanity/history artifacts) via a thin subclass, so an ablation arm is trained
under exactly the same protocol as the full CISSN model -- only the model
architecture and conformal strategy vary per config.

Usage:
    python experiments/run_ablation.py --data ETTh1 --pred_len 96 --train_epochs 20 --seed 42
    python experiments/run_ablation.py --data ETTh1 --all_horizons --seeds 42,123,456
"""
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from cissn.baselines import FlatConformal
from cissn.data.registry import get_dataset_spec
from cissn.models.encoder import DisentangledStateEncoder
from cissn.models.forecast_head import ForecastHead

try:
    from .run_benchmark import (
        Experiment, build_protocol_manifest, build_setting_name, require_clean_source,
        set_random_seed, parse_args as parse_benchmark_args,
    )
except ImportError:
    from run_benchmark import (
        Experiment, build_protocol_manifest, build_setting_name, require_clean_source,
        set_random_seed, parse_args as parse_benchmark_args,
    )


def reject_unsupported_evidence_role(args) -> None:
    """Ablation runs never go through run_benchmark.py's immutable-artifacts
    temp-root/finalize/completion-manifest pipeline, so no --evidence_role
    other than the "development" default can be made true here. Reject the
    other roles outright rather than silently accepting flags this runner
    cannot honor -- an ablation config.json that claimed evidence_role
    "confirmation" without the immutability guarantees behind it would be a
    false record of how the run was produced."""
    role = getattr(args, "evidence_role", "development")
    if role != "development":
        raise ValueError(
            f"run_ablation.py does not support --evidence_role {role}: it never routes "
            "through the immutable-artifacts finalize pipeline, so confirmation- or "
            "selection-grade guarantees cannot be made true for ablation runs."
        )

# ── Ablation configurations ────────────────────────────────────────────────

ABLATION_CONFIGS = {
    "full": {
        "structured_A": True,
        "disentanglement_loss": True,
        "sccp": True,
        "correction_mlp": True,
        "state_dim": 5,
        "description": "Full CISSN model",
    },
    "no_structured_A": {
        "structured_A": False,
        "disentanglement_loss": True,
        "sccp": True,
        "correction_mlp": True,
        "state_dim": 5,
        "description": "Replace structured A with dense learned matrix",
    },
    "no_disentanglement_loss": {
        "structured_A": True,
        "disentanglement_loss": False,
        "sccp": True,
        "correction_mlp": True,
        "state_dim": 5,
        "description": "Disable disentanglement loss (lambda_cov=0, lambda_temp=0)",
    },
    "flat_cp": {
        "structured_A": True,
        "disentanglement_loss": True,
        "sccp": False,
        "correction_mlp": True,
        "state_dim": 5,
        "description": "Flat (marginal) conformal prediction instead of SCCP",
    },
    "no_correction_mlp": {
        "structured_A": True,
        "disentanglement_loss": True,
        "sccp": True,
        "correction_mlp": False,
        "state_dim": 5,
        "description": "Remove correction MLP (pure linear encoder)",
    },
    "state_dim_4": {
        "structured_A": True,
        "disentanglement_loss": True,
        "sccp": True,
        "correction_mlp": True,
        "state_dim": 4,
        "description": "Scalar seasonal instead of 2D rotation (state_dim=4)",
    },
}


# ── Custom encoder for ablation (state_dim != 5, or structured_A/correction_mlp disabled) ──


class DisentangledStateEncoderCustom(nn.Module):
    """Flexible encoder supporting state_dim=4 and toggling structured A / correction MLP."""

    def __init__(self, input_dim, state_dim=4, hidden_dim=64, dropout=0.0,
                 structured_A=True, correction_mlp=True, seasonal_period=None):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.structured_A = structured_A
        self.correction_mlp_flag = correction_mlp

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.innovation = nn.Linear(hidden_dim, state_dim)
        if correction_mlp:
            self.correction_mlp = nn.Sequential(
                nn.Linear(state_dim + hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, state_dim),
            )
            self.correction_scale = nn.Parameter(torch.tensor(0.01))

        if structured_A:
            self.A_level = nn.Parameter(torch.zeros(1))
            self.A_trend = nn.Parameter(torch.zeros(1))
            self.A_gamma = nn.Parameter(torch.zeros(1))
            self.A_resid = nn.Parameter(torch.zeros(1))
            # Same init as DisentangledStateEncoder / _dynamics.py: start the
            # seasonal rotation at the dataset's actual frequency instead of
            # 0, so this arm is comparable to the base model on the seasonal
            # block's init rather than silently using a different one.
            omega_init = 2.0 * math.pi / seasonal_period if seasonal_period is not None else 0.0
            self.omega = nn.Parameter(torch.full((1,), float(omega_init)))
        else:
            self.A_dense = nn.Parameter(torch.eye(state_dim) * 0.9)

    def _get_A(self):
        if self.structured_A:
            a_l = torch.sigmoid(self.A_level) * 0.15 + 0.85
            a_t = torch.sigmoid(self.A_trend) * 0.25 + 0.70
            gamma = torch.sigmoid(self.A_gamma) * 0.20 + 0.80
            a_r = torch.sigmoid(self.A_resid) * 0.40
            w = self.omega
            c, s = torch.cos(w), torch.sin(w)
            A = torch.zeros(self.state_dim, self.state_dim, device=a_l.device)
            A[0, 0] = a_l
            A[1, 1] = a_t
            if self.state_dim >= 5:
                A[2, 2] = gamma * c
                A[2, 3] = -gamma * s
                A[3, 2] = gamma * s
                A[3, 3] = gamma * c
                A[4, 4] = a_r
            elif self.state_dim == 4:
                A[2, 2] = gamma * c  # scalar seasonal
                A[3, 3] = a_r
            return A
        else:
            return self.A_dense

    def _step_from_hidden(self, h_t, s_prev, A):
        b_x = self.innovation(h_t)
        s_linear = s_prev @ A.T + b_x
        if self.correction_mlp_flag:
            corr_in = torch.cat([s_linear, h_t], dim=-1)
            correction = self.correction_scale * torch.tanh(self.correction_mlp(corr_in))
            return s_linear + correction
        return s_linear

    def forward(self, x, return_all_states=False):
        projected = self.input_proj(x)
        A = self._get_A()
        batch, seq_len, _ = projected.shape
        s = torch.zeros(batch, self.state_dim, device=x.device, dtype=x.dtype)
        if return_all_states:
            outs = projected.new_empty(batch, seq_len, self.state_dim)
            for t in range(seq_len):
                s = self._step_from_hidden(projected[:, t, :], s, A)
                outs[:, t, :] = s
            return outs
        for t in range(seq_len):
            s = self._step_from_hidden(projected[:, t, :], s, A)
        return s


# ── Ablation experiment: reuses Experiment's train/test protocol ──────────


class AblationExperiment(Experiment):
    """Experiment subclass that swaps model architecture and conformal
    strategy per ablation config, while inheriting the full training loop
    (early stopping, LR schedule, grad clipping, checkpoint restore, history
    and sanity artifacts) unchanged from Experiment."""

    def __init__(self, args, config: dict):
        self.config = config  # must be set before super().__init__ calls _build_model/_build_head
        super().__init__(args)

    def _build_model(self):
        c = self.config
        seasonal_period = get_dataset_spec(self.args.data)["seasonal_period"]
        if c["state_dim"] == 5 and c["structured_A"] and c["correction_mlp"]:
            return DisentangledStateEncoder(
                input_dim=self.args.enc_in, state_dim=5,
                hidden_dim=self.args.d_model, dropout=self.args.dropout,
                seasonal_period=seasonal_period,
            )
        return DisentangledStateEncoderCustom(
            input_dim=self.args.enc_in, state_dim=c["state_dim"],
            hidden_dim=self.args.d_model, dropout=self.args.dropout,
            structured_A=c["structured_A"], correction_mlp=c["correction_mlp"],
            seasonal_period=seasonal_period,
        )

    def _build_head(self):
        return ForecastHead(
            state_dim=self.config["state_dim"], output_dim=self.args.c_out,
            horizon=self.args.pred_len, hidden_dim=self.args.d_model // 2,
            dropout=self.args.dropout,
        )

    def _select_disentangle_criterion(self):
        # DisentanglementLoss.forward hard-requires state_dim==5 (it targets
        # the five named physical components), so the state_dim_4 ablation
        # arm must train without this loss term entirely, not just with
        # lambda_cov/lambda_temp zeroed.
        if self.config["state_dim"] != 5 or not self.config["disentanglement_loss"]:
            return None
        return super()._select_disentangle_criterion()

    def _calibrate_conformal(self, cal_loader, artifact_dir=None):
        if self.config["sccp"]:
            return super()._calibrate_conformal(cal_loader, artifact_dir)

        # flat_cp arm: same residual collection as the base class, but a
        # single global quantile (FlatConformal) instead of state clustering.
        # Calibrated on the SAME second-half window as the SCCP arms
        # (_shared_calibration_indices / _split_calibration_indices), not the
        # full calibration split -- otherwise this arm would be calibrated on
        # roughly twice the residuals of the arm it is compared against,
        # which is the exact "method calibrated on more data than its
        # comparator" error TestConditioningComparisonFairness exists to
        # prevent.
        self.conformal = FlatConformal(
            alpha=self.args.conformal_alpha,
            multivariate_strategy=self.args.multivariate_strategy,
        )
        all_residuals = []
        self.model.eval()
        self.head.eval()
        with torch.no_grad():
            for batch_x, batch_y, _batch_x_mark, _batch_y_mark in cal_loader:
                _, outputs, batch_y = self._forward_and_slice(batch_x, batch_y)
                all_residuals.append((outputs - batch_y).abs().detach().cpu())
        all_residuals = torch.cat(all_residuals, dim=0)
        selected_indices = self._shared_calibration_indices(all_residuals.shape[0])
        _, calibration_indices = self._split_calibration_indices(selected_indices)
        self.conformal.fit(all_residuals[calibration_indices])
        print("Flat conformal predictor calibrated on held-out calibration split (second half, matching SCCP arms).")
        if artifact_dir is not None:
            Path(artifact_dir).mkdir(parents=True, exist_ok=True)
            (Path(artifact_dir) / "cluster_stats.json").write_text(
                json.dumps({"multivariate_strategy": "flat_cp", "coverage_scope": self.conformal.coverage_scope}),
                encoding="utf-8",
            )
        self.model.train()
        self.head.train()

    def _predict_intervals(self, test_states, preds):
        if self.config["sccp"]:
            return super()._predict_intervals(test_states, preds)
        lower, upper = self.conformal.predict(torch.from_numpy(preds).float())
        return lower.numpy(), upper.numpy(), None


def run_ablation(args, config_key: str, config: dict) -> dict:
    """Train and evaluate a single ablation configuration; return its metrics.json payload."""
    print(f"\n{'='*70}")
    print(f"ABLATION: {config_key} — {config['description']}")
    print(f"{'='*70}")

    ablation_args = argparse_namespace_with_overrides(args, config)
    ablation_args.protocol = build_protocol_manifest(ablation_args)
    set_random_seed(ablation_args.seed)
    exp = AblationExperiment(ablation_args, config)
    setting = f"ABLATION_{config_key}_{build_setting_name(ablation_args)}"
    exp.train(setting)
    exp.test(setting)

    metrics_path = Path(ablation_args.results_dir) / setting / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["config"] = config_key
    payload["description"] = config["description"]
    return payload


def argparse_namespace_with_overrides(args, config: dict):
    """Copy args with the ablation's state_dim and (if disabled) zeroed
    disentanglement-loss weights, so build_setting_name and _build_model see
    values consistent with this arm."""
    import copy
    overridden = copy.deepcopy(args)
    overridden.state_dim = config["state_dim"]
    if not config["disentanglement_loss"]:
        overridden.lambda_cov = 0.0
        overridden.lambda_temp = 0.0
    return overridden


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_ablation_args(argv=None):
    """Reuses run_benchmark's full argparse surface (so ablations always accept
    the same --train_epochs/--patience/--lradj/--grad_clip/etc as the main
    runner and cannot silently drift onto a different protocol), plus two
    ablation-only options.
    """
    import argparse
    pre_parser = argparse.ArgumentParser(
        description="CISSN ablation runner. Remaining options are forwarded to run_benchmark.py.",
    )
    pre_parser.add_argument('--output', type=str, default='./results/ablations.json')
    pre_parser.add_argument('--ablations', type=str, default='all',
                             help='Comma-separated ablation keys, or "all" for all six')
    pre_args, remaining = pre_parser.parse_known_args(args=argv)
    benchmark_args = parse_benchmark_args(remaining)
    return pre_args, benchmark_args


def main(argv=None) -> None:
    wrapper_args, args = parse_ablation_args(argv)
    reject_unsupported_evidence_role(args)
    require_clean_source(args)

    if wrapper_args.ablations == 'all':
        configs = list(ABLATION_CONFIGS.keys())
    else:
        configs = [k.strip() for k in wrapper_args.ablations.split(',')]
        for k in configs:
            if k not in ABLATION_CONFIGS:
                raise ValueError(f"Unknown ablation '{k}'. Available: {list(ABLATION_CONFIGS)}")

    results = {}
    t0 = time.time()
    for key in configs:
        cfg = ABLATION_CONFIGS[key]
        results[key] = run_ablation(args, key, cfg)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Ablation study complete in {elapsed:.1f}s")
    print(f"Results saved to {wrapper_args.output}")
    print(f"{'='*70}")

    os.makedirs(os.path.dirname(wrapper_args.output) or '.', exist_ok=True)
    with open(wrapper_args.output, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
