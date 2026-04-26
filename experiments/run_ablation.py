#!/usr/bin/env python
"""
Ablation study runner for CISSN Paper 1.

Runs each ablation configuration on ETTh1 and collects metrics.
Usage:
    python experiments/run_ablation.py --data ETTh1 --pred_len 96 --train_epochs 10 --seed 42
    python experiments/run_ablation.py --data ETTh1 --all_horizons --seeds 42,123,456
"""
import os
import sys
import argparse
import json
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from cissn.models.encoder import DisentangledStateEncoder
from cissn.models.forecast_head import ForecastHead
from cissn.conformal import StateConditionalConformal
from cissn.losses.disentangle_loss import DisentanglementLoss
from cissn.data.data_loader import get_data_loader
from cissn.evaluation.metrics import (
    mean_squared_error, mean_absolute_error,
    compute_picp, compute_mpiw, winkler_score, calibration_error,
)
from sklearn.metrics import mean_squared_error as sk_mse, mean_absolute_error as sk_mae

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


# ── Main ablation runner ───────────────────────────────────────────────────

def run_ablation(args, config_key, config):
    """Train and evaluate a single ablation configuration."""
    print(f"\n{'='*70}")
    print(f"ABLATION: {config_key} — {config['description']}")
    print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Build encoder with ablation flags ──────────────────────────────────
    input_dim = args.enc_in
    state_dim = config["state_dim"]
    hidden_dim = args.d_model

    if state_dim == 5:
        encoder = DisentangledStateEncoder(
            input_dim=input_dim, state_dim=5, hidden_dim=hidden_dim, dropout=args.dropout,
        ).to(device)
    else:
        encoder = DisentangledStateEncoderCustom(
            input_dim=input_dim, state_dim=state_dim, hidden_dim=hidden_dim,
            structured_A=config["structured_A"],
            correction_mlp=config["correction_mlp"],
        ).to(device)

    head = ForecastHead(
        state_dim=state_dim, output_dim=args.c_out, horizon=args.pred_len,
        hidden_dim=args.d_model // 2,
    ).to(device)

    # ── Data ────────────────────────────────────────────────────────────────
    _, train_loader = get_data_loader(args, 'train')
    _, vali_loader = get_data_loader(args, 'val')
    _, test_loader = get_data_loader(args, 'test')

    criterion = nn.MSELoss()
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate)

    if config["disentanglement_loss"] and state_dim == 5:
        disentangle_criterion = DisentanglementLoss(
            lambda_cov=args.lambda_cov, lambda_temporal=args.lambda_temp,
        ).to(device)
    else:
        disentangle_criterion = None

    # ── Training ───────────────────────────────────────────────────────────
    for epoch in range(args.train_epochs):
        encoder.train()
        head.train()
        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)

            optimizer.zero_grad()
            states = encoder(batch_x, return_all_states=True)
            final_state = states[:, -1, :]
            outputs = head(final_state)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            pred_loss = criterion(outputs, batch_y)
            if disentangle_criterion is not None:
                dis_loss = disentangle_criterion(states)
                loss = pred_loss + dis_loss
            else:
                loss = pred_loss

            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{args.train_epochs} — loss: {loss.item():.6f}")

    # ── Calibrate conformal ────────────────────────────────────────────────
    all_states = []
    all_residuals = []
    encoder.eval()
    head.eval()
    with torch.no_grad():
        for batch_x, batch_y, _, _ in vali_loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            final_state = encoder(batch_x)
            outputs = head(final_state)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            all_states.append(final_state.detach().cpu())
            all_residuals.append((outputs - batch_y).abs().detach().cpu())

    all_states = torch.cat(all_states, dim=0)
    all_residuals = torch.cat(all_residuals, dim=0)

    if config["sccp"]:
        conformal = StateConditionalConformal(alpha=0.1, n_clusters=5, random_state=args.seed)
        conformal.fit(all_states, all_residuals)
    else:
        from cissn.baselines import FlatConformal
        conformal = FlatConformal(alpha=0.1)
        conformal.fit(all_residuals)

    # ── Test evaluation ────────────────────────────────────────────────────
    preds = []
    trues = []
    test_states_list = []

    encoder.eval()
    head.eval()
    with torch.no_grad():
        for batch_x, batch_y, _, _ in test_loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            final_state = encoder(batch_x)
            outputs = head(final_state)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
            test_states_list.append(final_state.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    test_states = np.concatenate(test_states_list, axis=0)

    mse = sk_mse(trues.flatten(), preds.flatten())
    mae = sk_mae(trues.flatten(), preds.flatten())

    # Conformal intervals
    if isinstance(conformal, StateConditionalConformal):
        lower, upper = conformal.predict(
            torch.from_numpy(test_states).float(),
            torch.from_numpy(preds).float(),
        )
        lower, upper = lower.numpy(), upper.numpy()
    else:
        lower, upper = conformal.predict(torch.from_numpy(preds).float())
        lower, upper = lower.numpy(), upper.numpy()

    coverage = compute_picp(lower, upper, trues)
    width = compute_mpiw(lower, upper)
    winkler = winkler_score(lower, upper, trues, alpha=0.1)
    calib_err = calibration_error(lower, upper, trues, alpha=0.1)

    result = {
        "config": config_key,
        "description": config["description"],
        "mse": float(mse),
        "mae": float(mae),
        "coverage": float(coverage),
        "mpiw": float(width),
        "winkler": float(winkler),
        "calibration_error": float(calib_err),
    }
    print(f"  Result: MSE={mse:.4f}, MAE={mae:.4f}, Coverage={coverage:.4f}, MPIW={width:.4f}")
    return result


# ── Custom encoder for ablation (state_dim != 5) ──────────────────────────


class DisentangledStateEncoderCustom(nn.Module):
    """Flexible encoder supporting state_dim=4 and toggling structured A / correction MLP."""

    def __init__(self, input_dim, state_dim=4, hidden_dim=64, dropout=0.0,
                 structured_A=True, correction_mlp=True):
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
            self.omega = nn.Parameter(torch.zeros(1))
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


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CISSN Ablation Study Runner')
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./data/ETT/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--lambda_cov', type=float, default=1.0)
    parser.add_argument('--lambda_temp', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='./results/ablations.json')
    parser.add_argument('--ablations', type=str, default='all',
                        help='Comma-separated ablation keys, or "all" for all six')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.ablations == 'all':
        configs = list(ABLATION_CONFIGS.keys())
    else:
        configs = [k.strip() for k in args.ablations.split(',')]
        for k in configs:
            if k not in ABLATION_CONFIGS:
                raise ValueError(f"Unknown ablation '{k}'. Available: {list(ABLATION_CONFIGS)}")

    results = {}
    t0 = time.time()
    for key in configs:
        cfg = ABLATION_CONFIGS[key]
        result = run_ablation(args, key, cfg)
        results[key] = result

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Ablation study complete in {elapsed:.1f}s")
    print(f"Results saved to {args.output}")
    print(f"{'='*70}")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
