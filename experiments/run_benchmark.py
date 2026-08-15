import os
import sys
import random
import warnings
import json
import platform
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
from pathlib import Path
from typing import Optional
from cissn.models.encoder import DisentangledStateEncoder
from cissn.models.forecast_head import ForecastHead
from cissn.models.hybrid import HybridCISSN
from cissn.models.revin import RevIN
from cissn.losses.disentangle_loss import DisentanglementLoss
from cissn.conformal import StateConditionalConformal, StateScaledConformal
from cissn.baselines import FlatConformal
from cissn.data.data_loader import get_data_loader
from cissn.data.registry import get_dataset_spec, supported_datasets, verify_dataset
from cissn.utils import (
    EarlyStopping, canonical_hash, create_temporary_result_root, finalize_result_directory,
    print_epoch_summary, print_run_header, require_new_run, select_device, track,
    write_completion_manifest,
)
from cissn.evaluation.metrics import (
    mean_squared_error, mean_absolute_error,
    compute_picp, compute_joint_picp, compute_mpiw, winkler_score, per_origin_interval_scores,
    mean_scaled_interval_score,
    fit_coverage_bin_edges, conditional_coverage_by_bin,
)
from cissn.evaluation.sanity import check_forecast_sanity
from cissn.evaluation.collapse import DispersionAccumulator, dispersion_summary


def _format_float_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def build_setting_name(args) -> str:
    # Architecture variants must not collide on disk: a hybrid run and a legacy
    # run with otherwise-identical settings would otherwise share a checkpoint
    # and results directory and silently overwrite each other.
    architecture = getattr(args, "architecture", "legacy")
    variant = ""
    if architecture != "legacy":
        variant = f"_{architecture}_{getattr(args, 'state_dynamics', 'legacy')}"
        if getattr(args, "state_revin", False):
            variant += "_revin"
    if getattr(args, "revin", False):
        variant += "_fullrevin"
    # Only a non-default conditioning mode changes the setting name, so every
    # existing 'cluster' run directory (the default) stays byte-identical.
    conditioning = getattr(args, "conformal_conditioning", "cluster")
    if conditioning != "cluster":
        variant += f"_{conditioning}cond"
    return (
        f"CISSN_{args.data}_{args.features}"
        f"_sl{args.seq_len}_pl{args.pred_len}_sd{args.state_dim}_dm{args.d_model}"
        f"{variant}"
        f"_lc{_format_float_token(args.lambda_cov)}_lt{_format_float_token(args.lambda_temp)}"
        f"_a{_format_float_token(args.conformal_alpha)}_{args.multivariate_strategy}"
        f"_seed{args.seed}"
    )


def build_run_setting(args, base_setting: str) -> str:
    if not getattr(args, "immutable_artifacts", False):
        return base_setting
    return f"{base_setting}__{args.protocol['design_hash'][:12]}"


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    return str(value)


def save_json(path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)


def load_config_defaults(path: Optional[str]) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML configs require PyYAML. Install it or use a JSON config.") from exc
        data = yaml.safe_load(text) or {}
    return _flatten_config(data)


def _flatten_config(data: dict, prefix: str = "") -> dict:
    flattened = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flattened.update(_flatten_config(value, prefix=f"{prefix}{key}."))
        else:
            flattened[prefix + key] = value
    aliases = {
        "dataset.data": "data",
        "dataset.root_path": "root_path",
        "dataset.data_path": "data_path",
        "dataset.features": "features",
        "dataset.target": "target",
        "dataset.freq": "freq",
        "model.enc_in": "enc_in",
        "model.c_out": "c_out",
        "model.d_model": "d_model",
        "model.state_dim": "state_dim",
        "model.dropout": "dropout",
        "training.train_epochs": "train_epochs",
        "training.seq_len": "seq_len",
        "training.label_len": "label_len",
        "training.pred_len": "pred_len",
        "training.batch_size": "batch_size",
        "training.learning_rate": "learning_rate",
        "training.patience": "patience",
        "training.num_workers": "num_workers",
        "training.lradj": "lradj",
        "training.seed": "seed",
        "training.grad_clip": "grad_clip",
        "training.lambda_correction_scale": "lambda_correction_scale",
        "loss.lambda_cov": "lambda_cov",
        "loss.lambda_temp": "lambda_temp",
        "conformal.alpha": "conformal_alpha",
        "conformal.n_clusters": "n_clusters",
        "conformal.multivariate_strategy": "multivariate_strategy",
        "paths.checkpoints": "checkpoints",
        "paths.results_dir": "results_dir",
    }
    return {aliases.get(k, k): v for k, v in flattened.items()}


def provided_cli_options(argv: list[str]) -> set[str]:
    options = set()
    for token in argv:
        if token.startswith("--"):
            options.add(token[2:].split("=", 1)[0].replace("-", "_"))
    return options


def validate_config_defaults(parser: argparse.ArgumentParser, defaults: dict) -> None:
    """Reject config keys that argparse would otherwise accept silently."""
    actions = {action.dest: action for action in parser._actions}
    unknown = sorted(set(defaults) - set(actions))
    if unknown:
        raise ValueError(f"Unknown config key(s): {', '.join(unknown)}")
    for key, value in defaults.items():
        action = actions[key]
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if not isinstance(value, bool):
                raise ValueError(f"Config key {key!r} must be boolean.")
            continue
        if action.type is not None and value is not None:
            try:
                value = action.type(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid value for config key {key!r}: {value!r}") from exc
        if action.choices is not None and value not in action.choices:
            raise ValueError(f"Invalid value for config key {key!r}: {value!r}")


def validate_runtime_args(args) -> None:
    positive = ("seq_len", "label_len", "pred_len", "enc_in", "c_out", "d_model", "state_dim",
                "batch_size", "train_epochs", "n_clusters", "calibration_stride")
    for key in positive:
        if hasattr(args, key) and getattr(args, key) <= 0:
            raise ValueError(f"--{key} must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if not 0.0 < args.conformal_alpha < 1.0:
        raise ValueError("--conformal_alpha must be in (0, 1).")
    if not 0.0 < args.cal_fraction < 1.0:
        raise ValueError("--cal_fraction must be in (0, 1).")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning_rate must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative.")


def apply_dataset_defaults(args, protected_keys: set[str]) -> None:
    spec = get_dataset_spec(args.data)
    for key in ("root_path", "data_path", "freq", "target", "enc_in", "c_out"):
        if key not in protected_keys:
            setattr(args, key, spec[key])


def set_random_seed(seed: int, strict: bool = False) -> None:
    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def environment_snapshot(device: torch.device) -> dict:
    def _git_value(command: list[str]) -> Optional[str]:
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    git_status = _git_value(["git", "status", "--short"])
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
        "git_status": git_status,
        "git_dirty": None if git_status is None else bool(git_status),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def build_protocol_manifest(args) -> dict:
    """Capture immutable launch inputs needed to compare and reproduce a run."""
    excluded = {"checkpoints", "results_dir", "require_clean_git", "immutable_artifacts"}
    config = {key: value for key, value in vars(args).items() if key not in excluded and key != "protocol"}
    dataset = verify_dataset(
        args.data,
        data_root=args.root_path,
        strict=True,
        require_exact_checksum=getattr(args, "immutable_artifacts", False),
    )
    source = environment_snapshot(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    design_payload = {
        "config": config,
        "dataset": {
            key: dataset.get(key)
            for key in ("dataset", "actual_sha256", "actual_semantic_sha256")
        },
        "git_commit": source["git_commit"],
    }
    payload = {
        "protocol": "cissn-publication-2026",
        "config": config,
        "dataset": dataset,
        "source": source,
        "design_hash": canonical_hash(design_payload),
        "execution_fingerprint": canonical_hash(source),
    }
    payload["protocol_hash"] = canonical_hash(payload)
    return payload


def require_clean_source(args) -> None:
    if not getattr(args, "require_clean_git", False):
        return
    source = environment_snapshot(torch.device("cpu"))
    if source["git_commit"] is None or source["git_status"] is None:
        raise RuntimeError("Publication runs require a readable committed Git worktree.")
    if source["git_dirty"]:
        raise RuntimeError("Publication runs require a clean committed Git worktree.")


def enforce_evidence_contract(args) -> None:
    """Make a sealed confirmation run reproducible and non-overwritable."""
    role = getattr(args, "evidence_role", "development")
    if role == "selection":
        raise ValueError(
            "Selection is not supported by the test-evaluation runner; use a chronological "
            "validation-only controller so the test loader is never instantiated."
        )
    if role != "confirmation":
        return
    missing = [
        flag for flag in ("immutable_artifacts", "strict_determinism", "require_clean_git")
        if not getattr(args, flag, False)
    ]
    if missing:
        names = ", ".join(f"--{flag}" for flag in missing)
        raise ValueError(f"Confirmation runs require {names}.")


class Experiment:
    def __init__(self, args):
        self.args = args
        self.device = select_device(require_gpu=getattr(args, 'require_gpu', False))
        self.model = self._build_model().to(self.device)
        self.head = self._build_head().to(self.device)
        # RevIN owns learnable affine parameters, so it must be built before the
        # optimizer collects parameters and moved to the device with the model.
        self.revin = (
            RevIN(num_features=self.args.enc_in).to(self.device)
            if getattr(self.args, 'revin', False)
            else None
        )

    def _build_model(self):
        return DisentangledStateEncoder(
            input_dim=self.args.enc_in,
            state_dim=self.args.state_dim,
            hidden_dim=self.args.d_model,
            dropout=self.args.dropout,
            seasonal_period=get_dataset_spec(self.args.data)["seasonal_period"],
        )

    def _build_head(self):
        return ForecastHead(
            state_dim=self.args.state_dim,
            output_dim=self.args.c_out,
            horizon=self.args.pred_len,
            hidden_dim=self.args.d_model // 2,
            dropout=self.args.dropout,
            use_refinement=not getattr(self.args, 'no_refinement', False),
        )

    def _get_data(self, flag):
        return get_data_loader(self.args, flag)

    def _forward_and_slice(self, batch_x, batch_y, return_all_states=False):
        """Run encoder + head and slice to the prediction window.

        Args:
            return_all_states: If True, return all intermediate states (B, L, S)
                alongside the final-state-based outputs. Used during training
                for the disentanglement loss.

        Returns:
            If return_all_states is False:
                final_state, outputs, batch_y  (sliced)
            If return_all_states is True:
                all_states, final_state, outputs, batch_y  (sliced)
        """
        batch_x = batch_x.float().to(self.device, non_blocking=True)
        batch_y = batch_y.float().to(self.device, non_blocking=True)

        # With RevIN the model sees a per-window standardised input and predicts
        # shape only; the window's own level and scale are restored afterwards.
        revin = getattr(self, "revin", None)
        model_input = revin(batch_x, "norm") if revin is not None else batch_x

        if return_all_states:
            all_states = self.model(model_input, return_all_states=True)
            final_state = all_states[:, -1, :]
        else:
            final_state = self.model(model_input)
            all_states = None

        outputs = self.head(final_state)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
        if revin is not None:
            # Slice first, then denormalise with statistics for exactly those
            # channels. Under MS the forecast is one column, so it must be
            # rescaled with the target's own statistics, not feature 0's.
            scaler = revin.select_channels(-1) if f_dim == -1 else revin
            outputs = scaler(outputs, "denorm")

        if return_all_states:
            return all_states, final_state, outputs, batch_y
        return final_state, outputs, batch_y

    def _trainable_modules(self):
        """Every module holding parameters that training updates."""
        modules = [self.model, self.head]
        if getattr(self, "revin", None) is not None:
            modules.append(self.revin)
        return modules

    def _set_train_mode(self, training: bool) -> None:
        """Toggle train/eval on every parameterised module together."""
        for module in self._trainable_modules():
            if module is not None:
                module.train(training)

    def _select_optimizer(self):
        params = [p for m in self._trainable_modules() for p in m.parameters()]
        return optim.Adam(params, lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _select_disentangle_criterion(self):
        # DisentanglementLoss hard-requires state_dim==5 (it targets the five
        # named physical components); overridden to return None for ablation
        # arms with a different state_dim.
        return DisentanglementLoss(
            lambda_cov=self.args.lambda_cov,
            lambda_temporal=self.args.lambda_temp,
        ).to(self.device)

    @staticmethod
    def _concatenate_batches(batches, name):
        if not batches:
            raise RuntimeError(f"No {name} batches were produced.")
        return np.concatenate(batches, axis=0)

    @staticmethod
    def _summarize_epoch_diagnostics(disentangle_metrics_source, head, state_batches, final_state_batches):
        """disentangle_metrics_source only needs a get_metrics(states) staticmethod
        (DisentanglementLoss.get_metrics has no state_dim==5 requirement, unlike
        .forward's loss term, so it stays valid for ablation arms with a
        different state_dim -- pass the class itself, an instance, or a test double)."""
        if not state_batches or not final_state_batches:
            raise RuntimeError("Epoch diagnostics require at least one training batch.")

        epoch_states = torch.cat(state_batches, dim=0)
        epoch_final_states = torch.cat(final_state_batches, dim=0)
        with torch.no_grad():
            disent_metrics = disentangle_metrics_source.get_metrics(epoch_states)
            refinement_ratio = head.get_refinement_ratio(epoch_final_states)
        return disent_metrics, refinement_ratio

    @staticmethod
    def _coverage_by_cluster(lower, upper, trues, cluster_labels):
        covered = (trues >= lower) & (trues <= upper)
        out = {}
        for k in sorted(set(int(v) for v in cluster_labels.tolist())):
            mask = cluster_labels == k
            out[k] = {
                "n_samples": int(mask.sum()),
                "coverage": float(covered[mask].mean()),
                "mean_width": float((upper[mask] - lower[mask]).mean()),
            }
        return out

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        cal_data, cal_loader = self._get_data(flag='cal')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        self.checkpoint_path = path
        save_json(Path(path) / "config.json", vars(self.args))
        save_json(Path(path) / "environment.json", environment_snapshot(self.device))

        train_start = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        model_parameters = [p for m in self._trainable_modules() for p in m.parameters()]
        criterion = self._select_criterion()
        disentangle_criterion = self._select_disentangle_criterion()
        history = []

        for epoch in range(self.args.train_epochs):
            total_train_loss = torch.zeros((), device=self.device)
            total_train_weight = 0
            epoch_states = []
            epoch_final_states = []

            self.model.train()
            self.head.train()
            epoch_time = time.time()

            batches = track(
                train_loader,
                description=f"Epoch {epoch + 1}/{self.args.train_epochs}",
                total=train_steps,
                enabled=not getattr(self.args, "no_progress", True),
            )
            for i, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(batches):
                model_optim.zero_grad(set_to_none=True)

                states, final_state, outputs, batch_y = self._forward_and_slice(
                    batch_x, batch_y, return_all_states=True
                )

                loss = criterion(outputs, batch_y)
                if disentangle_criterion is not None:
                    loss = loss + disentangle_criterion(states)
                if self.args.lambda_refinement > 0 and hasattr(self.head, "refinement_scale"):
                    # The refinement MLP is not attributable to state coordinates,
                    # so an unpenalised head drifts toward it and the structured
                    # decomposition stops explaining the forecast. Penalising its
                    # scale keeps the interpretable linear path dominant.
                    # refinement_scale only exists when the head was built with
                    # use_refinement=True; --no_refinement removes the module
                    # this penalty targets, so there is nothing to penalise.
                    loss = loss + self.args.lambda_refinement * self.head.refinement_scale.abs()
                if self.args.lambda_correction_scale > 0 and hasattr(self.model, "_correction_scale"):
                    loss = loss + self.args.lambda_correction_scale * (self.model._correction_scale() - 0.01) ** 2
                batch_weight = outputs.numel()
                total_train_loss += loss.detach() * batch_weight
                total_train_weight += batch_weight
                epoch_states.append(states.detach())
                epoch_final_states.append(final_state.detach())

                loss.backward()
                if self.args.grad_clip and self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model_parameters,
                        max_norm=self.args.grad_clip,
                    )
                model_optim.step()

            if total_train_weight == 0:
                raise RuntimeError("Training loader produced no prediction elements.")
            train_loss = float((total_train_loss / total_train_weight).item())
            vali_loss = self.vali(vali_loader, criterion)

            disent_metrics, refinement_ratio = self._summarize_epoch_diagnostics(
                DisentanglementLoss,
                self.head,
                epoch_states,
                epoch_final_states,
            )

            dispersion = getattr(self, "last_vali_dispersion_", {}) or {}
            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "vali_loss": vali_loss,
                "lr": model_optim.param_groups[0]['lr'],
                "off_diag_corr": disent_metrics["mean_abs_off_diag_corr"],
                "refinement_ratio": refinement_ratio,
                "vali_variance_ratio": dispersion.get("variance_ratio"),
                "vali_corr": dispersion.get("corr"),
            })

            improved = early_stopping(vali_loss, self.model, self.head, path)
            if improved:
                self._save_revin(path)
            print_epoch_summary(
                epoch=epoch + 1, total_epochs=self.args.train_epochs, train_loss=train_loss,
                validation_loss=vali_loss, learning_rate=model_optim.param_groups[0]['lr'],
                elapsed_seconds=time.time() - epoch_time, improved=improved,
                patience_counter=early_stopping.counter, patience=early_stopping.patience,
            )
            variance_ratio = dispersion.get("variance_ratio")
            collapse_note = "" if variance_ratio is None else f" | var_ratio={variance_ratio:.4f}"
            print(
                f"  state_corr={disent_metrics['mean_abs_off_diag_corr']:.4f} | "
                f"refinement={refinement_ratio:.4f}{collapse_note}"
            )
            if early_stopping.early_stop:
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        save_json(Path(path) / "history.json", history)

        self._load_checkpoint(path)
        fit_seconds = time.time() - train_start
        self._build_conditioning_predictors()
        self._calibrate_conformal(cal_loader, path)
        self.train_runtime_ = {
            "train_seconds": fit_seconds,
            "fit_seconds": fit_seconds,
            "conditioning_fit_seconds": getattr(self, "conditioning_fit_seconds_", 0.0),
            "calibration_seconds": getattr(self, "quantile_calibration_seconds_", 0.0),
            "total_seconds": time.time() - train_start,
            "train_samples": len(train_data),
            "validation_samples": len(vali_data),
            "calibration_samples": len(cal_data),
            "epochs_requested": self.args.train_epochs,
            "epochs_run": len(history),
            "early_stopped": early_stopping.early_stop,
            "best_val_loss": early_stopping.val_loss_min,
        }
        save_json(Path(path) / "runtime.json", self.train_runtime_)
        return self.model

    def _revin_checkpoint_path(self, path) -> Path:
        return Path(path) / "checkpoint_revin.pth"

    def _save_revin(self, path) -> None:
        """Persist RevIN's affine parameters alongside the model checkpoint.

        EarlyStopping only knows about the model and head, so without this the
        learned affine scale/shift would be silently lost when the best
        checkpoint is restored.
        """
        if getattr(self, "revin", None) is not None:
            torch.save(self.revin.state_dict(), self._revin_checkpoint_path(path))

    def _load_checkpoint(self, path):
        self.model.load_state_dict(
            torch.load(os.path.join(path, 'checkpoint.pth'), map_location=self.device, weights_only=True)
        )
        self.head.load_state_dict(
            torch.load(os.path.join(path, 'checkpoint_head.pth'), map_location=self.device, weights_only=True)
        )
        revin_path = self._revin_checkpoint_path(path)
        if getattr(self, "revin", None) is not None and revin_path.exists():
            self.revin.load_state_dict(
                torch.load(revin_path, map_location=self.device, weights_only=True)
            )

    def vali(self, vali_loader, criterion):
        total_loss = 0.0
        total_weight = 0
        # Records var(pred)/var(true) on the validation split so a forecast that
        # buys MSE by shrinking toward the mean is visible in history.json.
        dispersion = DispersionAccumulator()
        self._set_train_mode(False)
        with torch.no_grad():
            for batch_x, batch_y, _batch_x_mark, _batch_y_mark in track(
                vali_loader,
                description="Validation",
                total=len(vali_loader),
                enabled=not getattr(self.args, "no_progress", True),
            ):
                _, outputs, batch_y = self._forward_and_slice(batch_x, batch_y)
                batch_weight = outputs.numel()
                total_loss += criterion(outputs, batch_y).item() * batch_weight
                total_weight += batch_weight
                dispersion.update(outputs, batch_y)
            if total_weight == 0:
                raise RuntimeError("Validation loader produced no prediction elements.")
        self.last_vali_dispersion_ = dispersion.summary()
        self._set_train_mode(True)
        return total_loss / total_weight

    def _build_conformal(self):
        """Construct the primary conditioning predictor for --conformal_conditioning."""
        mode = getattr(self.args, "conformal_conditioning", "cluster")
        if mode == "scale":
            return StateScaledConformal(
                alpha=self.args.conformal_alpha,
                multivariate_strategy=self.args.multivariate_strategy,
                scale_geometry=getattr(self.args, "scale_geometry", "scalar"),
            )
        return StateConditionalConformal(
            alpha=self.args.conformal_alpha,
            n_clusters=self.args.n_clusters,
            multivariate_strategy=self.args.multivariate_strategy,
            random_state=self.args.seed,
            calibration_stride=1,
        )

    def _build_secondary_conformal(self):
        """The conditioning mode NOT selected by --conformal_conditioning.

        Every run reports both cluster and scale results, paired against the
        same forecasts and calibration residuals as the primary predictor, so
        the choice of default never hides the comparison.
        """
        mode = getattr(self.args, "conformal_conditioning", "cluster")
        if mode == "scale":
            return StateConditionalConformal(
                alpha=self.args.conformal_alpha,
                n_clusters=self.args.n_clusters,
                multivariate_strategy=self.args.multivariate_strategy,
                random_state=self.args.seed,
                calibration_stride=1,
            )
        return StateScaledConformal(
            alpha=self.args.conformal_alpha,
            multivariate_strategy=self.args.multivariate_strategy,
            scale_geometry=getattr(self.args, "scale_geometry", "scalar"),
        )

    def _save_conditioning_stats(self, folder_path: Path) -> None:
        """Write each conditioning predictor's diagnostics under a name that
        matches its actual mode, not its primary/secondary role, so
        'cluster_stats.json' always means the K-Means predictor's stats
        regardless of which mode --conformal_conditioning selected."""
        for predictor in (getattr(self, "conformal", None), getattr(self, "secondary_conformal", None)):
            if predictor is None or not getattr(predictor, "calibrated", False):
                continue
            if isinstance(predictor, StateScaledConformal):
                save_json(folder_path / "scale_stats.json", predictor.get_scale_stats())
            elif hasattr(predictor, "get_cluster_stats"):
                save_json(folder_path / "cluster_stats.json", predictor.get_cluster_stats())

    def _build_conditioning_predictors(self):
        """Construct the conditioning predictors. Fitting happens in _calibrate_conformal.

        Both `StateConditionalConformal.fit_partition` and
        `StateScaledConformal.fit_scale` are fit on the SAME conditioning-half
        calibration window there, not here and not on train states: fitting
        the cluster partition on train states (as an earlier version did)
        gave it a ~9x larger, in-sample fitting set relative to the sigma
        regression, which confounded any ordering between the two mechanisms.
        See docs/methodology.md, "Asymmetry between the two conditioning
        mechanisms".
        """
        self.conformal = self._build_conformal()
        self.secondary_conformal = self._build_secondary_conformal()

    def _calibrate_conformal(self, cal_loader, artifact_dir=None):
        """Calibrate both conditioning predictors on the held-out calibration split."""
        all_states = []
        all_residuals = []

        self._set_train_mode(False)
        with torch.no_grad():
            for batch_x, batch_y, _batch_x_mark, _batch_y_mark in track(
                cal_loader,
                description="Calibration",
                total=len(cal_loader),
                enabled=not getattr(self.args, "no_progress", True),
            ):
                final_state, outputs, batch_y = self._forward_and_slice(batch_x, batch_y)
                all_states.append(final_state.detach().cpu())
                all_residuals.append((outputs - batch_y).abs().detach().cpu())

        all_states = torch.cat(all_states, dim=0)
        all_residuals = torch.cat(all_residuals, dim=0)
        selected_indices = self._shared_calibration_indices(all_states.shape[0])
        conditioning_indices, calibration_indices = self._split_calibration_indices(selected_indices)
        conditioning_states = all_states[conditioning_indices]
        conditioning_residuals = all_residuals[conditioning_indices]
        calibration_states = all_states[calibration_indices]
        calibration_residuals = all_residuals[calibration_indices]
        self.conditioning_indices_ = conditioning_indices
        self.calibration_indices_ = calibration_indices

        scale_predictor = next(
            (p for p in (self.conformal, self.secondary_conformal) if isinstance(p, StateScaledConformal)), None
        )
        if scale_predictor is None:
            raise RuntimeError("CISSN calibration requires a state-scaled comparator.")
        conditioning_start = time.time()
        # Both conditioning mechanisms fit on the SAME conditioning_states /
        # conditioning_residuals -- the first half of the calibration split --
        # so neither sees train residuals and neither has a sample-size
        # advantage over the other.
        for predictor in (self.conformal, self.secondary_conformal):
            if not isinstance(predictor, StateScaledConformal):
                predictor.fit_partition(conditioning_states)
        scale_predictor.fit_scale(conditioning_states, conditioning_residuals)
        self._coverage_bin_edges = fit_coverage_bin_edges(
            np.linalg.norm(conditioning_states.numpy(), axis=1), n_bins=5
        )
        self.conditioning_fit_seconds_ = time.time() - conditioning_start
        quantile_calibration_start = time.time()
        self.conformal.calibrate(calibration_states, calibration_residuals)
        self.secondary_conformal.calibrate(calibration_states, calibration_residuals)
        # Flat CP, cluster SCCP, and state-scaled CP are all calibrated on the
        # SAME residuals from the SAME model, so every comparison isolates the
        # conditioning mechanism. Running any comparator as a separate
        # training run instead would confound it with training variance.
        self.flat_conformal = FlatConformal(
            alpha=self.args.conformal_alpha,
            multivariate_strategy=self.args.multivariate_strategy,
        )
        self.flat_conformal.fit(calibration_residuals)
        self.quantile_calibration_seconds_ = time.time() - quantile_calibration_start
        print("Calibration complete | primary, secondary, and flat conformal intervals ready")

        if artifact_dir is not None:
            np.save(Path(artifact_dir) / "conditioning_states.npy", conditioning_states.numpy())
            np.save(Path(artifact_dir) / "conditioning_residuals.npy", conditioning_residuals.numpy())
            np.save(Path(artifact_dir) / "calibration_states.npy", calibration_states.numpy())
            np.save(Path(artifact_dir) / "calibration_residuals.npy", calibration_residuals.numpy())
            np.save(Path(artifact_dir) / "conditioning_indices.npy", conditioning_indices)
            np.save(Path(artifact_dir) / "calibration_indices.npy", calibration_indices)
            self._save_conditioning_stats(Path(artifact_dir))

        # Record serial dependence for the primary conditioning mechanism when
        # it is cluster-based; diagnostics never alter interval widths.
        if hasattr(self.conformal, "diagnose_dependence"):
            exchange_results = self.conformal.diagnose_dependence(
                calibration_states, calibration_residuals, origin_indices=calibration_indices
            )
            if artifact_dir is not None:
                save_json(Path(artifact_dir) / "dependence_diagnostics.json", exchange_results)
            for k, value in exchange_results.items():
                if value.get("warning"):
                    print(f"  calibration warning | cluster {k}: {value['warning']}")
        self._set_train_mode(True)

    def _shared_calibration_indices(self, n_samples: int) -> np.ndarray:
        if n_samples <= 0:
            raise RuntimeError("Calibration loader produced no samples.")
        return np.arange(0, n_samples, self.args.calibration_stride, dtype=np.int64)

    @staticmethod
    def _split_calibration_indices(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if indices.size < 4:
            raise RuntimeError("Need at least four calibration origins for conditioning and quantile calibration.")
        split = indices.size // 2
        return indices[:split], indices[split:]

    def _predict_intervals(self, test_states: np.ndarray, preds: np.ndarray):
        """Dispatch to the calibrated conformal predictor's interval call.

        Isolated so subclasses (e.g. the flat_cp ablation arm, whose
        FlatConformal.predict() takes only point forecasts, no states) can
        override just this one call without touching the rest of test().
        """
        lower, upper = self.conformal.predict(
            torch.from_numpy(test_states).float(),
            torch.from_numpy(preds).float(),
        )
        return lower.numpy(), upper.numpy(), getattr(self.conformal, "last_predicted_clusters_", None)

    def _conditional_coverage_scores(self, test_states: np.ndarray):
        """Label-free latent-state magnitude used by every interval method."""
        return np.linalg.norm(np.asarray(test_states), axis=1)

    def _score_interval_comparator(
        self, lower_np, upper_np, trues, coverage_scope, test_states=None, interval_origin=None,
    ) -> dict:
        """Shared scoring for any calibrated comparator's already-built bounds."""
        coverage = compute_picp(lower_np, upper_np, trues)
        coverage_joint = compute_joint_picp(lower_np, upper_np, trues)
        primary = coverage_joint if coverage_scope == "simultaneous" else coverage
        result = {
            "coverage": float(coverage),
            "coverage_joint": float(coverage_joint),
            "coverage_primary": float(primary),
            "mean_width": float(compute_mpiw(lower_np, upper_np)),
            "winkler": float(winkler_score(lower_np, upper_np, trues, alpha=self.args.conformal_alpha)),
            "calibration_error": float(abs(primary - (1.0 - self.args.conformal_alpha))),
            "coverage_scope": coverage_scope,
        }
        if interval_origin is not None:
            result["interval_origin"] = interval_origin
        bin_edges = getattr(self, "_coverage_bin_edges", None)
        if test_states is not None and bin_edges is not None:
            scores = self._conditional_coverage_scores(test_states)
            if scores is not None:
                result["conditional_coverage"] = conditional_coverage_by_bin(
                    lower_np, upper_np, trues, scores, bin_edges, alpha=self.args.conformal_alpha
                )
        return result

    def _compare_against_flat_conformal(self, preds, trues, test_states=None) -> dict:
        """Score flat CP on the same forecasts, isolating the value of state conditioning.

        Every conditioning mechanism only earns its extra machinery if it
        beats a single global quantile. All calibrators are fitted on
        identical calibration residuals from the same model, so this
        comparison is paired: the only difference is the conditioning
        mechanism.

        This flat CP is calibrated on the SECOND HALF of the calibration
        split only, matching the conditional methods' quantile-fitting n for
        a fair paired comparison. That makes it a DIFFERENT estimator from
        run_baseline.py's flat CP (calibrated on the FULL calibration split)
        and from diagnose_conditioning_headroom.py's (a cut window) -- do not
        cross-reference a Winkler number from this dict against either. The
        interval_origin below is deliberately distinct from
        "conformalized" (run_baseline.py's label) for that reason.
        """
        flat = getattr(self, "flat_conformal", None)
        if flat is None or not flat.calibrated:
            return {}
        lower, upper = flat.predict(torch.from_numpy(preds).float())
        self._flat_per_origin_interval_scores = per_origin_interval_scores(
            lower.numpy(), upper.numpy(), trues, self.args.conformal_alpha
        )
        return self._score_interval_comparator(
            lower.numpy(), upper.numpy(), trues, flat.coverage_scope, test_states=test_states,
            interval_origin="conformalized_paired_half_cal",
        )

    def _compare_against_secondary_conformal(self, test_states, preds, trues) -> dict:
        """Score the non-primary state conditioning mode on the same forecasts.

        --conformal_conditioning selects which mode is primary (drives the
        headline interval/coverage_by_cluster fields); this reports the other
        mode paired against the same calibration residuals and test
        forecasts, so a run never has to be repeated to get both numbers.
        """
        secondary = getattr(self, "secondary_conformal", None)
        if secondary is None or not secondary.calibrated:
            return {}
        lower, upper = secondary.predict(
            torch.from_numpy(test_states).float(), torch.from_numpy(preds).float()
        )
        self._secondary_per_origin_interval_scores = per_origin_interval_scores(
            lower.numpy(), upper.numpy(), trues, self.args.conformal_alpha
        )
        result = self._score_interval_comparator(
            lower.numpy(), upper.numpy(), trues, secondary.coverage_scope, test_states=test_states
        )
        result["mode"] = "cluster" if isinstance(secondary, StateConditionalConformal) else "scale"
        return result

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        self._load_checkpoint(path)
        test_start = time.time()

        preds = []
        trues = []
        test_states = []

        self._set_train_mode(False)
        test_origin_stride = getattr(self.args, "test_origin_stride", 1)
        test_origin_indices = np.arange(0, len(test_data), test_origin_stride, dtype=np.int64)

        with torch.no_grad():
            if test_origin_stride > 1:
                n_windows = len(test_data)
                print(f"Test-origin stride evaluation | stride={test_origin_stride}")
                for i in track(
                    test_origin_indices,
                    description="Testing",
                    total=len(test_origin_indices),
                    enabled=not getattr(self.args, "no_progress", True),
                ):
                    bx, by, bxm, bym = test_data[i]
                    batch_x = torch.from_numpy(bx).unsqueeze(0)
                    batch_y = torch.from_numpy(by).unsqueeze(0)
                    final_state, outputs, batch_y = self._forward_and_slice(batch_x, batch_y)
                    preds.append(outputs.detach().cpu().numpy())
                    trues.append(batch_y.detach().cpu().numpy())
                    test_states.append(final_state.detach().cpu().numpy())
            else:
                for batch_x, batch_y, _batch_x_mark, _batch_y_mark in track(
                    test_loader,
                    description="Testing",
                    total=len(test_loader),
                    enabled=not getattr(self.args, "no_progress", True),
                ):
                    final_state, outputs, batch_y = self._forward_and_slice(batch_x, batch_y)
                    preds.append(outputs.detach().cpu().numpy())
                    trues.append(batch_y.detach().cpu().numpy())
                    test_states.append(final_state.detach().cpu().numpy())

            preds = self._concatenate_batches(preds, 'prediction')
            trues = self._concatenate_batches(trues, 'target')
            test_states = self._concatenate_batches(test_states, 'state')

        history_path = Path(self.args.checkpoints) / setting / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else None

        mae = mean_absolute_error(trues.flatten(), preds.flatten())
        mse = mean_squared_error(trues.flatten(), preds.flatten())
        rmse = np.sqrt(mse)

        coverage = None
        coverage_joint = None
        coverage_primary = None
        mean_width = None
        winkler = None
        calib_err = None
        msis = None
        cluster_labels = None
        coverage_by_cluster = {}
        flat_comparison = {}
        secondary_comparison = {}
        self._flat_per_origin_interval_scores = None
        self._secondary_per_origin_interval_scores = None
        conditional_coverage = None
        if hasattr(self, 'conformal') and self.conformal.calibrated:
            lower_np, upper_np, cluster_labels = self._predict_intervals(test_states, preds)
            coverage = compute_picp(lower_np, upper_np, trues)
            coverage_joint = compute_joint_picp(lower_np, upper_np, trues)
            mean_width = compute_mpiw(lower_np, upper_np)
            winkler = winkler_score(lower_np, upper_np, trues, alpha=self.args.conformal_alpha)
            # A simultaneous-coverage strategy (e.g. multivariate_strategy='max')
            # collapses all H*C residuals to one scalar before taking the
            # quantile, so marginal PICP saturates near 1.0 by construction --
            # that is not a miscalibration, it's the wrong metric for what the
            # method guarantees. Score against whichever coverage the fitted
            # strategy actually promises (coverage_joint for simultaneous
            # scopes, marginal coverage otherwise).
            coverage_primary = coverage_joint if self.conformal.coverage_scope == "simultaneous" else coverage
            calib_err = abs(coverage_primary - (1.0 - self.args.conformal_alpha))
            if cluster_labels is not None:
                coverage_by_cluster = self._coverage_by_cluster(lower_np, upper_np, trues, cluster_labels)
            conditional_coverage = None
            bin_edges = getattr(self, "_coverage_bin_edges", None)
            if bin_edges is not None:
                primary_scores = self._conditional_coverage_scores(test_states)
                if primary_scores is not None:
                    conditional_coverage = conditional_coverage_by_bin(
                        lower_np, upper_np, trues, primary_scores, bin_edges, alpha=self.args.conformal_alpha
                    )
            flat_comparison = self._compare_against_flat_conformal(preds, trues, test_states=test_states)
            secondary_comparison = self._compare_against_secondary_conformal(test_states, preds, trues)
            train_data, _ = self._get_data(flag='train')
            seasonal_period = get_dataset_spec(self.args.data)["seasonal_period"]
            train_targets = train_data.data_y
            if self.args.features == "MS":
                train_targets = train_targets[:, -1:]
            msis = mean_scaled_interval_score(
                lower_np, upper_np, trues, train_targets, seasonal_period, alpha=self.args.conformal_alpha
            )
            print(
                f"Intervals | coverage@{(1.0 - self.args.conformal_alpha) * 100:.0f}%={coverage:.4f} | "
                f"joint={coverage_joint:.4f} | width={mean_width:.4f} | winkler={winkler:.4f}"
            )
            if flat_comparison:
                delta = winkler - flat_comparison["winkler"]
                verdict = "primary better" if delta < 0 else "flat CP better or equal"
                print(
                    f"  flat CP  | coverage={flat_comparison['coverage']:.4f} | "
                    f"width={flat_comparison['mean_width']:.4f} | "
                    f"winkler={flat_comparison['winkler']:.4f}"
                )
                print(f"  state conditioning vs flat | winkler delta={delta:+.4f} -> {verdict}")
            if secondary_comparison:
                delta2 = winkler - secondary_comparison["winkler"]
                verdict2 = "primary better" if delta2 < 0 else "secondary better or equal"
                print(
                    f"  secondary ({secondary_comparison['mode']}) | coverage={secondary_comparison['coverage']:.4f} | "
                    f"width={secondary_comparison['mean_width']:.4f} | "
                    f"winkler={secondary_comparison['winkler']:.4f}"
                )
                print(f"  primary vs secondary | winkler delta={delta2:+.4f} -> {verdict2}")
        else:
            lower_np = np.full_like(preds, np.nan)
            upper_np = np.full_like(preds, np.nan)

        test_dispersion = dispersion_summary(preds, trues)
        variance_ratio = test_dispersion["variance_ratio"]
        dispersion_note = "" if variance_ratio is None else (
            f" | var_ratio={variance_ratio:.4f} | corr={test_dispersion['corr']:.4f}"
        )
        print(f"Point forecast | mse={mse:.6f} | mae={mae:.6f} | rmse={rmse:.6f}{dispersion_note}")

        # Structural validity is checked against the interval bounds too, so it
        # runs after they exist. Quality references come from the training split
        # only -- never from test statistics.
        train_data_for_ref, _ = self._get_data(flag='train')
        sanity_report = check_forecast_sanity(
            preds,
            trues,
            history=history,
            lower=lower_np,
            upper=upper_np,
            y_train=train_data_for_ref.data_y,
            seasonal_period=get_dataset_spec(self.args.data)["seasonal_period"],
            horizon=self.args.pred_len,
        )
        for msg in sanity_report["failures"]:
            print(f"Result review | structural failure: {msg}")
        for msg in sanity_report["warnings"]:
            print(f"Result review | quality note: {msg}")

        folder_path = Path(self.args.results_dir) / setting
        folder_path.mkdir(parents=True, exist_ok=True)

        conditioning_mode = getattr(self.args, "conformal_conditioning", "cluster")
        point_metrics = {"mae": mae, "mse": mse, "rmse": rmse, **test_dispersion}
        interval_metrics = {
            "coverage": coverage if coverage is not None else None,
            "coverage_joint": coverage_joint if coverage_joint is not None else None,
            "coverage_primary": coverage_primary if coverage_primary is not None else None,
            "mean_width": mean_width if mean_width is not None else None,
            "winkler": winkler if winkler is not None else None,
            "calibration_error": calib_err if calib_err is not None else None,
            "msis": msis if msis is not None else None,
            "alpha": self.args.conformal_alpha,
            "coverage_scope": getattr(self.conformal, "coverage_scope", "marginal") if hasattr(self, "conformal") else None,
            "conditioning_mode": conditioning_mode,
            "conditional_coverage": conditional_coverage,
            "interval_origin": "conformalized",
            "units": "z-scored (per-feature train-split standardization)",
        }
        # Mode-tagged keys so a downstream reader always finds cluster and
        # scale results under fixed names regardless of which was primary for
        # this run -- interval_metrics/secondary_comparison already share one
        # underlying score, this only relabels which dict holds which.
        cluster_result = interval_metrics if conditioning_mode == "cluster" else secondary_comparison
        scaled_result = secondary_comparison if conditioning_mode == "cluster" else interval_metrics
        metrics_payload = {
            "setting": setting,
            "point": point_metrics,
            "interval": interval_metrics,
            # Paired comparators on identical forecasts and calibration
            # residuals: the evidence for or against state conditioning, and
            # for the continuous-scale mode over the discrete-cluster mode.
            "interval_flat_cp": flat_comparison,
            "interval_cluster_cp": cluster_result,
            "interval_state_scaled": scaled_result,
            "sanity_passed": sanity_report["passed"],
            "structural_passed": sanity_report["structural_passed"],
            "quality_flags": sanity_report["warnings"],
        }

        np.save(folder_path / 'metrics.npy', np.array([mae, mse, rmse]))
        np.save(folder_path / 'conformal.npy', np.array([coverage if coverage is not None else -1,
                                                          mean_width if mean_width is not None else -1,
                                                          winkler if winkler is not None else -1]))
        np.save(folder_path / 'pred.npy', preds)
        np.save(folder_path / 'true.npy', trues)
        np.save(folder_path / 'lower.npy', lower_np)
        np.save(folder_path / 'upper.npy', upper_np)
        np.save(folder_path / 'states.npy', test_states)
        np.save(folder_path / 'residuals.npy', np.abs(preds - trues))
        np.save(folder_path / "test_origin_indices.npy", test_origin_indices)
        per_origin = per_origin_interval_scores(lower_np, upper_np, trues, self.args.conformal_alpha)
        np.save(folder_path / "per_origin_interval_metrics.npy", per_origin)
        flat_per_origin = getattr(self, "_flat_per_origin_interval_scores", None)
        if flat_per_origin is None:
            flat_per_origin = np.full((len(preds), 3), np.nan, dtype=np.float64)
        np.save(folder_path / "per_origin_interval_flat_cp_metrics.npy", flat_per_origin)
        if cluster_labels is not None:
            np.save(folder_path / 'cluster_labels.npy', cluster_labels)

        save_json(folder_path / "metrics.json", metrics_payload)
        save_json(folder_path / "sanity.json", sanity_report)
        save_json(folder_path / "coverage_by_cluster.json", coverage_by_cluster)
        save_json(folder_path / "config.json", vars(self.args))
        save_json(folder_path / "environment.json", environment_snapshot(self.device))
        if history is not None:
            save_json(folder_path / "history.json", history)
        runtime = dict(getattr(self, "train_runtime_", {}))
        runtime["test_seconds"] = time.time() - test_start
        runtime["test_samples"] = len(test_data)
        runtime["test_origins_evaluated"] = int(preds.shape[0])
        runtime["test_origin_stride"] = int(test_origin_stride)
        save_json(folder_path / "runtime.json", runtime)
        save_json(folder_path / "protocol.json", self.args.protocol)
        self._save_conditioning_stats(folder_path)
        if getattr(self.args, "immutable_artifacts", False):
            write_completion_manifest(
                folder_path,
                [
                    "metrics.json", "sanity.json", "config.json", "environment.json", "protocol.json",
                    "runtime.json", "history.json", "pred.npy", "true.npy", "lower.npy", "upper.npy",
                    "states.npy", "residuals.npy", "test_origin_indices.npy", "per_origin_interval_metrics.npy",
                    "per_origin_interval_flat_cp_metrics.npy",
                ],
                self.args.protocol,
            )

        return sanity_report


class HybridExperiment(Experiment):
    """Two-stage hybrid training: fit DLinear, freeze it, then fit the correction.

    Stage 1 trains the DLinear base alone with the standard budget and restores
    its best validation checkpoint. Stage 2 freezes that base and trains only
    the state encoder and correction head.

    Because the correction head is zero-initialised, stage-2 epoch 0 reproduces
    the frozen base exactly. That epoch-0 state is saved as a fallback before
    any correction step runs, so a correction stage that fails to improve
    validation loss degrades to DLinear rather than to something worse.

    ``self.model``/``self.head`` are bound to the encoder and correction head so
    every inherited method (partition fitting, calibration, interval prediction,
    checkpoint IO) keeps working unchanged. Only the forward pass differs, and
    that is overridden in ``_forward_and_slice``.
    """

    def __init__(self, args):
        self.args = args
        self.device = select_device(require_gpu=getattr(args, 'require_gpu', False))
        self.hybrid = self._build_hybrid().to(self.device)
        # Bind to the inherited names so base-class machinery is reused as-is.
        self.model = self.hybrid.encoder
        self.head = self.hybrid.correction
        # The hybrid handles instance normalisation internally via --state_revin;
        # full-model RevIN is a legacy-architecture option only.
        self.revin = None

    def _build_hybrid(self) -> HybridCISSN:
        return HybridCISSN(
            input_dim=self.args.enc_in,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            output_dim=self.args.c_out,
            state_dim=self.args.state_dim,
            hidden_dim=self.args.d_model,
            dropout=self.args.dropout,
            state_revin=getattr(self.args, 'state_revin', False),
            state_dynamics=getattr(self.args, 'state_dynamics', 'legacy'),
            seasonal_period=get_dataset_spec(self.args.data)["seasonal_period"],
        )

    def _forward_and_slice(self, batch_x, batch_y, return_all_states=False):
        batch_x = batch_x.float().to(self.device, non_blocking=True)
        batch_y = batch_y.float().to(self.device, non_blocking=True)

        # forward_components() already returns the state it used, so the forecast
        # and the state come from a single encoder pass rather than two.
        parts = self.hybrid.forward_components(batch_x)
        final_state = parts["state"]
        outputs = parts["total"]

        # The per-timestep state sequence is only needed for the disentanglement
        # penalty, and only the encoder can produce it.
        all_states = (
            self.hybrid.encoder(batch_x, return_all_states=True)
            if return_all_states
            else None
        )

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

        if return_all_states:
            return all_states, final_state, outputs, batch_y
        return final_state, outputs, batch_y

    def _select_optimizer(self):
        # Stage 2 only ever updates the correction branch; the frozen base is
        # excluded from the optimizer as well as from autograd.
        return optim.Adam(self.hybrid.correction_parameters(), lr=self.args.learning_rate)

    def _select_disentangle_criterion(self):
        # The hybrid defaults to no covariance/temporal penalty: those terms were
        # introduced to shape a state that carried the whole forecast, and here
        # the state only carries a correction. Legacy reproduction can re-enable
        # them by passing non-zero lambdas explicitly.
        if self.args.lambda_cov == 0 and self.args.lambda_temp == 0:
            return None
        return super()._select_disentangle_criterion()

    def _base_checkpoint_path(self, path) -> Path:
        return Path(path) / "checkpoint_base.pth"

    def _train_base(self, train_loader, vali_loader, path) -> list[dict]:
        """Stage 1: train the DLinear base alone and restore its best checkpoint."""
        base = self.hybrid.base
        optimizer = optim.Adam(base.parameters(), lr=self.args.learning_rate)
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        best_path = self._base_checkpoint_path(path)
        history: list[dict] = []

        for epoch in range(self.args.train_epochs):
            base.train()
            epoch_time = time.time()
            total_loss = torch.zeros((), device=self.device)
            total_weight = 0

            for batch_x, batch_y, _bxm, _bym in track(
                train_loader,
                description=f"Base epoch {epoch + 1}/{self.args.train_epochs}",
                total=len(train_loader),
                enabled=not getattr(self.args, "no_progress", True),
            ):
                optimizer.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = base(batch_x)[:, -self.args.pred_len:, f_dim:]
                targets = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, targets)
                loss.backward()
                if self.args.grad_clip and self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(base.parameters(), max_norm=self.args.grad_clip)
                optimizer.step()

                total_loss += loss.detach() * outputs.numel()
                total_weight += outputs.numel()

            if total_weight == 0:
                raise RuntimeError("Training loader produced no prediction elements.")
            train_loss = float((total_loss / total_weight).item())
            vali_loss = self._vali_base(vali_loader, criterion)

            improved = early_stopping(vali_loss, base, None, str(path))
            if improved:
                torch.save(base.state_dict(), best_path)
            history.append({
                "stage": "base",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "vali_loss": vali_loss,
                "lr": optimizer.param_groups[0]['lr'],
            })
            print_epoch_summary(
                epoch=epoch + 1, total_epochs=self.args.train_epochs, train_loss=train_loss,
                validation_loss=vali_loss, learning_rate=optimizer.param_groups[0]['lr'],
                elapsed_seconds=time.time() - epoch_time, improved=improved,
                patience_counter=early_stopping.counter, patience=early_stopping.patience,
            )
            if early_stopping.early_stop:
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        if best_path.exists():
            base.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
        return history

    def _vali_base(self, vali_loader, criterion) -> float:
        base = self.hybrid.base
        base.eval()
        total_loss = 0.0
        total_weight = 0
        f_dim = -1 if self.args.features == 'MS' else 0
        with torch.no_grad():
            for batch_x, batch_y, _bxm, _bym in vali_loader:
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                outputs = base(batch_x)[:, -self.args.pred_len:, f_dim:]
                targets = batch_y[:, -self.args.pred_len:, f_dim:]
                total_loss += criterion(outputs, targets).item() * outputs.numel()
                total_weight += outputs.numel()
        if total_weight == 0:
            raise RuntimeError("Validation loader produced no prediction elements.")
        base.train()
        return total_loss / total_weight

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        cal_data, cal_loader = self._get_data(flag='cal')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        self.checkpoint_path = path
        save_json(Path(path) / "config.json", vars(self.args))
        save_json(Path(path) / "environment.json", environment_snapshot(self.device))

        train_start = time.time()
        print("\n  Stage 1/2 | DLinear base")
        history = self._train_base(train_loader, vali_loader, path)

        print("\n  Stage 2/2 | five-state correction (base frozen)")
        self.hybrid.freeze_base()
        criterion = self._select_criterion()

        # Save the epoch-0 fallback BEFORE any correction step. The correction
        # head is still zero here, so this checkpoint is exactly the frozen
        # DLinear base and is what a failed correction stage falls back to.
        base_vali_loss = self.vali(vali_loader, criterion)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        early_stopping(base_vali_loss, self.model, self.head, path)
        history.append({
            "stage": "correction",
            "epoch": 0,
            "train_loss": None,
            "vali_loss": base_vali_loss,
            "lr": self.args.learning_rate,
            "note": "zero-initialised correction; identical to frozen DLinear base",
        })
        print(f"  epoch 0 (frozen base) | vali_loss={base_vali_loss:.7f}")

        model_optim = self._select_optimizer()
        disentangle_criterion = self._select_disentangle_criterion()
        correction_parameters = self.hybrid.correction_parameters()

        for epoch in range(self.args.train_epochs):
            total_train_loss = torch.zeros((), device=self.device)
            total_forecast_loss = torch.zeros((), device=self.device)
            total_train_weight = 0
            epoch_states = []
            epoch_final_states = []

            self.hybrid.train()
            epoch_time = time.time()

            for batch_x, batch_y, _bxm, _bym in track(
                train_loader,
                description=f"Correction epoch {epoch + 1}/{self.args.train_epochs}",
                total=len(train_loader),
                enabled=not getattr(self.args, "no_progress", True),
            ):
                model_optim.zero_grad(set_to_none=True)
                states, final_state, outputs, targets = self._forward_and_slice(
                    batch_x, batch_y, return_all_states=True
                )
                forecast_loss = criterion(outputs, targets)
                loss = forecast_loss
                if disentangle_criterion is not None:
                    loss = loss + disentangle_criterion(states)

                batch_weight = outputs.numel()
                total_train_loss += loss.detach() * batch_weight
                total_forecast_loss += forecast_loss.detach() * batch_weight
                total_train_weight += batch_weight
                epoch_states.append(states.detach())
                epoch_final_states.append(final_state.detach())

                loss.backward()
                if self.args.grad_clip and self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(correction_parameters, max_norm=self.args.grad_clip)
                model_optim.step()

            if total_train_weight == 0:
                raise RuntimeError("Training loader produced no prediction elements.")
            train_loss = float((total_train_loss / total_train_weight).item())
            forecast_loss = float((total_forecast_loss / total_train_weight).item())
            vali_loss = self.vali(vali_loader, criterion)

            disent_metrics = DisentanglementLoss.get_metrics(torch.cat(epoch_states, dim=0))
            improved = early_stopping(vali_loss, self.model, self.head, path)
            history.append({
                "stage": "correction",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "forecast_loss": forecast_loss,
                "vali_loss": vali_loss,
                "lr": model_optim.param_groups[0]['lr'],
                "off_diag_corr": disent_metrics["mean_abs_off_diag_corr"],
            })
            print_epoch_summary(
                epoch=epoch + 1, total_epochs=self.args.train_epochs, train_loss=train_loss,
                validation_loss=vali_loss, learning_rate=model_optim.param_groups[0]['lr'],
                elapsed_seconds=time.time() - epoch_time, improved=improved,
                patience_counter=early_stopping.counter, patience=early_stopping.patience,
            )
            if early_stopping.early_stop:
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_correction_loss = float(early_stopping.val_loss_min)
        fell_back = best_correction_loss >= base_vali_loss
        if fell_back:
            print(
                "  correction stage did not improve on the frozen base; "
                "restoring the epoch-0 (DLinear-equivalent) checkpoint."
            )
        save_json(Path(path) / "history.json", history)

        self._load_checkpoint(path)
        calibration_start = time.time()
        self._build_conditioning_predictors()
        self._calibrate_conformal(cal_loader, path)
        self.train_runtime_ = {
            "train_seconds": time.time() - train_start,
            "calibration_seconds": time.time() - calibration_start,
            "train_samples": len(train_data),
            "validation_samples": len(vali_data),
            "calibration_samples": len(cal_data),
            "epochs_requested": self.args.train_epochs,
            "epochs_run": len(history),
            "early_stopped": early_stopping.early_stop,
            "best_val_loss": best_correction_loss,
            "base_val_loss": base_vali_loss,
            "correction_improved_on_base": not fell_back,
        }
        save_json(Path(path) / "runtime.json", self.train_runtime_)
        return self.hybrid

    def _load_checkpoint(self, path):
        """Restore the base plus the best correction-stage checkpoint."""
        base_path = self._base_checkpoint_path(path)
        if base_path.exists():
            self.hybrid.base.load_state_dict(
                torch.load(base_path, map_location=self.device, weights_only=True)
            )
        super()._load_checkpoint(path)

    def vali(self, vali_loader, criterion):
        loss = super().vali(vali_loader, criterion)
        # The parent restores train() on both submodules; re-assert the freeze so
        # a frozen base cannot silently re-enter train mode mid-run.
        self.hybrid.train()
        return loss


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust learning rate according to the chosen schedule.

    Supported policies (--lradj):
        type1   — halve LR every epoch: lr * 0.5^(epoch-1)
        type2   — fixed milestone schedule (hardcoded for up to 20 epochs)
        cosine  — cosine annealing over train_epochs; requires args.train_epochs
    """
    if args.lradj == 'type1':
        lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8,
        }
        if epoch in lr_adjust:
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    elif args.lradj == 'cosine':
        # Cosine annealing: smoothly decays to 0 over train_epochs.
        # eta_min is set to 1% of the initial LR.
        lr = args.learning_rate * 0.5 * (
            1.0 + np.cos(np.pi * epoch / args.train_epochs)
        )
        lr = max(lr, args.learning_rate * 0.01)  # floor at 1% of initial
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        raise ValueError(f"Unknown lradj policy: {args.lradj!r}. Use 'type1', 'type2', or 'cosine'.")

    return optimizer.param_groups[0]['lr']

def parse_args(argv: Optional[list[str]] = None):
    cli_argv = argv if argv is not None else sys.argv[1:]

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None, help='YAML/JSON config file')
    pre_args, _ = pre_parser.parse_known_args(args=cli_argv)
    config_defaults = load_config_defaults(pre_args.config)
    cli_options = provided_cli_options(cli_argv)

    parser = argparse.ArgumentParser(description='CISSN Benchmark Runner', parents=[pre_parser])
    parser.add_argument('--data', type=str, default='ETTh1', choices=supported_datasets(), help='dataset name')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='data root directory')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data filename')
    parser.add_argument('--features', type=str, default='M', help='forecasting task [M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature for S/MS tasks')
    parser.add_argument('--freq', type=str, default='h', help='time feature encoding frequency')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='checkpoint directory')
    parser.add_argument('--results_dir', type=str, default='./results/', help='results directory')

    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='decoder start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction horizon')

    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--architecture', type=str, default='legacy', choices=['legacy', 'hybrid'],
                        help="'legacy' encodes history through the 5-d state only; "
                             "'hybrid' forecasts with DLinear and adds a five-state correction")
    parser.add_argument('--state_dynamics', type=str, default='legacy', choices=['legacy', 'anchored'],
                        help='state transition parameterisation for the hybrid correction branch')
    parser.add_argument('--state_revin', action='store_true',
                        help='apply reversible instance norm to the state branch only (hybrid)')
    parser.add_argument('--revin', action='store_true',
                        help='apply reversible instance norm to the whole model: the network sees a '
                             'per-window standardised input and predicts shape only, removing the '
                             'level-tracking burden that drives mean-shrinkage under MSE')
    parser.add_argument('--d_model', type=int, default=64, help='model hidden dimension')
    parser.add_argument('--state_dim', type=int, default=5, help='latent state dimension')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate')
    parser.add_argument('--lambda_cov', type=float, default=1.0, help='covariance loss weight')
    parser.add_argument('--lambda_temp', type=float, default=0.5, help='temporal consistency loss weight')
    parser.add_argument('--lambda_correction_scale', type=float, default=0.0, help='penalty weight keeping encoder correction scale near 0.01')
    parser.add_argument('--no_refinement', action='store_true',
                        help='drop the forecast-head refinement MLP, leaving only the interpretable '
                             'linear state->forecast map (removes ~77%% of model parameters)')
    parser.add_argument('--lambda_refinement', type=float, default=0.0,
                        help='penalty on the forecast head refinement scale; keeps the interpretable '
                             'linear path dominant over the non-attributable MLP path')

    parser.add_argument('--num_workers', type=int, default=0, help='dataloader workers')
    parser.add_argument('--require_gpu', action='store_true',
                        help='fail instead of falling back to CPU when no GPU is available')
    parser.add_argument('--require_clean_git', action='store_true',
                        help='require a clean committed worktree for a publication run')
    parser.add_argument('--no_progress', action='store_true',
                        help='disable terminal progress bars for CI or captured logs')
    parser.add_argument('--strict_artifacts', action='store_true',
                        help='exit nonzero when a run produces structurally invalid artifacts')
    parser.add_argument('--strict_determinism', action='store_true',
                        help='require deterministic PyTorch algorithms and record the setting')
    parser.add_argument('--immutable_artifacts', action='store_true',
                        help='write a content-addressed run and refuse existing artifacts')
    parser.add_argument('--evidence_role', choices=['development', 'selection', 'confirmation'],
                        default='development',
                        help='evidence tier recorded in protocol; confirmation enforces sealed-run safeguards')
    # Retired alias: quality never excluded a run, so this now maps to the
    # structural check. Accepted silently for existing scripts.
    parser.add_argument('--strict_sanity', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--train_epochs', type=int, default=20, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (128 keeps GPU step time amortised without starving '
                             'the optimiser of updates; see CLAUDE.md)')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lradj', type=str, default='cosine',
                        help="lr schedule [type1, type2, cosine]. Default 'cosine' anneals over the "
                             "full --train_epochs budget so early stopping (not LR decay) ends training; "
                             "'type1' halves every epoch and collapses training if train_epochs > ~6.")
    parser.add_argument('--grad_clip', type=float, default=1.0, help='max gradient norm; <=0 disables clipping')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # New arguments for improvements
    parser.add_argument('--test_origin_stride', type=int, default=1,
                        help='evaluate every kth fixed test origin; does not refit or update the model')
    parser.add_argument('--walk_forward', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--conformal_alpha', type=float, default=0.1, help='conformal significance level')
    parser.add_argument('--n_clusters', type=int, default=5, help='requested SCCP clusters')
    parser.add_argument('--multivariate_strategy', choices=['per_feature', 'max'], default='per_feature', help='Conformal strategy [per_feature, max]')
    parser.add_argument('--conformal_conditioning', type=str, default='cluster', choices=['cluster', 'scale'],
                        help="primary state conditioning mechanism: 'cluster' calibrates one quantile per "
                             "K-Means state cluster (StateConditionalConformal); 'scale' calibrates a single "
                             "quantile on residuals normalized by a continuous log-linear scale sigma(state) "
                             "(StateScaledConformal). Every run calibrates and reports both, paired against the "
                             "same forecasts; this flag only selects which one drives the primary 'interval' "
                             "block and coverage_by_cluster.json.")
    parser.add_argument('--scale_geometry', type=str, default='scalar', choices=['scalar', 'per_cell'],
                        help="shape of the state-scaled predictor's sigma(state): 'scalar' fits one scale per "
                             "sample (default, original behaviour); 'per_cell' fits one scale per horizon-feature "
                             "cell, letting the state reshape the quantile surface instead of only rescaling its "
                             "level. Development measurements on ETTh1-h336 RevIN runs favour 'per_cell' "
                             "(-0.237 Winkler vs flat CP, 12/12 seed-cut wins, against +0.011 for 'scalar'); "
                             "see docs/methodology.md. Ignored under --multivariate_strategy max.")
    parser.add_argument('--calibration_stride', type=int, default=1,
                        help='keep every kth chronological calibration origin for dependence-aware calibration')
    parser.add_argument('--cal_fraction', type=float, default=0.2,
                        help='fraction of the canonical train window carved out as the calibration split')
    validate_config_defaults(parser, config_defaults)
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(args=cli_argv)
    # --strict_sanity is a deprecated alias for --strict_artifacts.
    if vars(args).pop("strict_sanity", False):
        args.strict_artifacts = True
    if args.walk_forward:
        if "test_origin_stride" in cli_options:
            raise ValueError("Use --test_origin_stride alone; --walk_forward is a deprecated alias.")
        warnings.warn("--walk_forward is deprecated; using --test_origin_stride pred_len.", DeprecationWarning)
        args.test_origin_stride = args.pred_len
    protected = set(config_defaults) | cli_options
    apply_dataset_defaults(args, protected)
    if args.features == 'MS' and 'c_out' not in protected:
        args.c_out = 1
    validate_runtime_args(args)
    if args.test_origin_stride <= 0:
        raise ValueError("--test_origin_stride must be positive.")

    return args


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    enforce_evidence_contract(args)
    require_clean_source(args)
    set_random_seed(args.seed, strict=args.strict_determinism)
    args.protocol = build_protocol_manifest(args)

    setting = build_run_setting(args, build_setting_name(args))
    final_results_dir = None
    if args.immutable_artifacts:
        require_new_run(Path(args.checkpoints) / setting, Path(args.results_dir) / setting)
        final_results_dir = Path(args.results_dir)
        args.results_dir = str(create_temporary_result_root(final_results_dir))
    print_run_header("CISSN benchmark", args, setting)
    
    exp = HybridExperiment(args) if args.architecture == 'hybrid' else Experiment(args)
    print("\n[1/2] Training and calibration")
    exp.train(setting)
    print("\n[2/2] Test evaluation")
    sanity_report = exp.test(setting)
    if final_results_dir is not None:
        finalized = finalize_result_directory(args.results_dir, final_results_dir, setting)
        print(f"Finalized immutable artifacts at {finalized}")

    # Only structural invalidity is fatal. A poor-but-well-formed forecast is a
    # valid result and must still exit zero so it stays publication-visible.
    if args.strict_artifacts and not sanity_report["structural_passed"]:
        print("Structural artifact validation failed; see sanity.json.", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
