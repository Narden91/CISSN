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
from cissn.losses.disentangle_loss import DisentanglementLoss
from cissn.conformal import StateConditionalConformal
from cissn.data.data_loader import get_data_loader
from cissn.data.registry import get_dataset_spec, supported_datasets
from cissn.utils import EarlyStopping
from cissn.evaluation.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error,
    compute_picp, compute_mpiw, winkler_score,
    calibration_error,
)


def _format_float_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def build_setting_name(args) -> str:
    return (
        f"CISSN_{args.data}_{args.features}"
        f"_sl{args.seq_len}_pl{args.pred_len}_sd{args.state_dim}_dm{args.d_model}"
        f"_lc{_format_float_token(args.lambda_cov)}_lt{_format_float_token(args.lambda_temp)}"
        f"_a{_format_float_token(args.conformal_alpha)}_{args.multivariate_strategy}"
        f"_seed{args.seed}"
    )


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


def apply_dataset_defaults(args, protected_keys: set[str]) -> None:
    spec = get_dataset_spec(args.data)
    for key in ("root_path", "data_path", "freq", "target", "enc_in", "c_out"):
        if key not in protected_keys:
            setattr(args, key, spec[key])


def environment_snapshot(device: torch.device) -> dict:
    def _git_value(command: list[str]) -> Optional[str]:
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(["git", "status", "--short"])),
    }


class Experiment:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)
        self.head = self._build_head().to(self.device)
        print(f"Model and Head initialized on {self.device}")

    def _build_model(self):
        return DisentangledStateEncoder(
            input_dim=self.args.enc_in,
            state_dim=self.args.state_dim,
            hidden_dim=self.args.d_model,
            dropout=self.args.dropout
        )

    def _build_head(self):
        return ForecastHead(
            state_dim=self.args.state_dim,
            output_dim=self.args.c_out,
            horizon=self.args.pred_len,
            hidden_dim=self.args.d_model // 2
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

        if return_all_states:
            all_states = self.model(batch_x, return_all_states=True)
            final_state = all_states[:, -1, :]
        else:
            final_state = self.model(batch_x)
            all_states = None

        outputs = self.head(final_state)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

        if return_all_states:
            return all_states, final_state, outputs, batch_y
        return final_state, outputs, batch_y

    def _select_optimizer(self):
        params = list(self.model.parameters()) + list(self.head.parameters())
        return optim.Adam(params, lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    @staticmethod
    def _concatenate_batches(batches, name):
        if not batches:
            raise RuntimeError(f"No {name} batches were produced.")
        return np.concatenate(batches, axis=0)

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

        time_now = time.time()
        train_start = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        disentangle_criterion = DisentanglementLoss(
            lambda_cov=self.args.lambda_cov,
            lambda_temporal=self.args.lambda_temp
        ).to(self.device)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            self.head.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                states, final_state, outputs, batch_y = self._forward_and_slice(
                    batch_x, batch_y, return_all_states=True
                )

                loss = criterion(outputs, batch_y) + disentangle_criterion(states)
                if self.args.lambda_correction_scale > 0 and hasattr(self.model, "_correction_scale"):
                    target_scale = torch.tensor(0.01, device=self.device, dtype=outputs.dtype)
                    loss = loss + self.args.lambda_correction_scale * (self.model._correction_scale() - target_scale) ** 2
                train_loss.append(loss.item())

                loss.backward()
                if self.args.grad_clip and self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.head.parameters()),
                        max_norm=self.args.grad_clip,
                    )
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.2f}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)

            disent_metrics = disentangle_criterion.get_metrics(states)
            refinement_ratio = self.head.get_refinement_ratio(final_state)

            print(f"Epoch: {epoch+1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
            print(f"  Disentanglement: off_diag_corr={disent_metrics['mean_abs_off_diag_corr']:.4f} | per_dim_var={[f'{v:.4f}' for v in disent_metrics['per_dim_variance']]}")
            print(f"  Refinement ratio: {refinement_ratio:.4f} ({'linear dominates' if refinement_ratio < 0.5 else 'refinement dominates'})")

            if self.args.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "vali_loss": vali_loss,
                    "lr": model_optim.param_groups[0]['lr'],
                    "disent_off_diag_corr": disent_metrics["mean_abs_off_diag_corr"],
                    "refinement_ratio": refinement_ratio,
                })

            early_stopping(vali_loss, self.model, self.head, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self._load_checkpoint(path)
        calibration_start = time.time()
        self._calibrate_conformal(cal_loader, path)
        self.train_runtime_ = {
            "train_seconds": time.time() - train_start,
            "calibration_seconds": time.time() - calibration_start,
            "train_samples": len(train_data),
            "validation_samples": len(vali_data),
            "calibration_samples": len(cal_data),
        }
        save_json(Path(path) / "runtime.json", self.train_runtime_)
        return self.model

    def _load_checkpoint(self, path):
        self.model.load_state_dict(
            torch.load(os.path.join(path, 'checkpoint.pth'), map_location=self.device, weights_only=True)
        )
        self.head.load_state_dict(
            torch.load(os.path.join(path, 'checkpoint_head.pth'), map_location=self.device, weights_only=True)
        )

    def vali(self, vali_loader, criterion):
        total_loss = 0.0
        total_weight = 0
        self.model.eval()
        self.head.eval()
        with torch.no_grad():
            for batch_x, batch_y, _batch_x_mark, _batch_y_mark in vali_loader:
                _, outputs, batch_y = self._forward_and_slice(batch_x, batch_y)
                batch_weight = outputs.numel()
                total_loss += criterion(outputs, batch_y).item() * batch_weight
                total_weight += batch_weight
            if total_weight == 0:
                raise RuntimeError("Validation loader produced no prediction elements.")
        self.model.train()
        self.head.train()
        return total_loss / total_weight

    def _calibrate_conformal(self, cal_loader, artifact_dir=None):
        """Calibrate the StateConditionalConformal predictor on the held-out calibration split."""
        self.conformal = StateConditionalConformal(
            alpha=self.args.conformal_alpha,
            n_clusters=self.args.n_clusters,
            multivariate_strategy=self.args.multivariate_strategy,
            random_state=self.args.seed,
        )
        all_states = []
        all_residuals = []

        self.model.eval()
        self.head.eval()
        with torch.no_grad():
            for batch_x, batch_y, _batch_x_mark, _batch_y_mark in cal_loader:
                final_state, outputs, batch_y = self._forward_and_slice(batch_x, batch_y)
                all_states.append(final_state.detach().cpu())
                all_residuals.append((outputs - batch_y).abs().detach().cpu())

        all_states = torch.cat(all_states, dim=0)
        all_residuals = torch.cat(all_residuals, dim=0)
        self.conformal.fit(all_states, all_residuals)
        print("Conformal predictor calibrated on held-out calibration split.")

        if artifact_dir is not None:
            np.save(Path(artifact_dir) / "calibration_states.npy", all_states.numpy())
            np.save(Path(artifact_dir) / "calibration_residuals.npy", all_residuals.numpy())
            save_json(Path(artifact_dir) / "cluster_stats.json", self.conformal.get_cluster_stats())

        # Check exchangeability assumption
        exchange_results = self.conformal.check_exchangeability(all_states, all_residuals)
        if artifact_dir is not None:
            save_json(Path(artifact_dir) / "exchangeability.json", exchange_results)
        for k, v in exchange_results.items():
            acf1 = v.get("acf_lag1")
            if acf1 is not None and abs(acf1) > 0.3:
                if v.get("corrected"):
                    print(f"  Cluster {k} ACF(1)={acf1:.3f} — corrected (inflation ×{v['correction_factor']:.3f})")
                else:
                    print(f"  Cluster {k} ACF(1)={acf1:.3f} — consider re-running with correct_acf=True")
        self.model.train()
        self.head.train()

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        self._load_checkpoint(path)
        test_start = time.time()

        preds = []
        trues = []
        test_states = []

        self.model.eval()
        self.head.eval()

        with torch.no_grad():
            if getattr(self.args, 'walk_forward', False):
                n_windows = len(test_data)
                n_covered = (n_windows // self.args.pred_len) * self.args.pred_len
                if n_covered < n_windows:
                    warnings.warn(
                        f"Walk-forward evaluation: {n_windows - n_covered} of {n_windows} "
                        f"trailing test samples are dropped because {n_windows} is not "
                        f"divisible by pred_len={self.args.pred_len}.",
                        UserWarning,
                        stacklevel=2,
                    )
                print("Running walk-forward rolling window evaluation...")
                for i in range(0, n_covered, self.args.pred_len):
                    bx, by, bxm, bym = test_data[i]
                    batch_x = torch.from_numpy(bx).unsqueeze(0)
                    batch_y = torch.from_numpy(by).unsqueeze(0)
                    final_state, outputs, batch_y = self._forward_and_slice(batch_x, batch_y)
                    preds.append(outputs.detach().cpu().numpy())
                    trues.append(batch_y.detach().cpu().numpy())
                    test_states.append(final_state.detach().cpu().numpy())
            else:
                for batch_x, batch_y, _batch_x_mark, _batch_y_mark in test_loader:
                    final_state, outputs, batch_y = self._forward_and_slice(batch_x, batch_y)
                    preds.append(outputs.detach().cpu().numpy())
                    trues.append(batch_y.detach().cpu().numpy())
                    test_states.append(final_state.detach().cpu().numpy())

            preds = self._concatenate_batches(preds, 'prediction')
            trues = self._concatenate_batches(trues, 'target')
            test_states = self._concatenate_batches(test_states, 'state')

        mae = mean_absolute_error(trues.flatten(), preds.flatten())
        mse = mean_squared_error(trues.flatten(), preds.flatten())
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(trues.flatten(), preds.flatten())

        coverage = None
        mean_width = None
        winkler = None
        calib_err = None
        cluster_labels = None
        coverage_by_cluster = {}
        if hasattr(self, 'conformal') and self.conformal.calibrated:
            lower, upper = self.conformal.predict(
                torch.from_numpy(test_states).float(),
                torch.from_numpy(preds).float(),
            )
            lower_np = lower.numpy()
            upper_np = upper.numpy()
            cluster_labels = self.conformal.last_predicted_clusters_
            coverage = compute_picp(lower_np, upper_np, trues)
            mean_width = compute_mpiw(lower_np, upper_np)
            winkler = winkler_score(lower_np, upper_np, trues, alpha=self.args.conformal_alpha)
            calib_err = calibration_error(lower_np, upper_np, trues, alpha=self.args.conformal_alpha)
            coverage_by_cluster = self._coverage_by_cluster(lower_np, upper_np, trues, cluster_labels)
            print(
                f'Coverage@{(1.0 - self.args.conformal_alpha) * 100:.0f}%: {coverage:.4f}, '
                f'MPIW: {mean_width:.4f}, Winkler: {winkler:.4f}'
            )
        else:
            lower_np = np.full_like(preds, np.nan)
            upper_np = np.full_like(preds, np.nan)

        print(f'MSE:{mse:.6f} MAE:{mae:.6f} RMSE:{rmse:.6f} MAPE:{mape:.2f}%')

        if self.args.use_wandb:
            import wandb
            wandb.log({
                "test_mse": mse,
                "test_mae": mae,
                "test_coverage": coverage if coverage is not None else 0.0,
                "test_mean_width": mean_width if mean_width is not None else 0.0,
            })

        folder_path = Path(self.args.results_dir) / setting
        folder_path.mkdir(parents=True, exist_ok=True)

        point_metrics = {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}
        interval_metrics = {
            "coverage": coverage if coverage is not None else None,
            "mean_width": mean_width if mean_width is not None else None,
            "winkler": winkler if winkler is not None else None,
            "calibration_error": calib_err if calib_err is not None else None,
            "alpha": self.args.conformal_alpha,
            "coverage_scope": getattr(self.conformal, "coverage_scope", None) if hasattr(self, "conformal") else None,
        }
        metrics_payload = {
            "setting": setting,
            "point": point_metrics,
            "interval": interval_metrics,
        }

        np.save(folder_path / 'metrics.npy', np.array([mae, mse, rmse, mape]))
        np.save(folder_path / 'conformal.npy', np.array([coverage if coverage is not None else -1,
                                                          mean_width if mean_width is not None else -1,
                                                          winkler if winkler is not None else -1]))
        np.save(folder_path / 'pred.npy', preds)
        np.save(folder_path / 'true.npy', trues)
        np.save(folder_path / 'lower.npy', lower_np)
        np.save(folder_path / 'upper.npy', upper_np)
        np.save(folder_path / 'states.npy', test_states)
        np.save(folder_path / 'residuals.npy', np.abs(preds - trues))
        if cluster_labels is not None:
            np.save(folder_path / 'cluster_labels.npy', cluster_labels)

        save_json(folder_path / "metrics.json", metrics_payload)
        save_json(folder_path / "coverage_by_cluster.json", coverage_by_cluster)
        save_json(folder_path / "config.json", vars(self.args))
        save_json(folder_path / "environment.json", environment_snapshot(self.device))
        runtime = dict(getattr(self, "train_runtime_", {}))
        runtime["test_seconds"] = time.time() - test_start
        runtime["test_samples"] = len(test_data)
        save_json(folder_path / "runtime.json", runtime)
        if hasattr(self, "conformal") and self.conformal.calibrated:
            save_json(folder_path / "cluster_stats.json", self.conformal.get_cluster_stats())

        return


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
        print(f'Updating learning rate to {lr}')
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8,
        }
        if epoch in lr_adjust:
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f'Updating learning rate to {lr}')
    elif args.lradj == 'cosine':
        # Cosine annealing: smoothly decays to 0 over train_epochs.
        # eta_min is set to 1% of the initial LR.
        lr = args.learning_rate * 0.5 * (
            1.0 + np.cos(np.pi * epoch / args.train_epochs)
        )
        lr = max(lr, args.learning_rate * 0.01)  # floor at 1% of initial
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr:.2e} (cosine, epoch {epoch}/{args.train_epochs})')
    else:
        raise ValueError(f"Unknown lradj policy: {args.lradj!r}. Use 'type1', 'type2', or 'cosine'.")

if __name__ == '__main__':
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None, help='YAML/JSON config file')
    pre_args, _ = pre_parser.parse_known_args()
    config_defaults = load_config_defaults(pre_args.config)
    cli_options = provided_cli_options(sys.argv[1:])

    parser = argparse.ArgumentParser(description='CISSN Benchmark Runner', parents=[pre_parser])
    parser.set_defaults(**config_defaults)

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
    parser.add_argument('--d_model', type=int, default=64, help='model hidden dimension')
    parser.add_argument('--state_dim', type=int, default=5, help='latent state dimension')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate')
    parser.add_argument('--lambda_cov', type=float, default=1.0, help='covariance loss weight')
    parser.add_argument('--lambda_temp', type=float, default=0.5, help='temporal consistency loss weight')
    parser.add_argument('--lambda_correction_scale', type=float, default=0.0, help='penalty weight keeping encoder correction scale near 0.01')

    parser.add_argument('--num_workers', type=int, default=0, help='dataloader workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='lr schedule [type1, type2, cosine]')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='max gradient norm; <=0 disables clipping')

    parser.add_argument('--use_wandb', action='store_true', help='enable wandb logging')
    parser.add_argument('--project_name', type=str, default='CISSN_Benchmark', help='wandb project name')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # New arguments for improvements
    parser.add_argument('--walk_forward', action='store_true', help='Enable walk-forward rolling window evaluation')
    parser.add_argument('--conformal_alpha', type=float, default=0.1, help='conformal significance level')
    parser.add_argument('--n_clusters', type=int, default=5, help='requested SCCP clusters')
    parser.add_argument('--multivariate_strategy', type=str, default='per_feature', help='Conformal strategy [per_feature, max, mean, mahalanobis]')

    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    protected = set(config_defaults) | cli_options
    apply_dataset_defaults(args, protected)
    if args.features == 'MS' and 'c_out' not in protected:
        args.c_out = 1

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('Args in experiment:')
    print(args)

    setting = build_setting_name(args)
    
    if args.use_wandb:
        import wandb
        wandb.init(project=args.project_name, config=args, name=setting)

    exp = Experiment(args)
    print(f'Training: {setting}')
    exp.train(setting)
    print(f'Testing:  {setting}')
    exp.test(setting)
    
    if args.use_wandb:
        wandb.finish()
