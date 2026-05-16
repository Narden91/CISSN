#!/usr/bin/env python
"""Unified experiment runner for implemented publication baselines."""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .run_benchmark import (
        _format_float_token,
        adjust_learning_rate,
        apply_dataset_defaults,
        environment_snapshot,
        load_config_defaults,
        provided_cli_options,
        save_json,
    )
except ImportError:
    from run_benchmark import (
        _format_float_token,
        adjust_learning_rate,
        apply_dataset_defaults,
        environment_snapshot,
        load_config_defaults,
        provided_cli_options,
        save_json,
    )

from cissn.baselines import (
    DLinear,
    DeepEnsemble,
    DeepState,
    MCDropout,
    PatchTST,
    evaluate_baseline,
    slice_forecast,
    train_baseline_epoch,
)
from cissn.data.data_loader import get_data_loader
from cissn.data.registry import supported_datasets
from cissn.evaluation.metrics import (
    calibration_error,
    compute_mpiw,
    compute_picp,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    winkler_score,
)
from cissn.losses.disentangle_loss import DisentanglementLoss
from cissn.models.encoder import DisentangledStateEncoder
from cissn.models.forecast_head import ForecastHead


SUPPORTED_MODELS = (
    "dlinear",
    "patchtst",
    "deepstate",
    "mc_dropout",
    "deep_ensemble",
)
POINT_MODELS = {"dlinear", "patchtst", "deepstate"}
BACKBONE_MODELS = {"mc_dropout", "deep_ensemble"}


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_setting_name(args) -> str:
    tokens = [
        "BASELINE",
        args.model,
        args.data,
        args.features,
        f"sl{args.seq_len}",
        f"pl{args.pred_len}",
        f"seed{args.seed}",
    ]
    if args.model in {"patchtst", "deepstate", "mc_dropout", "deep_ensemble"}:
        tokens.append(f"dm{args.d_model}")
    if args.model in {"mc_dropout", "deep_ensemble", "deepstate"}:
        tokens.append(f"a{_format_float_token(args.conformal_alpha)}")
    if args.model == "patchtst":
        tokens.append(f"pt{args.patch_len}")
        tokens.append(f"st{args.patch_stride}")
    if args.model == "mc_dropout":
        tokens.append(f"mcs{args.mc_samples}")
        tokens.append(f"sd{args.state_dim}")
    if args.model == "deep_ensemble":
        tokens.append(f"ens{args.ensemble_size}")
        tokens.append(f"sd{args.state_dim}")
    return "_".join(tokens)


def build_member_setting(setting: str, member_seed: int, member_index: int) -> str:
    return f"{setting}_member{member_index}_seed{member_seed}"


def concatenate_batches(batches: list[np.ndarray], name: str) -> np.ndarray:
    if not batches:
        raise RuntimeError(f"No {name} batches were produced.")
    return np.concatenate(batches, axis=0)


def infer_coverage_scope(lower: Optional[np.ndarray] = None, upper: Optional[np.ndarray] = None) -> Optional[str]:
    if lower is None or upper is None:
        return None
    return "marginal"


def compute_metrics(args, preds: np.ndarray, trues: np.ndarray, lower: Optional[np.ndarray] = None, upper: Optional[np.ndarray] = None):
    mae = mean_absolute_error(trues.flatten(), preds.flatten())
    mse = mean_squared_error(trues.flatten(), preds.flatten())
    rmse = float(np.sqrt(mse))
    mape = mean_absolute_percentage_error(trues.flatten(), preds.flatten())

    point_metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
    }
    interval_metrics = {
        "coverage": None,
        "mean_width": None,
        "winkler": None,
        "calibration_error": None,
        "alpha": args.conformal_alpha,
        "coverage_scope": infer_coverage_scope(lower=lower, upper=upper),
    }
    if lower is not None and upper is not None:
        interval_metrics.update(
            {
                "coverage": compute_picp(lower, upper, trues),
                "mean_width": compute_mpiw(lower, upper),
                "winkler": winkler_score(lower, upper, trues, alpha=args.conformal_alpha),
                "calibration_error": calibration_error(lower, upper, trues, alpha=args.conformal_alpha),
            }
        )
    return point_metrics, interval_metrics


def save_result_artifacts(
    args,
    setting: str,
    point_metrics: dict,
    interval_metrics: dict,
    preds: np.ndarray,
    trues: np.ndarray,
    runtime: dict,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
) -> Path:
    folder_path = Path(args.results_dir) / setting
    folder_path.mkdir(parents=True, exist_ok=True)

    np.save(folder_path / "pred.npy", preds)
    np.save(folder_path / "true.npy", trues)
    if lower is not None and upper is not None:
        np.save(folder_path / "lower.npy", lower)
        np.save(folder_path / "upper.npy", upper)

    save_json(
        folder_path / "metrics.json",
        {
            "setting": setting,
            "model": args.model,
            "point": point_metrics,
            "interval": interval_metrics,
        },
    )
    save_json(folder_path / "config.json", vars(args))
    save_json(folder_path / "environment.json", environment_snapshot(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    save_json(folder_path / "runtime.json", runtime)
    return folder_path


def save_single_checkpoint(path: Path, model: nn.Module) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "checkpoint.pth")


def load_single_checkpoint(path: Path, model: nn.Module, device: torch.device) -> None:
    model.load_state_dict(torch.load(path / "checkpoint.pth", map_location=device, weights_only=True))


def save_backbone_checkpoint(path: Path, encoder: nn.Module, head: nn.Module) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), path / "checkpoint.pth")
    torch.save(head.state_dict(), path / "checkpoint_head.pth")


def load_backbone_checkpoint(path: Path, encoder: nn.Module, head: nn.Module, device: torch.device) -> None:
    encoder.load_state_dict(torch.load(path / "checkpoint.pth", map_location=device, weights_only=True))
    head.load_state_dict(torch.load(path / "checkpoint_head.pth", map_location=device, weights_only=True))


def validate_single_model(model: nn.Module, loader, criterion: nn.Module, device: torch.device, args) -> float:
    model.eval()
    total_loss = 0.0
    total_weight = 0
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            outputs, targets = slice_forecast(model(batch_x), batch_y, args.pred_len, args.features)
            batch_weight = outputs.numel()
            total_loss += criterion(outputs, targets).item() * batch_weight
            total_weight += batch_weight
    if total_weight == 0:
        raise RuntimeError("Validation loader produced no prediction elements.")
    model.train()
    return total_loss / total_weight


def evaluate_deepstate(model: DeepState, loader, device: torch.device, args):
    model.eval()
    preds, trues, lowers, uppers = [], [], [], []
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            mean, lower, upper = model.predict_interval(batch_x)
            mean, targets = slice_forecast(mean, batch_y, args.pred_len, args.features)
            lower, _ = slice_forecast(lower, batch_y, args.pred_len, args.features)
            upper, _ = slice_forecast(upper, batch_y, args.pred_len, args.features)
            preds.append(mean.detach().cpu().numpy())
            trues.append(targets.detach().cpu().numpy())
            lowers.append(lower.detach().cpu().numpy())
            uppers.append(upper.detach().cpu().numpy())
    return (
        concatenate_batches(preds, "prediction"),
        concatenate_batches(trues, "target"),
        concatenate_batches(lowers, "lower interval"),
        concatenate_batches(uppers, "upper interval"),
    )


def build_single_model(args) -> nn.Module:
    if args.model == "dlinear":
        return DLinear(
            input_dim=args.enc_in,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            output_dim=args.c_out,
            kernel_size=args.kernel_size,
        )
    if args.model == "patchtst":
        num_layers = args.num_layers if args.num_layers is not None else 3
        return PatchTST(
            input_dim=args.enc_in,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )
    if args.model == "deepstate":
        num_layers = args.num_layers if args.num_layers is not None else 2
        return DeepState(
            input_dim=args.enc_in,
            pred_len=args.pred_len,
            output_dim=args.c_out,
            hidden_dim=args.d_model,
            num_layers=num_layers,
            dropout=args.dropout,
            alpha=args.conformal_alpha,
        )
    raise ValueError(f"Unsupported point baseline: {args.model}")


def run_point_baseline(args, setting: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_single_model(args).to(device)
    train_data, train_loader = get_data_loader(args, "train")
    vali_data, vali_loader = get_data_loader(args, "val")
    test_data, test_loader = get_data_loader(args, "test")

    checkpoint_dir = Path(args.checkpoints) / setting
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val = float("inf")
    bad_epochs = 0
    train_start = time.time()

    for epoch in range(args.train_epochs):
        train_loss = train_baseline_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            pred_len=args.pred_len,
            features=args.features,
            grad_clip=args.grad_clip,
        )
        vali_loss = validate_single_model(model, vali_loader, criterion, device, args)
        print(
            f"Epoch: {epoch + 1}, Steps: {len(train_loader)} | "
            f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}"
        )
        if vali_loss < best_val:
            save_single_checkpoint(checkpoint_dir, model)
            best_val = vali_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            print(f"EarlyStopping counter: {bad_epochs} out of {args.patience}")
            if bad_epochs >= args.patience:
                print("Early stopping")
                break
        adjust_learning_rate(optimizer, epoch + 1, args)

    load_single_checkpoint(checkpoint_dir, model, device)
    test_start = time.time()
    lower = None
    upper = None
    if args.model == "deepstate":
        preds, trues, lower, upper = evaluate_deepstate(model, test_loader, device, args)
    else:
        eval_result = evaluate_baseline(model, test_loader, device, args.pred_len, args.features)
        preds = eval_result.predictions
        trues = eval_result.targets

    point_metrics, interval_metrics = compute_metrics(args, preds, trues, lower=lower, upper=upper)
    runtime = {
        "train_seconds": time.time() - train_start,
        "test_seconds": time.time() - test_start,
        "train_samples": len(train_data),
        "validation_samples": len(vali_data),
        "test_samples": len(test_data),
    }
    return save_result_artifacts(args, setting, point_metrics, interval_metrics, preds, trues, runtime, lower=lower, upper=upper)


def build_backbone(args):
    if args.state_dim != 5:
        raise ValueError(f"{args.model} requires state_dim=5; got {args.state_dim}.")
    encoder = DisentangledStateEncoder(
        input_dim=args.enc_in,
        state_dim=args.state_dim,
        hidden_dim=args.d_model,
        dropout=args.dropout,
    )
    head = ForecastHead(
        state_dim=args.state_dim,
        output_dim=args.c_out,
        horizon=args.pred_len,
        hidden_dim=args.d_model // 2,
    )
    return encoder, head


def forward_backbone(encoder: nn.Module, head: nn.Module, batch_x, batch_y, device: torch.device, args, return_all_states: bool = False):
    batch_x = batch_x.float().to(device, non_blocking=True)
    batch_y = batch_y.float().to(device, non_blocking=True)
    if return_all_states:
        all_states = encoder(batch_x, return_all_states=True)
        final_state = all_states[:, -1, :]
    else:
        all_states = None
        final_state = encoder(batch_x)
    outputs = head(final_state)
    outputs, targets = slice_forecast(outputs, batch_y, args.pred_len, args.features)
    if return_all_states:
        return all_states, final_state, outputs, targets
    return final_state, outputs, targets


def train_backbone_epoch(encoder: nn.Module, head: nn.Module, loader, optimizer, criterion: nn.Module, disentangle_criterion: DisentanglementLoss, device: torch.device, args) -> float:
    encoder.train()
    head.train()
    losses = []
    for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
        optimizer.zero_grad()
        states, _final_state, outputs, targets = forward_backbone(
            encoder, head, batch_x, batch_y, device, args, return_all_states=True
        )
        loss = criterion(outputs, targets) + disentangle_criterion(states)
        if args.lambda_correction_scale > 0 and hasattr(encoder, "_correction_scale"):
            target_scale = torch.tensor(0.01, device=device, dtype=outputs.dtype)
            loss = loss + args.lambda_correction_scale * (encoder._correction_scale() - target_scale) ** 2
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()),
                max_norm=args.grad_clip,
            )
        optimizer.step()
        losses.append(float(loss.item()))
    if not losses:
        raise RuntimeError("Backbone training loader produced no batches.")
    return float(np.mean(losses))


def validate_backbone(encoder: nn.Module, head: nn.Module, loader, criterion: nn.Module, device: torch.device, args) -> float:
    encoder.eval()
    head.eval()
    total_loss = 0.0
    total_weight = 0
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            _final_state, outputs, targets = forward_backbone(encoder, head, batch_x, batch_y, device, args)
            batch_weight = outputs.numel()
            total_loss += criterion(outputs, targets).item() * batch_weight
            total_weight += batch_weight
    if total_weight == 0:
        raise RuntimeError("Validation loader produced no prediction elements.")
    encoder.train()
    head.train()
    return total_loss / total_weight


def evaluate_backbone_point(encoder: nn.Module, head: nn.Module, loader, device: torch.device, args):
    encoder.eval()
    head.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            _final_state, outputs, targets = forward_backbone(encoder, head, batch_x, batch_y, device, args)
            preds.append(outputs.detach().cpu().numpy())
            trues.append(targets.detach().cpu().numpy())
    return concatenate_batches(preds, "prediction"), concatenate_batches(trues, "target")


def evaluate_mc_dropout(encoder: nn.Module, head: nn.Module, loader, device: torch.device, args):
    wrapper = MCDropout(n_samples=args.mc_samples, alpha=args.conformal_alpha)
    preds, trues, lowers, uppers = [], [], [], []
    for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
        batch_x = batch_x.float().to(device, non_blocking=True)
        batch_y = batch_y.float().to(device, non_blocking=True)
        mean, lower, upper = wrapper.predict(encoder, head, batch_x)
        mean, targets = slice_forecast(mean, batch_y, args.pred_len, args.features)
        lower, _ = slice_forecast(lower, batch_y, args.pred_len, args.features)
        upper, _ = slice_forecast(upper, batch_y, args.pred_len, args.features)
        preds.append(mean.detach().cpu().numpy())
        trues.append(targets.detach().cpu().numpy())
        lowers.append(lower.detach().cpu().numpy())
        uppers.append(upper.detach().cpu().numpy())
    return (
        concatenate_batches(preds, "prediction"),
        concatenate_batches(trues, "target"),
        concatenate_batches(lowers, "lower interval"),
        concatenate_batches(uppers, "upper interval"),
    )


def train_backbone_member(args, setting: str, member_seed: int):
    set_random_seed(member_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, head = build_backbone(args)
    encoder = encoder.to(device)
    head = head.to(device)

    train_data, train_loader = get_data_loader(args, "train")
    vali_data, vali_loader = get_data_loader(args, "val")
    checkpoint_dir = Path(args.checkpoints) / setting
    criterion = nn.MSELoss()
    disentangle_criterion = DisentanglementLoss(
        lambda_cov=args.lambda_cov,
        lambda_temporal=args.lambda_temp,
    ).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=args.learning_rate)
    best_val = float("inf")
    bad_epochs = 0
    train_start = time.time()

    for epoch in range(args.train_epochs):
        train_loss = train_backbone_epoch(
            encoder=encoder,
            head=head,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            disentangle_criterion=disentangle_criterion,
            device=device,
            args=args,
        )
        vali_loss = validate_backbone(encoder, head, vali_loader, criterion, device, args)
        print(
            f"Epoch: {epoch + 1}, Steps: {len(train_loader)} | "
            f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}"
        )
        if vali_loss < best_val:
            save_backbone_checkpoint(checkpoint_dir, encoder, head)
            best_val = vali_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            print(f"EarlyStopping counter: {bad_epochs} out of {args.patience}")
            if bad_epochs >= args.patience:
                print("Early stopping")
                break
        adjust_learning_rate(optimizer, epoch + 1, args)

    load_backbone_checkpoint(checkpoint_dir, encoder, head, device)
    runtime = {
        "train_seconds": time.time() - train_start,
        "train_samples": len(train_data),
        "validation_samples": len(vali_data),
        "seed": member_seed,
    }
    return encoder, head, device, runtime


def run_mc_dropout(args, setting: str):
    encoder, head, device, runtime = train_backbone_member(args, setting, args.seed)
    test_data, test_loader = get_data_loader(args, "test")
    test_start = time.time()
    preds, trues, lower, upper = evaluate_mc_dropout(encoder, head, test_loader, device, args)
    point_metrics, interval_metrics = compute_metrics(args, preds, trues, lower=lower, upper=upper)
    runtime.update(
        {
            "test_seconds": time.time() - test_start,
            "test_samples": len(test_data),
        }
    )
    return save_result_artifacts(args, setting, point_metrics, interval_metrics, preds, trues, runtime, lower=lower, upper=upper)


def parse_ensemble_seeds(args) -> list[int]:
    if args.ensemble_seeds:
        seeds = [int(token.strip()) for token in args.ensemble_seeds.split(",") if token.strip()]
    else:
        seeds = [args.seed + offset for offset in range(args.ensemble_size)]
    if len(seeds) < 2:
        raise ValueError("deep_ensemble requires at least two member seeds.")
    return seeds


def run_deep_ensemble(args, setting: str):
    member_seeds = parse_ensemble_seeds(args)
    test_data, test_loader = get_data_loader(args, "test")
    ensemble_forecasts = []
    reference_targets = None
    member_runtimes = []
    test_start = time.time()

    for index, member_seed in enumerate(member_seeds, start=1):
        member_setting = build_member_setting(setting, member_seed, index)
        print(f"Training ensemble member {index}/{len(member_seeds)} with seed={member_seed}")
        encoder, head, device, runtime = train_backbone_member(args, member_setting, member_seed)
        preds, trues = evaluate_backbone_point(encoder, head, test_loader, device, args)
        ensemble_forecasts.append(torch.from_numpy(preds))
        if reference_targets is None:
            reference_targets = trues
        elif not np.allclose(reference_targets, trues):
            raise RuntimeError("Deep ensemble members produced mismatched test targets.")
        member_runtimes.append(runtime)

    wrapper = DeepEnsemble(alpha=args.conformal_alpha)
    mean, lower, upper = wrapper.predict(ensemble_forecasts)
    preds = mean.numpy()
    lower_np = lower.numpy()
    upper_np = upper.numpy()
    point_metrics, interval_metrics = compute_metrics(args, preds, reference_targets, lower=lower_np, upper=upper_np)
    runtime = {
        "train_seconds": sum(item["train_seconds"] for item in member_runtimes),
        "test_seconds": time.time() - test_start,
        "train_samples": member_runtimes[0]["train_samples"],
        "validation_samples": member_runtimes[0]["validation_samples"],
        "test_samples": len(test_data),
        "member_seeds": member_seeds,
        "member_train_seconds": [item["train_seconds"] for item in member_runtimes],
    }
    return save_result_artifacts(
        args,
        setting,
        point_metrics,
        interval_metrics,
        preds,
        reference_targets,
        runtime,
        lower=lower_np,
        upper=upper_np,
    )


def maybe_log_final_metrics(args, point_metrics: dict, interval_metrics: dict) -> None:
    if not args.use_wandb:
        return
    import wandb

    payload = {
        "test_mse": point_metrics["mse"],
        "test_mae": point_metrics["mae"],
        "test_rmse": point_metrics["rmse"],
        "test_mape": point_metrics["mape"],
    }
    if interval_metrics["coverage"] is not None:
        payload.update(
            {
                "test_coverage": interval_metrics["coverage"],
                "test_mean_width": interval_metrics["mean_width"],
                "test_winkler": interval_metrics["winkler"],
                "test_calibration_error": interval_metrics["calibration_error"],
            }
        )
    wandb.log(payload)


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="YAML/JSON config file")
    pre_args, _ = pre_parser.parse_known_args()
    config_defaults = load_config_defaults(pre_args.config)
    cli_options = provided_cli_options(sys.argv[1:])

    parser = argparse.ArgumentParser(description="Baseline Experiment Runner", parents=[pre_parser])
    parser.set_defaults(**config_defaults)

    parser.add_argument("--model", type=str, required=True, choices=SUPPORTED_MODELS, help="baseline model name")
    parser.add_argument("--data", type=str, default="ETTh1", choices=supported_datasets(), help="dataset name")
    parser.add_argument("--root_path", type=str, default="./data/ETT/", help="data root directory")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data filename")
    parser.add_argument("--features", type=str, default="M", help="forecasting task [M, S, MS]")
    parser.add_argument("--target", type=str, default="OT", help="target feature for S/MS tasks")
    parser.add_argument("--freq", type=str, default="h", help="time feature encoding frequency")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="checkpoint directory")
    parser.add_argument("--results_dir", type=str, default="./results/", help="results directory")

    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="decoder start token length")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction horizon")

    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=64, help="model hidden dimension")
    parser.add_argument("--state_dim", type=int, default=5, help="latent state dimension")
    parser.add_argument("--dropout", type=float, default=0.05, help="dropout rate")
    parser.add_argument("--lambda_cov", type=float, default=1.0, help="covariance loss weight")
    parser.add_argument("--lambda_temp", type=float, default=0.5, help="temporal consistency loss weight")
    parser.add_argument("--lambda_correction_scale", type=float, default=0.0, help="penalty weight keeping encoder correction scale near 0.01")

    parser.add_argument("--num_workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--train_epochs", type=int, default=10, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--lradj", type=str, default="type1", help="lr schedule [type1, type2, cosine]")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="max gradient norm; <=0 disables clipping")

    parser.add_argument("--use_wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--project_name", type=str, default="CISSN_Baselines", help="wandb project name")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--conformal_alpha", type=float, default=0.1, help="interval significance level")

    parser.add_argument("--kernel_size", type=int, default=25, help="DLinear moving-average kernel size")
    parser.add_argument("--patch_len", type=int, default=16, help="PatchTST patch length")
    parser.add_argument("--patch_stride", type=int, default=8, help="PatchTST patch stride")
    parser.add_argument("--nhead", type=int, default=8, help="PatchTST attention heads")
    parser.add_argument("--num_layers", type=int, default=None, help="optional model layer count override")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="PatchTST feedforward dimension")
    parser.add_argument("--mc_samples", type=int, default=50, help="MC-Dropout stochastic forward passes")
    parser.add_argument("--ensemble_size", type=int, default=3, help="Deep Ensemble member count when ensemble_seeds is not provided")
    parser.add_argument("--ensemble_seeds", type=str, default="", help="comma-separated Deep Ensemble member seeds")

    parser.set_defaults(**config_defaults)
    args = parser.parse_args()

    protected = set(config_defaults) | cli_options
    apply_dataset_defaults(args, protected)
    if args.features == "MS" and "c_out" not in protected:
        args.c_out = 1
    if args.model in BACKBONE_MODELS and args.state_dim != 5:
        raise ValueError(f"{args.model} requires state_dim=5; got {args.state_dim}.")
    if args.model == "deep_ensemble" and args.ensemble_seeds:
        args.ensemble_size = len(parse_ensemble_seeds(args))
    return args


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    setting = build_setting_name(args)

    if args.use_wandb:
        import wandb

        wandb.init(project=args.project_name, config=vars(args), name=setting)

    print("Args in experiment:")
    print(args)
    print(f"Running baseline: {setting}")

    if args.model in POINT_MODELS:
        result_dir = run_point_baseline(args, setting)
    elif args.model == "mc_dropout":
        result_dir = run_mc_dropout(args, setting)
    elif args.model == "deep_ensemble":
        result_dir = run_deep_ensemble(args, setting)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    metrics_payload = result_dir / "metrics.json"
    if metrics_payload.exists():
        import json

        payload = json.loads(metrics_payload.read_text(encoding="utf-8"))
        maybe_log_final_metrics(args, payload["point"], payload["interval"])

    print(f"Saved artifacts to {result_dir}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()