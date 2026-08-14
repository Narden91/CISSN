#!/usr/bin/env python
"""Unified experiment runner for implemented publication baselines."""
from __future__ import annotations

import argparse
import json
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
        build_protocol_manifest,
        build_run_setting,
        environment_snapshot,
        enforce_evidence_contract,
        load_config_defaults,
        provided_cli_options,
        require_clean_source,
        save_json,
        set_random_seed,
        validate_config_defaults,
        validate_runtime_args,
    )
except ImportError:
    from run_benchmark import (
        _format_float_token,
        adjust_learning_rate,
        apply_dataset_defaults,
        build_protocol_manifest,
        build_run_setting,
        environment_snapshot,
        enforce_evidence_contract,
        load_config_defaults,
        provided_cli_options,
        require_clean_source,
        save_json,
        set_random_seed,
        validate_config_defaults,
        validate_runtime_args,
    )

from cissn.baselines import (
    DLinear,
    DeepEnsemble,
    DeepState,
    FlatConformal,
    MCDropout,
    PatchTST,
    evaluate_baseline,
    slice_forecast,
    train_baseline_epoch,
)
from cissn.data.data_loader import get_data_loader
from cissn.data.registry import get_dataset_spec, supported_datasets
from cissn.evaluation.metrics import (
    compute_joint_picp,
    compute_mpiw,
    compute_picp,
    mean_absolute_error,
    mean_scaled_interval_score,
    mean_squared_error,
    per_origin_interval_scores,
    winkler_score,
)
from cissn.evaluation.sanity import check_forecast_sanity
from cissn.losses.disentangle_loss import DisentanglementLoss
from cissn.models.encoder import DisentangledStateEncoder
from cissn.models.forecast_head import ForecastHead
from cissn.utils import (
    EarlyStopping, create_temporary_result_root, finalize_result_directory,
    require_new_run, print_epoch_summary, print_run_header, select_device, track,
    write_completion_manifest,
)


SUPPORTED_MODELS = (
    "dlinear",
    "patchtst",
    "deepstate",
    "mc_dropout",
    "deep_ensemble",
)
POINT_MODELS = {"dlinear", "patchtst", "deepstate"}
BACKBONE_MODELS = {"mc_dropout", "deep_ensemble"}
MODEL_DISPLAY_NAMES = {
    "patchtst": "patch_transformer_lite",
    "deepstate": "structured_gru_nll",
    "mc_dropout": "cissn_mc_dropout",
    "deep_ensemble": "cissn_deep_ensemble",
}


def canonical_model_name(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


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
        tokens.append(args.multivariate_strategy)
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


def compute_metrics(
    args,
    preds: np.ndarray,
    trues: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    coverage_scope: str = "marginal",
    interval_origin: str = "conformalized",
):
    """Point + interval metrics. Units are z-scored (per-feature train-split
    standardization) -- the LTSF convention for MSE/MAE/MPIW/Winkler/MSIS.
    MAPE is not reported: on zero-centred standardized data its denominator
    is meaningless (see cissn/evaluation/metrics.py docstring history)."""
    mae = mean_absolute_error(trues.flatten(), preds.flatten())
    mse = mean_squared_error(trues.flatten(), preds.flatten())
    rmse = float(np.sqrt(mse))

    point_metrics = {"mae": mae, "mse": mse, "rmse": rmse}
    interval_metrics = {
        "coverage": None,
        "coverage_joint": None,
        "coverage_primary": None,
        "mean_width": None,
        "winkler": None,
        "calibration_error": None,
        "msis": None,
        "alpha": args.conformal_alpha,
        "coverage_scope": coverage_scope if lower is not None else None,
        "interval_origin": interval_origin if lower is not None else None,
        "units": "z-scored (per-feature train-split standardization)",
    }
    if lower is not None and upper is not None:
        msis_val = None
        if y_train is not None:
            seasonal_period = get_dataset_spec(args.data)["seasonal_period"]
            train_targets = y_train
            if args.features == "MS":
                train_targets = y_train[:, -1:]
            msis_val = mean_scaled_interval_score(
                lower, upper, trues, train_targets, seasonal_period, alpha=args.conformal_alpha
            )
        coverage = compute_picp(lower, upper, trues)
        coverage_joint = compute_joint_picp(lower, upper, trues)
        coverage_primary = coverage_joint if coverage_scope == "simultaneous" else coverage
        interval_metrics.update(
            {
                "coverage": coverage,
                "coverage_joint": coverage_joint,
                "coverage_primary": coverage_primary,
                "mean_width": compute_mpiw(lower, upper),
                "winkler": winkler_score(lower, upper, trues, alpha=args.conformal_alpha),
                "calibration_error": abs(coverage_primary - (1.0 - args.conformal_alpha)),
                "msis": msis_val,
            }
        )
    return point_metrics, interval_metrics


def save_history(checkpoint_dir: Path, history: list[dict]) -> None:
    save_json(checkpoint_dir / "history.json", history)


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
    history: Optional[list[dict]] = None,
    y_train: Optional[np.ndarray] = None,
) -> Path:
    folder_path = Path(args.results_dir) / setting
    folder_path.mkdir(parents=True, exist_ok=True)

    sanity_report = check_forecast_sanity(
        preds,
        trues,
        history=history,
        lower=lower,
        upper=upper,
        y_train=y_train,
        seasonal_period=get_dataset_spec(args.data)["seasonal_period"],
        horizon=args.pred_len,
    )
    for msg in sanity_report["failures"]:
        print(f"Result review | structural failure: {msg}")
    for msg in sanity_report["warnings"]:
        print(f"Result review | quality note: {msg}")

    np.save(folder_path / "pred.npy", preds)
    np.save(folder_path / "true.npy", trues)
    if lower is not None and upper is not None:
        lower_np, upper_np = lower, upper
    else:
        lower_np = np.full_like(preds, np.nan)
        upper_np = np.full_like(preds, np.nan)
    np.save(folder_path / "lower.npy", lower_np)
    np.save(folder_path / "upper.npy", upper_np)
    if lower is not None and upper is not None:
        per_origin = per_origin_interval_scores(lower_np, upper_np, trues, args.conformal_alpha)
    else:
        per_origin = np.full((len(preds), 3), np.nan, dtype=np.float64)
    np.save(folder_path / "per_origin_interval_metrics.npy", per_origin)

    save_json(
        folder_path / "metrics.json",
        {
            "setting": setting,
            "model": canonical_model_name(args.model),
            "point": point_metrics,
            "interval": interval_metrics,
            "sanity_passed": sanity_report["passed"],
            "structural_passed": sanity_report["structural_passed"],
            "quality_flags": sanity_report["warnings"],
        },
    )
    save_json(folder_path / "sanity.json", sanity_report)
    save_json(folder_path / "config.json", vars(args))
    save_json(folder_path / "environment.json", environment_snapshot(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    save_json(folder_path / "protocol.json", args.protocol)
    save_json(folder_path / "runtime.json", runtime)
    if history is not None:
        save_json(folder_path / "history.json", history)
    if getattr(args, "immutable_artifacts", False):
        write_completion_manifest(
            folder_path,
            [
                "metrics.json", "sanity.json", "config.json", "environment.json", "protocol.json",
                "runtime.json", "history.json", "pred.npy", "true.npy", "lower.npy", "upper.npy",
                "per_origin_interval_metrics.npy",
            ],
            args.protocol,
        )
    return folder_path


def load_single_checkpoint(path: Path, model: nn.Module, device: torch.device) -> None:
    model.load_state_dict(torch.load(path / "checkpoint.pth", map_location=device, weights_only=True))


def load_backbone_checkpoint(path: Path, encoder: nn.Module, head: nn.Module, device: torch.device) -> None:
    encoder.load_state_dict(torch.load(path / "checkpoint.pth", map_location=device, weights_only=True))
    head.load_state_dict(torch.load(path / "checkpoint_head.pth", map_location=device, weights_only=True))


def validate_single_model(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    args,
    probabilistic: bool = False,
) -> float:
    model.eval()
    total_loss = 0.0
    total_weight = 0
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in track(
            loader, description="Validation", total=len(loader), enabled=not getattr(args, "no_progress", True)
        ):
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            if probabilistic:
                mean, log_sigma = model.predict_distribution(batch_x)
                outputs, targets = slice_forecast(mean, batch_y, args.pred_len, args.features)
                log_sigma, _ = slice_forecast(log_sigma, batch_y, args.pred_len, args.features)
                loss = model.gaussian_nll(outputs, targets, log_sigma)
            else:
                outputs, targets = slice_forecast(model(batch_x), batch_y, args.pred_len, args.features)
                loss = criterion(outputs, targets)
            batch_weight = outputs.numel()
            total_loss += loss.item() * batch_weight
            total_weight += batch_weight
    if total_weight == 0:
        raise RuntimeError("Validation loader produced no prediction elements.")
    model.train()
    return total_loss / total_weight


def evaluate_deepstate(model: DeepState, loader, device: torch.device, args):
    model.eval()
    preds, trues, scales = [], [], []
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in track(
            loader, description="Testing", total=len(loader), enabled=not getattr(args, "no_progress", True)
        ):
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            mean, log_sigma = model.predict_distribution(batch_x)
            mean, targets = slice_forecast(mean, batch_y, args.pred_len, args.features)
            log_sigma, _ = slice_forecast(log_sigma, batch_y, args.pred_len, args.features)
            preds.append(mean.detach().cpu().numpy())
            trues.append(targets.detach().cpu().numpy())
            scales.append(torch.exp(log_sigma).detach().cpu().numpy())
    return (
        concatenate_batches(preds, "prediction"),
        concatenate_batches(trues, "target"),
        concatenate_batches(scales, "predictive scale"),
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
    device = select_device(require_gpu=getattr(args, "require_gpu", False))
    model = build_single_model(args).to(device)
    train_data, train_loader = get_data_loader(args, "train")
    vali_data, vali_loader = get_data_loader(args, "val")
    cal_data, cal_loader = get_data_loader(args, "cal")
    test_data, test_loader = get_data_loader(args, "test")

    checkpoint_dir = Path(args.checkpoints) / setting
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    train_start = time.time()
    history = []

    for epoch in range(args.train_epochs):
        epoch_start = time.time()
        train_loss = train_baseline_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            pred_len=args.pred_len,
            features=args.features,
            grad_clip=args.grad_clip,
            show_progress=not args.no_progress,
            progress_description=f"Epoch {epoch + 1}/{args.train_epochs}",
            probabilistic=args.model == "deepstate",
        )
        vali_loss = validate_single_model(
            model, vali_loader, criterion, device, args, probabilistic=args.model == "deepstate"
        )
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "vali_loss": vali_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })
        improved = early_stopping(vali_loss, model, path=str(checkpoint_dir))
        print_epoch_summary(
            epoch=epoch + 1, total_epochs=args.train_epochs, train_loss=train_loss,
            validation_loss=vali_loss, learning_rate=optimizer.param_groups[0]["lr"],
            elapsed_seconds=time.time() - epoch_start, improved=improved,
            patience_counter=early_stopping.counter, patience=early_stopping.patience,
        )
        if early_stopping.early_stop:
            break
        adjust_learning_rate(optimizer, epoch + 1, args)

    save_history(checkpoint_dir, history)
    load_single_checkpoint(checkpoint_dir, model, device)
    test_start = time.time()
    lower = None
    upper = None
    coverage_scope = "marginal"
    if args.model == "deepstate":
        preds, trues, test_scales = evaluate_deepstate(model, test_loader, device, args)
        if args.uq_interval_mode == "raw":
            z = float(np.sqrt(2.0) * torch.erfinv(torch.tensor(1.0 - args.conformal_alpha)).item())
            lower, upper = preds - z * test_scales, preds + z * test_scales
            interval_origin = "raw_parametric"
        else:
            cal_preds, cal_trues, cal_scales = evaluate_deepstate(model, cal_loader, device, args)
            conformal = FlatConformal(
                alpha=args.conformal_alpha, multivariate_strategy=args.multivariate_strategy
            )
            conformal.fit(np.abs(cal_preds - cal_trues), scales=cal_scales)
            lower_t, upper_t = conformal.predict(torch.from_numpy(preds).float(), scales=test_scales)
            lower, upper = lower_t.numpy(), upper_t.numpy()
            interval_origin = "conformalized_uq"
    else:
        eval_result = evaluate_baseline(model, test_loader, device, args.pred_len, args.features)
        preds = eval_result.predictions
        trues = eval_result.targets

        cal_result = evaluate_baseline(model, cal_loader, device, args.pred_len, args.features)
        cal_residuals = np.abs(cal_result.predictions - cal_result.targets)
        flat_cp = FlatConformal(
            alpha=args.conformal_alpha, multivariate_strategy=args.multivariate_strategy
        )
        flat_cp.fit(cal_residuals)
        lower_t, upper_t = flat_cp.predict(torch.from_numpy(preds).float())
        lower, upper = lower_t.numpy(), upper_t.numpy()
        interval_origin = "conformalized"

    point_metrics, interval_metrics = compute_metrics(
        args, preds, trues, lower=lower, upper=upper, y_train=train_data.data_y,
        coverage_scope=coverage_scope,
        interval_origin=interval_origin,
    )
    runtime = {
        "train_seconds": time.time() - train_start,
        "test_seconds": time.time() - test_start,
        "train_samples": len(train_data),
        "validation_samples": len(vali_data),
        "calibration_samples": len(cal_data),
        "test_samples": len(test_data),
        "epochs_requested": args.train_epochs,
        "epochs_run": len(history),
        "early_stopped": early_stopping.early_stop,
        "best_val_loss": early_stopping.val_loss_min,
    }
    return save_result_artifacts(
        args, setting, point_metrics, interval_metrics, preds, trues, runtime,
        lower=lower, upper=upper, history=history, y_train=train_data.data_y,
    )


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
        dropout=args.dropout,
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


def train_backbone_epoch(encoder: nn.Module, head: nn.Module, loader, optimizer, criterion: nn.Module, disentangle_criterion: DisentanglementLoss, device: torch.device, args, progress_description: str) -> float:
    encoder.train()
    head.train()
    parameters = [*encoder.parameters(), *head.parameters()]
    total_loss = torch.zeros((), device=device)
    total_weight = 0
    for batch_x, batch_y, _batch_x_mark, _batch_y_mark in track(
        loader, description=progress_description, total=len(loader), enabled=not getattr(args, "no_progress", True)
    ):
        optimizer.zero_grad(set_to_none=True)
        states, _final_state, outputs, targets = forward_backbone(
            encoder, head, batch_x, batch_y, device, args, return_all_states=True
        )
        loss = criterion(outputs, targets) + disentangle_criterion(states)
        if args.lambda_correction_scale > 0 and hasattr(encoder, "_correction_scale"):
            loss = loss + args.lambda_correction_scale * (encoder._correction_scale() - 0.01) ** 2
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=args.grad_clip,
            )
        optimizer.step()
        batch_weight = outputs.numel()
        total_loss += loss.detach() * batch_weight
        total_weight += batch_weight
    if total_weight == 0:
        raise RuntimeError("Backbone training loader produced no batches.")
    return float((total_loss / total_weight).item())


def validate_backbone(encoder: nn.Module, head: nn.Module, loader, criterion: nn.Module, device: torch.device, args) -> float:
    encoder.eval()
    head.eval()
    total_loss = 0.0
    total_weight = 0
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in track(
            loader, description="Validation", total=len(loader), enabled=not getattr(args, "no_progress", True)
        ):
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
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in track(
            loader, description="Testing", total=len(loader), enabled=not getattr(args, "no_progress", True)
        ):
            _final_state, outputs, targets = forward_backbone(encoder, head, batch_x, batch_y, device, args)
            preds.append(outputs.detach().cpu().numpy())
            trues.append(targets.detach().cpu().numpy())
    return concatenate_batches(preds, "prediction"), concatenate_batches(trues, "target")


def evaluate_mc_dropout(encoder: nn.Module, head: nn.Module, loader, device: torch.device, args):
    wrapper = MCDropout(n_samples=args.mc_samples, alpha=args.conformal_alpha)
    preds, trues, lowers, uppers = [], [], [], []
    for batch_x, batch_y, _batch_x_mark, _batch_y_mark in track(
        loader, description="Testing", total=len(loader), enabled=not getattr(args, "no_progress", True)
    ):
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


def conformalize_uq_intervals(
    predictions: np.ndarray,
    calibration_targets: np.ndarray,
    calibration_predictions: np.ndarray,
    calibration_lower: np.ndarray,
    calibration_upper: np.ndarray,
    raw_lower: np.ndarray,
    raw_upper: np.ndarray,
    alpha: float,
    multivariate_strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate raw UQ scales with per-horizon, per-feature CP scores."""
    calibration_scale = np.maximum((calibration_upper - calibration_lower) / 2.0, 1e-6)
    test_scale = np.maximum((raw_upper - raw_lower) / 2.0, 1e-6)
    calibrator = FlatConformal(alpha=alpha, multivariate_strategy=multivariate_strategy)
    calibrator.fit(np.abs(calibration_predictions - calibration_targets), scales=calibration_scale)
    lower, upper = calibrator.predict(
        torch.from_numpy(predictions).float(), torch.from_numpy(test_scale).float()
    )
    return lower.numpy(), upper.numpy()


def train_backbone_member(args, setting: str, member_seed: int):
    set_random_seed(member_seed, strict=getattr(args, "strict_determinism", False))
    device = select_device(require_gpu=getattr(args, "require_gpu", False))
    encoder, head = build_backbone(args)
    encoder = encoder.to(device)
    head = head.to(device)

    train_data, train_loader = get_data_loader(args, "train")
    vali_data, vali_loader = get_data_loader(args, "val")
    checkpoint_dir = Path(args.checkpoints) / setting
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.MSELoss()
    disentangle_criterion = DisentanglementLoss(
        lambda_cov=args.lambda_cov,
        lambda_temporal=args.lambda_temp,
    ).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=args.learning_rate)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    train_start = time.time()
    history = []

    for epoch in range(args.train_epochs):
        epoch_start = time.time()
        train_loss = train_backbone_epoch(
            encoder=encoder,
            head=head,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            disentangle_criterion=disentangle_criterion,
            device=device,
            args=args,
            progress_description=f"Epoch {epoch + 1}/{args.train_epochs}",
        )
        vali_loss = validate_backbone(encoder, head, vali_loader, criterion, device, args)
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "vali_loss": vali_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })
        improved = early_stopping(vali_loss, encoder, head, path=str(checkpoint_dir))
        print_epoch_summary(
            epoch=epoch + 1, total_epochs=args.train_epochs, train_loss=train_loss,
            validation_loss=vali_loss, learning_rate=optimizer.param_groups[0]["lr"],
            elapsed_seconds=time.time() - epoch_start, improved=improved,
            patience_counter=early_stopping.counter, patience=early_stopping.patience,
        )
        if early_stopping.early_stop:
            break
        adjust_learning_rate(optimizer, epoch + 1, args)

    save_history(checkpoint_dir, history)
    load_backbone_checkpoint(checkpoint_dir, encoder, head, device)
    runtime = {
        "train_seconds": time.time() - train_start,
        "train_samples": len(train_data),
        "validation_samples": len(vali_data),
        "seed": member_seed,
        "epochs_requested": args.train_epochs,
        "epochs_run": len(history),
        "early_stopped": early_stopping.early_stop,
        "best_val_loss": early_stopping.val_loss_min,
    }
    return encoder, head, device, runtime


def load_history(checkpoint_dir: Path) -> Optional[list[dict]]:
    history_path = checkpoint_dir / "history.json"
    if not history_path.exists():
        return None
    return json.loads(history_path.read_text(encoding="utf-8"))


def run_mc_dropout(args, setting: str):
    encoder, head, device, runtime = train_backbone_member(args, setting, args.seed)
    train_data, _ = get_data_loader(args, "train")
    cal_data, cal_loader = get_data_loader(args, "cal")
    test_data, test_loader = get_data_loader(args, "test")
    test_start = time.time()
    preds, trues, lower, upper = evaluate_mc_dropout(encoder, head, test_loader, device, args)
    interval_origin = "raw_uq"
    if args.uq_interval_mode == "conformalized":
        cal_preds, cal_trues, cal_lower, cal_upper = evaluate_mc_dropout(
            encoder, head, cal_loader, device, args
        )
        lower, upper = conformalize_uq_intervals(
            preds, cal_trues, cal_preds, cal_lower, cal_upper, lower, upper,
            args.conformal_alpha, args.multivariate_strategy,
        )
        interval_origin = "conformalized_uq"
    point_metrics, interval_metrics = compute_metrics(
        args, preds, trues, lower=lower, upper=upper, y_train=train_data.data_y,
        interval_origin=interval_origin,
    )
    runtime.update(
        {
            "test_seconds": time.time() - test_start,
            "test_samples": len(test_data),
            "calibration_samples": len(cal_data),
        }
    )
    history = load_history(Path(args.checkpoints) / setting)
    return save_result_artifacts(
        args, setting, point_metrics, interval_metrics, preds, trues, runtime,
        lower=lower, upper=upper, history=history, y_train=train_data.data_y,
    )


def parse_ensemble_seeds(args) -> list[int]:
    if args.ensemble_seeds:
        seeds = [int(token.strip()) for token in args.ensemble_seeds.split(",") if token.strip()]
    else:
        seeds = [args.seed + 1009 * offset for offset in range(args.ensemble_size)]
    if len(seeds) < 2:
        raise ValueError("deep_ensemble requires at least two member seeds.")
    return seeds


def run_deep_ensemble(args, setting: str):
    member_seeds = parse_ensemble_seeds(args)
    train_data, _ = get_data_loader(args, "train")
    cal_data, cal_loader = get_data_loader(args, "cal")
    test_data, test_loader = get_data_loader(args, "test")
    ensemble_forecasts = []
    calibration_forecasts = []
    reference_targets = None
    calibration_targets = None
    member_runtimes = []
    member_histories = []
    total_start = time.time()
    test_inference_seconds = 0.0
    calibration_inference_seconds = 0.0

    for index, member_seed in enumerate(member_seeds, start=1):
        member_setting = build_member_setting(setting, member_seed, index)
        print(f"\nMember {index}/{len(member_seeds)} | seed={member_seed}")
        encoder, head, device, runtime = train_backbone_member(args, member_setting, member_seed)
        test_start = time.time()
        preds, trues = evaluate_backbone_point(encoder, head, test_loader, device, args)
        test_inference_seconds += time.time() - test_start
        calibration_start = time.time()
        cal_preds, cal_trues = evaluate_backbone_point(encoder, head, cal_loader, device, args)
        calibration_inference_seconds += time.time() - calibration_start
        ensemble_forecasts.append(torch.from_numpy(preds))
        calibration_forecasts.append(torch.from_numpy(cal_preds))
        if reference_targets is None:
            reference_targets = trues
        elif not np.allclose(reference_targets, trues):
            raise RuntimeError("Deep ensemble members produced mismatched test targets.")
        if calibration_targets is None:
            calibration_targets = cal_trues
        elif not np.allclose(calibration_targets, cal_trues):
            raise RuntimeError("Deep ensemble members produced mismatched calibration targets.")
        member_runtimes.append(runtime)
        member_histories.append({
            "member_index": index,
            "seed": member_seed,
            "history": load_history(Path(args.checkpoints) / member_setting),
        })

    wrapper = DeepEnsemble(alpha=args.conformal_alpha)
    mean, lower, upper = wrapper.predict(ensemble_forecasts)
    preds = mean.numpy()
    lower_np = lower.numpy()
    upper_np = upper.numpy()
    interval_origin = "raw_uq"
    if args.uq_interval_mode == "conformalized":
        cal_mean, cal_lower, cal_upper = wrapper.predict(calibration_forecasts)
        lower_np, upper_np = conformalize_uq_intervals(
            preds,
            calibration_targets,
            cal_mean.numpy(),
            cal_lower.numpy(),
            cal_upper.numpy(),
            lower_np,
            upper_np,
            args.conformal_alpha,
            args.multivariate_strategy,
        )
        interval_origin = "conformalized_uq"
    point_metrics, interval_metrics = compute_metrics(
        args, preds, reference_targets, lower=lower_np, upper=upper_np, y_train=train_data.data_y,
        interval_origin=interval_origin,
    )
    runtime = {
        "train_seconds": sum(item["train_seconds"] for item in member_runtimes),
        "fit_seconds": sum(item["train_seconds"] for item in member_runtimes),
        "test_seconds": test_inference_seconds,
        "test_inference_seconds": test_inference_seconds,
        "calibration_inference_seconds": calibration_inference_seconds,
        "total_seconds": time.time() - total_start,
        "train_samples": member_runtimes[0]["train_samples"],
        "validation_samples": member_runtimes[0]["validation_samples"],
        "test_samples": len(test_data),
        "calibration_samples": len(cal_data),
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
        history=member_histories,
        y_train=train_data.data_y,
    )


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="YAML/JSON config file")
    pre_args, _ = pre_parser.parse_known_args()
    config_defaults = load_config_defaults(pre_args.config)
    cli_options = provided_cli_options(sys.argv[1:])

    parser = argparse.ArgumentParser(description="Baseline Experiment Runner", parents=[pre_parser])
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
    parser.add_argument("--require_gpu", action="store_true",
                        help="fail instead of falling back to CPU when no GPU is available")
    parser.add_argument("--require_clean_git", action="store_true",
                        help="require a clean committed worktree for a publication run")
    parser.add_argument("--no_progress", action="store_true",
                        help="disable terminal progress bars for CI or captured logs")
    parser.add_argument("--strict_artifacts", action="store_true",
                        help="exit nonzero when a run produces structurally invalid artifacts")
    parser.add_argument("--strict_determinism", action="store_true",
                        help="require deterministic PyTorch algorithms and record the setting")
    parser.add_argument("--immutable_artifacts", action="store_true",
                        help="write a content-addressed run and refuse existing artifacts")
    parser.add_argument("--evidence_role", choices=("development", "selection", "confirmation"),
                        default="development",
                        help="evidence tier recorded in protocol; confirmation enforces sealed-run safeguards")
    # Retired alias for --strict_artifacts; accepted silently for existing scripts.
    parser.add_argument("--strict_sanity", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--train_epochs", type=int, default=20, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size (matches the CISSN default so baselines stay comparable)")
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--lradj", type=str, default="cosine",
                        help="lr schedule [type1, type2, cosine]; must match run_benchmark.py's "
                             "default so CISSN and baselines train under the same protocol")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="max gradient norm; <=0 disables clipping")

    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--conformal_alpha", type=float, default=0.1, help="interval significance level")
    parser.add_argument("--multivariate_strategy", choices=("per_feature", "max"), default="per_feature",
                        help="shared conformal score geometry")
    parser.add_argument("--uq_interval_mode", choices=("conformalized", "raw"), default="conformalized",
                        help="report calibrated UQ intervals by default; raw UQ is secondary")
    parser.add_argument("--cal_fraction", type=float, default=0.2,
                        help="fraction of the canonical train window carved out as the calibration split")
    parser.add_argument("--kernel_size", type=int, default=25, help="DLinear moving-average kernel size")
    parser.add_argument("--patch_len", type=int, default=16, help="PatchTST patch length")
    parser.add_argument("--patch_stride", type=int, default=8, help="PatchTST patch stride")
    parser.add_argument("--nhead", type=int, default=8, help="PatchTST attention heads")
    parser.add_argument("--num_layers", type=int, default=None, help="optional model layer count override")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="PatchTST feedforward dimension")
    parser.add_argument("--mc_samples", type=int, default=50, help="MC-Dropout stochastic forward passes")
    parser.add_argument("--ensemble_size", type=int, default=3, help="Deep Ensemble member count when ensemble_seeds is not provided")
    parser.add_argument("--ensemble_seeds", type=str, default="", help="comma-separated Deep Ensemble member seeds")

    validate_config_defaults(parser, config_defaults)
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    if vars(args).pop("strict_sanity", False):
        args.strict_artifacts = True

    protected = set(config_defaults) | cli_options
    apply_dataset_defaults(args, protected)
    if args.features == "MS" and "c_out" not in protected:
        args.c_out = 1
    if args.model in BACKBONE_MODELS and args.state_dim != 5:
        raise ValueError(f"{args.model} requires state_dim=5; got {args.state_dim}.")
    if args.model == "deep_ensemble" and args.ensemble_seeds:
        args.ensemble_size = len(parse_ensemble_seeds(args))
    validate_runtime_args(args)
    return args


def main() -> None:
    args = parse_args()
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

    print_run_header("CISSN baseline", args, setting)

    if args.model in POINT_MODELS:
        result_dir = run_point_baseline(args, setting)
    elif args.model == "mc_dropout":
        result_dir = run_mc_dropout(args, setting)
    elif args.model == "deep_ensemble":
        result_dir = run_deep_ensemble(args, setting)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if final_results_dir is not None:
        result_dir = finalize_result_directory(args.results_dir, final_results_dir, setting)

    print(f"Saved artifacts to {result_dir}")


if __name__ == "__main__":
    main()
