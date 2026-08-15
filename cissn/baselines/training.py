"""Shared training/evaluation helpers for publication baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from cissn.evaluation.metrics import mean_absolute_error, mean_squared_error
from cissn.models.revin import RevIN
from cissn.utils.progress import track


@dataclass
class BaselineEvalResult:
    mse: float
    mae: float
    predictions: np.ndarray
    targets: np.ndarray


def forecast_channel_index(features: str) -> int:
    """The f_dim convention shared by every runner: last channel under MS, all under M/S."""
    return -1 if features == "MS" else 0


def slice_forecast(outputs: torch.Tensor, batch_y: torch.Tensor, pred_len: int, features: str):
    """Apply the same horizon/channel policy used by the CISSN runner."""
    f_dim = forecast_channel_index(features)
    return outputs[:, -pred_len:, f_dim:], batch_y[:, -pred_len:, f_dim:]


def denormalize_forecast(outputs: torch.Tensor, revin: RevIN, features: str) -> torch.Tensor:
    """Restore RevIN statistics to an already-sliced forecast.

    Mirrors run_benchmark.py's Experiment._forward_and_slice: slice first,
    then denormalise with statistics for exactly those channels. Under MS the
    forecast is one (the last) column, so it must be rescaled with the
    target's own statistics via select_channels(), never feature 0's --
    RevIN._denormalize refuses a channel-count mismatch rather than silently
    broadcasting the wrong channel's stats.
    """
    f_dim = forecast_channel_index(features)
    scaler = revin.select_channels(-1) if f_dim == -1 else revin
    return scaler(outputs, "denorm")


def train_baseline_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    pred_len: int,
    features: str,
    grad_clip: float = 1.0,
    show_progress: bool = False,
    progress_description: str = "Training",
    probabilistic: bool = False,
    revin: Optional[RevIN] = None,
) -> float:
    """Train one epoch for baselines exposing `forward(x) -> forecast`."""
    model.train()
    if revin is not None:
        revin.train()
    total_loss = torch.zeros((), device=device)
    total_weight = 0
    for batch_x, batch_y, _batch_x_mark, _batch_y_mark in track(
        loader, description=progress_description, total=len(loader), enabled=show_progress
    ):
        batch_x = batch_x.float().to(device, non_blocking=True)
        batch_y = batch_y.float().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        model_input = revin(batch_x, "norm") if revin is not None else batch_x
        if probabilistic:
            mean, log_sigma = model.predict_distribution(model_input)
            outputs, targets = slice_forecast(mean, batch_y, pred_len, features)
            log_sigma, _ = slice_forecast(log_sigma, batch_y, pred_len, features)
            if revin is not None:
                outputs = denormalize_forecast(outputs, revin, features)
            loss = model.gaussian_nll(outputs, targets, log_sigma)
        else:
            outputs, targets = slice_forecast(model(model_input), batch_y, pred_len, features)
            if revin is not None:
                outputs = denormalize_forecast(outputs, revin, features)
            loss = criterion(outputs, targets)
        loss.backward()
        parameters = list(model.parameters()) + (list(revin.parameters()) if revin is not None else [])
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        optimizer.step()
        batch_weight = outputs.numel()
        total_loss += loss.detach() * batch_weight
        total_weight += batch_weight
    if total_weight == 0:
        raise RuntimeError("Baseline training loader produced no batches.")
    return float((total_loss / total_weight).item())


def evaluate_baseline(
    model: nn.Module,
    loader,
    device: torch.device,
    pred_len: int,
    features: str,
    revin: Optional[RevIN] = None,
) -> BaselineEvalResult:
    """Evaluate point metrics for baselines under the shared slicing policy."""
    model.eval()
    if revin is not None:
        revin.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            model_input = revin(batch_x, "norm") if revin is not None else batch_x
            outputs, targets = slice_forecast(model(model_input), batch_y, pred_len, features)
            if revin is not None:
                outputs = denormalize_forecast(outputs, revin, features)
            preds.append(outputs.detach().cpu().numpy())
            trues.append(targets.detach().cpu().numpy())
    if not preds:
        raise RuntimeError("Baseline evaluation loader produced no batches.")
    pred_arr = np.concatenate(preds, axis=0)
    true_arr = np.concatenate(trues, axis=0)
    return BaselineEvalResult(
        mse=mean_squared_error(true_arr.flatten(), pred_arr.flatten()),
        mae=mean_absolute_error(true_arr.flatten(), pred_arr.flatten()),
        predictions=pred_arr,
        targets=true_arr,
    )
