"""Shared training/evaluation helpers for publication baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from cissn.evaluation.metrics import mean_absolute_error, mean_squared_error


@dataclass
class BaselineEvalResult:
    mse: float
    mae: float
    predictions: np.ndarray
    targets: np.ndarray


def slice_forecast(outputs: torch.Tensor, batch_y: torch.Tensor, pred_len: int, features: str):
    """Apply the same horizon/channel policy used by the CISSN runner."""
    f_dim = -1 if features == "MS" else 0
    return outputs[:, -pred_len:, f_dim:], batch_y[:, -pred_len:, f_dim:]


def train_baseline_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    pred_len: int,
    features: str,
    grad_clip: float = 1.0,
) -> float:
    """Train one epoch for baselines exposing `forward(x) -> forecast`."""
    model.train()
    losses: list[float] = []
    for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
        batch_x = batch_x.float().to(device, non_blocking=True)
        batch_y = batch_y.float().to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs, targets = slice_forecast(model(batch_x), batch_y, pred_len, features)
        loss = criterion(outputs, targets)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    if not losses:
        raise RuntimeError("Baseline training loader produced no batches.")
    return float(np.mean(losses))


def evaluate_baseline(
    model: nn.Module,
    loader,
    device: torch.device,
    pred_len: int,
    features: str,
) -> BaselineEvalResult:
    """Evaluate point metrics for baselines under the shared slicing policy."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            outputs, targets = slice_forecast(model(batch_x), batch_y, pred_len, features)
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
