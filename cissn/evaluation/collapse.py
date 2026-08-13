"""
Forecast-collapse diagnostics.

Under MSE with a weakly-predictable target, the loss-minimising forecast is
attenuated toward the conditional mean: a model can lower training loss by
shrinking its output variance rather than by tracking the signal. The failure
is invisible in the loss curve -- MSE keeps improving -- but the model ends up
emitting a near-constant, which no point metric distinguishes from "hard
dataset".

``variance_ratio`` = var(pred) / var(true) makes it visible directly:

    ~1.0   forecast is appropriately dispersed
    ~0.5   noticeable shrinkage
    <0.1   collapsed; the forecast is close to a constant

Tracked per epoch, a ratio that *falls* while MSE improves is direct evidence
that training is buying loss with shrinkage. That is a distinct failure from an
information bottleneck (which caps forecast *rank*, not amplitude), and it calls
for a different fix, so the two must not be conflated.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class DispersionAccumulator:
    """Streaming var(pred)/var(true) over batches.

    Accumulates raw moments so the ratio can be computed across a whole split
    without holding predictions and targets in memory.
    """

    __slots__ = ("_n", "_pred_sum", "_pred_sq", "_true_sum", "_true_sq", "_cross")

    def __init__(self) -> None:
        self._n = 0
        self._pred_sum = 0.0
        self._pred_sq = 0.0
        self._true_sum = 0.0
        self._true_sq = 0.0
        self._cross = 0.0

    def update(self, preds, trues) -> None:
        """Accumulate one batch. Accepts torch tensors or numpy arrays."""
        preds = _as_flat_float64(preds)
        trues = _as_flat_float64(trues)
        if preds.shape != trues.shape:
            raise ValueError(
                f"preds and trues must share a shape; got {preds.shape} and {trues.shape}."
            )
        self._n += preds.size
        self._pred_sum += float(preds.sum())
        self._pred_sq += float(np.dot(preds, preds))
        self._true_sum += float(trues.sum())
        self._true_sq += float(np.dot(trues, trues))
        self._cross += float(np.dot(preds, trues))

    def _variances(self) -> tuple[float, float, float]:
        n = self._n
        pred_mean = self._pred_sum / n
        true_mean = self._true_sum / n
        pred_var = max(self._pred_sq / n - pred_mean**2, 0.0)
        true_var = max(self._true_sq / n - true_mean**2, 0.0)
        covariance = self._cross / n - pred_mean * true_mean
        return pred_var, true_var, covariance

    def summary(self) -> dict[str, Optional[float]]:
        """Return dispersion diagnostics, or None values when no data was seen."""
        if self._n == 0:
            return {"variance_ratio": None, "pred_std": None, "true_std": None, "corr": None}

        pred_var, true_var, covariance = self._variances()
        pred_std = float(np.sqrt(pred_var))
        true_std = float(np.sqrt(true_var))
        denom = pred_std * true_std
        return {
            # A constant forecast has zero variance; the ratio is 0, not undefined.
            "variance_ratio": float(pred_var / true_var) if true_var > 0 else None,
            "pred_std": pred_std,
            "true_std": true_std,
            "corr": float(covariance / denom) if denom > 1e-12 else None,
        }


def _as_flat_float64(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def dispersion_summary(preds, trues) -> dict[str, Optional[float]]:
    """One-shot dispersion diagnostics for complete prediction/target arrays."""
    accumulator = DispersionAccumulator()
    accumulator.update(preds, trues)
    return accumulator.summary()
