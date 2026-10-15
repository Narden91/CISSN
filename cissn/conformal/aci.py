"""Adaptive conformal inference (Gibbs and Candes, 2021) with delayed feedback.

ACI moves the miscoverage level of each cell after every scored forecast:
``alpha_t+1 = alpha_t + gamma * (alpha - miss_t)``. The half-width at origin t is the
split-conformal quantile of the calibration scores at level ``alpha_t``, computed exactly
as ``FlatConformal`` does, so ``gamma = 0`` reproduces flat CP.

A multi-step forecast is fully observed only ``horizon`` steps after its origin. Origin
``i`` therefore updates the levels only for origins ``j`` with ``origins[i] + horizon <=
origins[j]``; using it earlier would read targets that lie after the forecast origin.

Unlike split conformal prediction, ACI uses the realised test errors as they arrive. A
comparison with flat CP must say that ACI uses more information.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def adaptive_conformal_quantiles(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    origins: np.ndarray,
    horizon: int,
    alpha: float,
    gamma: float,
) -> tuple[np.ndarray, int]:
    """Per-origin, per-cell ACI half-widths.

    Args:
        calibration_scores: absolute calibration residuals, shape (n, H, C).
        test_scores: absolute test residuals in origin order, shape (T, H, C).
        origins: window start index of each test origin, increasing, shape (T,).
        horizon: forecast horizon H in time steps.
        alpha: target miscoverage level.
        gamma: ACI step size.

    Returns:
        Half-widths of shape (T, H, C), and the number of (origin, cell) pairs whose level
        asked for more than the largest calibration score. Those pairs use the largest
        score, as the capped split-conformal level does.
    """
    if calibration_scores.ndim != 3 or test_scores.ndim != 3:
        raise ValueError("calibration_scores and test_scores must have shape (n, H, C).")
    if calibration_scores.shape[1:] != test_scores.shape[1:]:
        raise ValueError("calibration and test scores must share the (H, C) cell shape.")
    if origins.shape != (test_scores.shape[0],) or np.any(np.diff(origins) <= 0):
        raise ValueError("origins must be increasing, one per test origin.")
    if not 0 < alpha < 1 or gamma < 0:
        raise ValueError("alpha must lie in (0, 1) and gamma must be non-negative.")

    n = calibration_scores.shape[0]
    ordered = np.sort(calibration_scores, axis=0)
    level = np.full(test_scores.shape[1:], alpha, dtype=np.float64)
    half_widths = np.empty_like(test_scores, dtype=np.float64)
    pending: deque[int] = deque()
    n_capped = 0
    for j in range(test_scores.shape[0]):
        while pending and origins[pending[0]] + horizon <= origins[j]:
            i = pending.popleft()
            level += gamma * (alpha - (test_scores[i] > half_widths[i]))
        rank = np.ceil((n + 1) * (1.0 - level))
        n_capped += int((rank > n).sum())
        # Same level and the same "higher" index rule as np.quantile in FlatConformal.fit.
        q_level = np.minimum(rank / (n + 1), 1.0)
        index = np.clip(np.ceil(q_level * (n - 1)), 0, n - 1).astype(np.int64)
        half_widths[j] = np.take_along_axis(ordered, index[None], axis=0)[0]
        pending.append(j)
    return half_widths, n_capped
