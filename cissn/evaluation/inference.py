"""Paired inference on per-origin interval scores."""
from __future__ import annotations

import numpy as np


def moving_block_bootstrap_ratio_ci(
    state: np.ndarray,
    flat: np.ndarray,
    block: int,
    n_boot: int,
    seed: int,
    confidence: float = 0.95,
) -> dict:
    """Paired ratio-of-means interval; resample the denominator with the numerator."""
    state, flat = np.asarray(state, dtype=np.float64), np.asarray(flat, dtype=np.float64)
    if (state.ndim != 1 or state.size == 0 or flat.shape != state.shape
            or not np.all(np.isfinite(state)) or not np.all(np.isfinite(flat))
            or np.any(flat <= 0)):
        raise ValueError("Provide aligned finite one-dimensional scores with positive flat scores.")
    n = state.size
    if not 1 <= block <= n or n_boot < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid block length, resample count, or confidence.")
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_boot)
    # Chunk draws so memory does not grow as n_origins * n_boot.
    for start in range(0, n_boot, 64):
        size = min(64, n_boot - start)
        starts = rng.integers(0, n - block + 1, size=(size, -(-n // block)))
        indices = (starts[:, :, None] + np.arange(block)).reshape(size, -1)[:, :n]
        ratios[start:start + size] = state[indices].mean(axis=1) / flat[indices].mean(axis=1) - 1
    tail = (1 - confidence) / 2
    low, high = np.quantile(ratios, [tail, 1 - tail])
    return {"mean": float(state.mean() / flat.mean() - 1), "ci_low": float(low), "ci_high": float(high),
            "n_origins": n, "block_len": int(block), "n_boot": int(n_boot), "seed": int(seed),
            "confidence": float(confidence)}


def moving_block_bootstrap_ci(
    delta: np.ndarray,
    block_len: int,
    n_boot: int,
    seed: int,
    confidence: float = 0.95,
) -> dict:
    """Percentile CI for the mean of a chronologically ordered paired difference.

    Forecast origins at stride 1 share targets whenever they are closer than the
    horizon, so an i.i.d. resample understates the variance. Resampling blocks
    of consecutive origins keeps that dependence inside each block. Choose
    ``block_len >= pred_len``; a shorter block still cuts the dependence at its
    edges and gives an interval that is too narrow.
    """
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 1 or delta.size == 0 or not np.all(np.isfinite(delta)):
        raise ValueError("delta must be a non-empty, finite, one-dimensional array.")
    n = delta.size
    if not 1 <= block_len <= n:
        raise ValueError(f"block_len must be in [1, {n}]; got {block_len}.")
    if n_boot < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("n_boot must be positive and confidence must be in (0, 1).")

    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block_len)
    starts = rng.integers(0, n - block_len + 1, size=(n_boot, n_blocks))
    index = (starts[:, :, None] + np.arange(block_len)).reshape(n_boot, -1)[:, :n]
    means = delta[index].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return {
        "mean": float(delta.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_origins": int(n),
        "block_len": int(block_len),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "confidence": float(confidence),
    }
