"""Calendar-phase Mondrian intervals with a fixed, recorded fallback policy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cissn.conformal.state_conditional import split_conformal_q_level


def phase_from_timestamps(timestamps, period: int) -> np.ndarray:
    """Map first-target timestamps to daily slots, weekdays, or ISO weeks.

    Solar's loader supplies a deterministic date index, not observed timestamps.
    ISO week 53 shares bin zero with week 1 for the 52-bin annual partition.
    """
    dates = pd.DatetimeIndex(timestamps)
    if dates.hasnans:
        raise ValueError("Phase timestamps must not contain missing values.")
    if period in (24, 96, 144):
        minutes = dates.hour * 60 + dates.minute
        return np.asarray(minutes // (1440 // period), dtype=np.int64)
    if period == 7:
        return np.asarray(dates.dayofweek, dtype=np.int64)
    if period == 52:
        return (np.asarray(dates.isocalendar().week, dtype=np.int64) - 1) % 52
    raise ValueError(f"Unsupported phase period: {period}.")


class PhaseConformal:
    """Calibrate each phase, optionally crossed with a frozen state cluster.

    Groups below max(5, ceil(1/alpha)) use the phase quantile when available,
    otherwise the paired global quantile. No test labels enter fitting.
    """

    def __init__(self, alpha: float, period: int):
        if not 0 < alpha < 1 or period < 1:
            raise ValueError("alpha must be in (0, 1) and period must be positive.")
        self.alpha, self.period = alpha, period

    def _phases(self, values, n):
        phases = np.asarray(values)
        if phases.shape != (n,) or not np.issubdtype(phases.dtype, np.integer):
            raise ValueError("Provide one integer phase per origin.")
        if np.any((phases < 0) | (phases >= self.period)):
            raise ValueError("Phase is outside the configured period.")
        return phases

    def fit(self, residuals, phases, clusters=None, n_clusters=None):
        scores = np.asarray(residuals)
        if scores.ndim < 1 or scores.shape[0] == 0 or not np.all(np.isfinite(scores)) or np.any(scores < 0):
            raise ValueError("Residuals must be finite, non-negative, and non-empty.")
        phases = self._phases(phases, len(scores))
        minimum = max(5, int(np.ceil(1 / self.alpha)))

        def quantile(values):
            return np.quantile(values, split_conformal_q_level(len(values), self.alpha), axis=0, method="higher")

        self.global_quantile_ = quantile(scores)
        self.phase_quantiles_, self.group_counts_, self.fallbacks_ = {}, {}, {}
        for phase in range(self.period):
            mask = phases == phase
            count = int(mask.sum())
            self.phase_quantiles_[phase] = quantile(scores[mask]) if count >= minimum else self.global_quantile_
            self.group_counts_[str(phase)] = count
            self.fallbacks_[str(phase)] = None if count >= minimum else "paired_global"
        self.cross_quantiles_ = None
        if clusters is not None:
            clusters = np.asarray(clusters)
            if (clusters.shape != phases.shape or not np.issubdtype(clusters.dtype, np.integer)
                    or n_clusters is None or n_clusters < 1 or np.any((clusters < 0) | (clusters >= n_clusters))):
                raise ValueError("Provide valid frozen cluster labels and n_clusters.")
            self.cross_quantiles_ = {}
            for phase in range(self.period):
                for cluster in range(n_clusters):
                    key = (phase, cluster)
                    mask = (phases == phase) & (clusters == cluster)
                    count = int(mask.sum())
                    self.cross_quantiles_[key] = quantile(scores[mask]) if count >= minimum else self.phase_quantiles_[phase]
                    name = f"{phase}:{cluster}"
                    self.group_counts_[name] = count
                    self.fallbacks_[name] = None if count >= minimum else (
                        "phase" if self.fallbacks_[str(phase)] is None else "paired_global")
        self.calibration_samples_ = len(scores)
        return self

    def half_widths(self, phases, clusters=None):
        if not hasattr(self, "phase_quantiles_"):
            raise RuntimeError("Fit phase calibration before prediction.")
        phases = self._phases(phases, len(phases))
        if self.cross_quantiles_ is None:
            if clusters is not None:
                raise ValueError("This calibrator has no state partition.")
            return np.stack([self.phase_quantiles_[int(p)] for p in phases])
        clusters = np.asarray(clusters)
        if clusters.shape != phases.shape:
            raise ValueError("Provide one cluster per phase.")
        return np.stack([self.cross_quantiles_[(int(p), int(k))] for p, k in zip(phases, clusters)])

    def stats(self):
        return {"n_calibration": self.calibration_samples_, "minimum_group_count": max(5, int(np.ceil(1 / self.alpha))),
                "counts": self.group_counts_, "fallbacks": self.fallbacks_}
