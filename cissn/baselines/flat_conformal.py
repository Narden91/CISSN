"""
Flat (marginal) conformal prediction baseline.

Uses the same CISSN encoder and forecast head, but applies a single global
quantile (no state clustering) for interval construction. This isolates the
contribution of state-conditioning in SCCP.
"""
import logging
import numpy as np
import torch
from typing import Union, Tuple

from cissn.conformal import split_conformal_q_level

logger = logging.getLogger(__name__)


class FlatConformal:
    """
    Marginal conformal prediction: one global quantile for all predictions.
    Uses the same finite-sample correction as SCCP but without clustering.
    """

    VALID_STRATEGIES = {"per_feature", "max"}

    def __init__(self, alpha: float = 0.1, multivariate_strategy: str = "per_feature"):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
        if multivariate_strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Unknown multivariate_strategy={multivariate_strategy!r}.")
        self.alpha = alpha
        self.multivariate_strategy = multivariate_strategy
        self.quantile_ = None
        self.calibrated = False
        self.coverage_scope = "simultaneous" if multivariate_strategy == "max" else "marginal"

    def fit(self, residuals: Union[torch.Tensor, np.ndarray], scales: Union[torch.Tensor, np.ndarray, None] = None):
        """Calibrate horizon-feature scores, optionally normalized by a frozen scale."""
        if isinstance(residuals, torch.Tensor):
            residuals = residuals.detach().cpu().numpy()
        residuals = np.asarray(residuals)
        if residuals.size == 0:
            raise ValueError("residuals must contain at least one sample.")
        if np.any(residuals < 0):
            raise ValueError("residuals must be non-negative absolute errors.")
        if scales is not None:
            if isinstance(scales, torch.Tensor):
                scales = scales.detach().cpu().numpy()
            scales = np.asarray(scales)
            if scales.shape != residuals.shape:
                raise ValueError("scales must have the same shape as residuals.")
            residuals = residuals / np.maximum(scales, 1e-6)

        n = residuals.shape[0]
        q_level = split_conformal_q_level(n, self.alpha)
        if self.multivariate_strategy == "max" and residuals.ndim > 1:
            residuals = residuals.reshape(n, -1).max(axis=1)
        self.quantile_ = np.quantile(residuals, q_level, axis=0, method='higher')
        self.calibrated = True
        logger.info("Flat CP calibration: n=%d, alpha=%s, strategy=%s", n, self.alpha, self.multivariate_strategy)

    def predict(self, point_forecasts: torch.Tensor, scales: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate constant-width prediction intervals."""
        if not self.calibrated:
            raise RuntimeError("Flat CP not calibrated. Call fit() first.")
        if not isinstance(point_forecasts, torch.Tensor):
            raise TypeError("point_forecasts must be a torch.Tensor.")

        q = torch.as_tensor(self.quantile_, device=point_forecasts.device, dtype=point_forecasts.dtype)
        trailing_shape = tuple(point_forecasts.shape[1:])
        if q.ndim == 0:
            q = q.reshape((1,) * point_forecasts.ndim)
        elif tuple(q.shape) == trailing_shape:
            q = q.unsqueeze(0)
        elif q.ndim == 1 and point_forecasts.ndim == 3 and q.shape[0] == point_forecasts.shape[-1]:
            q = q.reshape(1, 1, -1)
        else:
            raise ValueError(
                "point_forecasts trailing shape is incompatible with calibrated quantiles: "
                f"quantile shape={tuple(q.shape)}, forecast shape={trailing_shape}."
            )
        if scales is not None:
            if not isinstance(scales, torch.Tensor):
                scales = torch.as_tensor(scales, device=point_forecasts.device, dtype=point_forecasts.dtype)
            if scales.shape != point_forecasts.shape:
                raise ValueError("scales must have the same shape as point_forecasts.")
            q = q * scales
        lower = point_forecasts - q
        upper = point_forecasts + q
        return lower, upper
