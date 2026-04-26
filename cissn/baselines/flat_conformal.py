"""
Flat (marginal) conformal prediction baseline.

Uses the same CISSN encoder and forecast head, but applies a single global
quantile (no state clustering) for interval construction. This isolates the
contribution of state-conditioning in SCCP.
"""
import numpy as np
import torch
from typing import Union, Tuple


class FlatConformal:
    """
    Marginal conformal prediction: one global quantile for all predictions.
    Uses the same finite-sample correction as SCCP but without clustering.
    """

    def __init__(self, alpha: float = 0.1):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
        self.alpha = alpha
        self.quantile_ = None
        self.calibrated = False

    def fit(self, residuals: Union[torch.Tensor, np.ndarray]):
        """Calibrate: compute the (1-alpha) quantile of absolute residuals."""
        if isinstance(residuals, torch.Tensor):
            residuals = residuals.detach().cpu().numpy()
        residuals = np.asarray(residuals).flatten()
        if residuals.size == 0:
            raise ValueError("residuals must contain at least one sample.")
        if np.any(residuals < 0):
            raise ValueError("residuals must be non-negative absolute errors.")

        n = residuals.shape[0]
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.quantile_ = float(np.quantile(residuals, q_level, method='higher'))
        self.calibrated = True
        print(f"Flat CP calibration: q={self.quantile_:.4f}, n={n}, alpha={self.alpha}")

    def predict(
        self,
        point_forecasts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate constant-width prediction intervals."""
        if not self.calibrated:
            raise RuntimeError("Flat CP not calibrated. Call fit() first.")
        if not isinstance(point_forecasts, torch.Tensor):
            raise TypeError("point_forecasts must be a torch.Tensor.")

        q = torch.tensor(self.quantile_, device=point_forecasts.device,
                         dtype=point_forecasts.dtype)
        while q.ndim < point_forecasts.ndim:
            q = q.unsqueeze(-1)
        lower = point_forecasts - q
        upper = point_forecasts + q
        return lower, upper
