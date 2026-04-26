"""
Deep Ensemble uncertainty baseline.

Trains M independent CISSN models from different random seeds and uses the
empirical variance of their predictions to construct intervals.

Reference: Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty", NeurIPS 2017.
"""
import torch
import numpy as np
from typing import List, Tuple
from scipy.stats import norm


class DeepEnsemble:
    """
    Deep Ensemble for prediction interval construction.
    Assumes M independently trained encoder + head pairs.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Significance level for Gaussian intervals.
        """
        self.alpha = alpha
        self.z_score = norm.ppf(1 - alpha / 2)
        self.calibrated = True

    def predict(
        self,
        ensemble_forecasts: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate prediction intervals from an ensemble of forecasts.

        Args:
            ensemble_forecasts: List of M forecast tensors, each (B, H, D_out).

        Returns:
            mean: Mean prediction over ensemble
            lower: Mean - z * std
            upper: Mean + z * std
        """
        stacked = torch.stack(ensemble_forecasts, dim=0)  # (M, B, H, D_out)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
        device = mean.device
        z = torch.tensor(self.z_score, device=device, dtype=mean.dtype)
        lower = mean - z * std
        upper = mean + z * std
        return mean, lower, upper
