"""
MC-Dropout uncertainty baseline.

Uses the CISSN forecast head with dropout enabled at inference time.
Samples N forward passes and uses the empirical standard deviation of
predictions as the interval half-width.

Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016.
"""
import torch
import torch.nn as nn
from typing import Tuple


class MCDropout:
    """
    Monte Carlo Dropout for prediction interval construction.
    Requires a forecast head with dropout layers applied during inference.
    """

    def __init__(self, n_samples: int = 50, alpha: float = 0.1):
        """
        Args:
            n_samples: Number of stochastic forward passes.
            alpha: Significance level (interval = mean ± z_{1-alpha/2} * std).
        """
        self.n_samples = n_samples
        self.alpha = alpha
        # z-score for Gaussian interval at coverage 1-alpha
        from scipy.stats import norm
        self.z_score = norm.ppf(1 - alpha / 2)
        self.calibrated = True

    @staticmethod
    def _enable_dropout(model: nn.Module):
        """Force dropout layers into training mode for stochastic inference."""
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def predict(
        self,
        model: nn.Module,
        head: nn.Module,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate prediction intervals via MC sampling.

        Args:
            model: CISSN encoder (DisentangledStateEncoder)
            head: CISSN forecast head (ForecastHead)
            state: Pre-computed final state (batch, state_dim)

        Returns:
            mean: Mean prediction over N samples
            lower: Mean - z * std
            upper: Mean + z * std
        """
        self._enable_dropout(head)
        samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                samples.append(head(state))
        samples = torch.stack(samples, dim=0)  # (N, B, H, D_out)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        z = torch.tensor(self.z_score, device=state.device, dtype=state.dtype)
        lower = mean - z * std
        upper = mean + z * std
        return mean, lower, upper
