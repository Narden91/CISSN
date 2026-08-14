"""
DeepState baseline: RNN-based deep state-space model for probabilistic forecasting.

Encodes the input sequence with a GRU, then decodes a structured latent state
(level + trend + seasonal) via linear observation equations. Gaussian observation
noise is learned per dimension.

This is a lightweight re-implementation capturing the core ideas of Rangapuram et al.
For the full Kalman-filter variant, see the GluonTS implementation.

Reference: Rangapuram et al., "Deep State Space Models for Time Series Forecasting",
    NeurIPS 2018.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

from cissn.models._dynamics import StructuredDecayMixin


class DeepState(StructuredDecayMixin):
    """
    GRU encoder + structured linear state decoder.

    The latent state is partitioned as [level, trend, s_cos, s_sin] (4 dims).
    Each is mapped to the output via a learned observation matrix, then mixed
    with per-step Gaussian noise whose log-variance is also decoded from the GRU.

    Point forecast = observation(state).
    Interval forecast = point ± z * sigma.
    """

    STATE_DIM = 4  # level, trend, seasonal_cos, seasonal_sin
    LOG_SIGMA_MIN = -7.0
    LOG_SIGMA_MAX = 5.0

    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        output_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        alpha: float = 0.1,
    ):
        """
        Args:
            input_dim: Number of input variates.
            pred_len: Forecast horizon.
            output_dim: Number of output variates.
            hidden_dim: GRU hidden size.
            num_layers: Number of GRU layers.
            dropout: Dropout rate (applied between GRU layers).
            alpha: Significance level for Gaussian prediction intervals.
        """
        super().__init__()
        if input_dim <= 0 or pred_len <= 0 or output_dim <= 0:
            raise ValueError("input_dim, pred_len, and output_dim must be positive.")
        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim and num_layers must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.alpha = alpha

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decode structured latent state from final hidden
        self.state_proj = nn.Linear(hidden_dim, self.STATE_DIM * output_dim)

        # Observation: map [level, trend, s_cos, s_sin] → scalar per output dim
        self.C = nn.Parameter(torch.randn(output_dim, self.STATE_DIM) * 0.02)

        # Structured transition: learn decay rates for each state component
        # (uses StructuredDecayMixin, which registers raw_alpha_L, raw_alpha_T,
        # raw_gamma, and omega; no residual component in this 4-state model)
        self._register_decay_params(n_dims=output_dim, include_residual=False)

        # Per-step log noise variance decoded from encoder
        self.log_sigma_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.C)
        nn.init.zeros_(self.log_sigma_proj.bias)

    def _transition_step(self, s: torch.Tensor) -> torch.Tensor:
        """Apply one step of the block-diagonal transition matrix."""
        a_l = self._level_scale()
        a_t = self._trend_scale()
        g = self._gamma()
        c, sn = torch.cos(self.omega), torch.sin(self.omega)
        rot00, rot01 = g * c, -g * sn
        rot10, rot11 = g * sn, g * c

        # s: (B, output_dim, STATE_DIM)
        new_s = torch.stack([
            s[..., 0] * a_l,
            s[..., 1] * a_t,
            s[..., 2] * rot00 + s[..., 3] * rot10,
            s[..., 2] * rot01 + s[..., 3] * rot11,
        ], dim=-1)
        return new_s

    def _forecast_from_state(self, state: torch.Tensor) -> torch.Tensor:
        """Project all linear state transitions across the forecast horizon."""
        steps = torch.arange(1, self.pred_len + 1, device=state.device, dtype=state.dtype).unsqueeze(1)
        level = self._level_scale().pow(steps)
        trend = self._trend_scale().pow(steps)
        gamma = self._gamma().pow(steps)
        angle = steps * self.omega
        cosine, sine = torch.cos(angle), torch.sin(angle)

        level_state = state[..., 0].unsqueeze(1) * level
        trend_state = state[..., 1].unsqueeze(1) * trend
        seasonal_cos = gamma * (state[..., 2].unsqueeze(1) * cosine + state[..., 3].unsqueeze(1) * sine)
        seasonal_sin = gamma * (-state[..., 2].unsqueeze(1) * sine + state[..., 3].unsqueeze(1) * cosine)
        return (
            level_state * self.C[:, 0]
            + trend_state * self.C[:, 1]
            + seasonal_cos * self.C[:, 2]
            + seasonal_sin * self.C[:, 3]
        )

    def _initial_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.encoder(x)
        final_hidden = hidden[-1]
        state = self.state_proj(final_hidden).view(x.size(0), self.output_dim, self.STATE_DIM)
        return state, final_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            forecast: (batch, pred_len, output_dim)
        """
        state, _ = self._initial_state(x)
        return self._forecast_from_state(state)

    def predict_distribution(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return Gaussian forecast mean and log standard deviation."""
        state, final_hidden = self._initial_state(x)
        mean = self._forecast_from_state(state)
        log_sigma = self.log_sigma_proj(final_hidden).clamp(
            min=self.LOG_SIGMA_MIN, max=self.LOG_SIGMA_MAX
        )
        return mean, log_sigma.unsqueeze(1).expand(-1, self.pred_len, -1)

    @staticmethod
    def gaussian_nll(mean: torch.Tensor, target: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """Elementwise Gaussian negative log likelihood, averaged over outputs."""
        variance = torch.exp(2.0 * log_sigma)
        return 0.5 * (((target - mean) ** 2) / variance + 2.0 * log_sigma + math.log(2.0 * math.pi)).mean()

    def predict_interval(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate Gaussian prediction intervals.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            mean, lower, upper — all (batch, pred_len, output_dim)
        """
        mean, log_sigma = self.predict_distribution(x)
        z = math.sqrt(2.0) * torch.erfinv(
            torch.tensor(1.0 - self.alpha, dtype=x.dtype, device=x.device)
        )
        sigma = torch.exp(log_sigma)
        return mean, mean - z * sigma, mean + z * sigma

    def get_contributions(self, state: Optional[torch.Tensor] = None) -> dict:
        """Return current state component names. No gradient-based attribution."""
        return {"components": ["level", "trend", "seasonal_cos", "seasonal_sin"]}
