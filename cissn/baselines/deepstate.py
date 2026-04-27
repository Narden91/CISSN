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

import torch
import torch.nn as nn
from typing import Tuple, Optional


class DeepState(nn.Module):
    """
    GRU encoder + structured linear state decoder.

    The latent state is partitioned as [level, trend, s_cos, s_sin] (4 dims).
    Each is mapped to the output via a learned observation matrix, then mixed
    with per-step Gaussian noise whose log-variance is also decoded from the GRU.

    Point forecast = observation(state).
    Interval forecast = point ± z * sigma.
    """

    STATE_DIM = 4  # level, trend, seasonal_cos, seasonal_sin

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
        self.raw_level_decay = nn.Parameter(torch.zeros(output_dim))
        self.raw_trend_decay = nn.Parameter(torch.zeros(output_dim))
        self.raw_gamma = nn.Parameter(torch.zeros(output_dim))
        self.omega = nn.Parameter(torch.zeros(output_dim))

        # Per-step log noise variance decoded from encoder
        self.log_sigma_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.eye_(self.C)
        nn.init.zeros_(self.log_sigma_proj.bias)

    def _level_decay(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_level_decay) * 0.15 + 0.85

    def _trend_decay(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_trend_decay) * 0.25 + 0.70

    def _gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_gamma) * 0.20 + 0.80

    def _transition_step(self, s: torch.Tensor) -> torch.Tensor:
        """Apply one step of the block-diagonal transition matrix."""
        a_l = self._level_decay()
        a_t = self._trend_decay()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            forecast: (batch, pred_len, output_dim)
        """
        B = x.size(0)
        enc_out, _ = self.encoder(x)
        final_h = enc_out[:, -1, :]

        # Initialize structured state from encoder output
        s = self.state_proj(final_h)                          # (B, output_dim * STATE_DIM)
        s = s.view(B, self.output_dim, self.STATE_DIM)        # (B, D, 4)

        preds = []
        for _ in range(self.pred_len):
            s = self._transition_step(s)
            # Observation: y = C @ s  (batch-wise)
            y = torch.einsum("ds,bds->bd", self.C, s)         # (B, output_dim)
            preds.append(y)

        return torch.stack(preds, dim=1)                      # (B, pred_len, output_dim)

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
        from scipy.stats import norm
        z = torch.tensor(norm.ppf(1 - self.alpha / 2), dtype=x.dtype, device=x.device)

        B = x.size(0)
        enc_out, _ = self.encoder(x)
        final_h = enc_out[:, -1, :]

        s = self.state_proj(final_h).view(B, self.output_dim, self.STATE_DIM)
        log_sigma = self.log_sigma_proj(final_h)              # (B, output_dim)
        sigma = torch.exp(log_sigma).clamp(min=1e-6)          # (B, output_dim)

        preds, sigmas = [], []
        for _ in range(self.pred_len):
            s = self._transition_step(s)
            y = torch.einsum("ds,bds->bd", self.C, s)
            preds.append(y)
            sigmas.append(sigma)

        mean = torch.stack(preds, dim=1)                      # (B, pred_len, D)
        sigma_out = torch.stack(sigmas, dim=1)                # (B, pred_len, D)

        return mean, mean - z * sigma_out, mean + z * sigma_out

    def get_contributions(self, state: Optional[torch.Tensor] = None) -> dict:
        """Return current state component names. No gradient-based attribution."""
        return {"components": ["level", "trend", "seasonal_cos", "seasonal_sin"]}
