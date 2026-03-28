"""
Hybrid linear + refinement forecast head with per-component linear contributions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional


class ForecastHead(nn.Module):
    """
    Predicts multi-horizon outputs from final state: linear (interpretable) + MLP refinement.
    """

    STRUCTURED_STATE_DIM = 5

    def __init__(
        self,
        state_dim: int,
        output_dim: int = 1,
        horizon: int = 1,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(32, state_dim * 2)
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.horizon = horizon

        self.lin_weight = nn.Parameter(torch.randn(state_dim, horizon, output_dim) * 0.02)
        self.lin_bias = nn.Parameter(torch.zeros(horizon, output_dim))
        self.refine = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon * output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (batch, state_dim) -> (batch, horizon, output_dim)"""
        lin = torch.einsum("bs,sho->bho", state, self.lin_weight) + self.lin_bias
        ref = self.refine(state).view(-1, self.horizon, self.output_dim)
        return lin + ref

    def get_contributions(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Linear decomposition for the first horizon step and first output channel (interpretability).
        """
        if self.state_dim != self.STRUCTURED_STATE_DIM:
            raise ValueError(
                f"get_contributions requires state_dim={self.STRUCTURED_STATE_DIM}; got {self.state_dim}."
            )
        w = self.lin_weight
        h0, o0 = 0, 0
        level = state[:, 0] * w[0, h0, o0]
        trend = state[:, 1] * w[1, h0, o0]
        seasonal = state[:, 2] * w[2, h0, o0] + state[:, 3] * w[3, h0, o0]
        residual = state[:, 4] * w[4, h0, o0]
        total_linear = level + trend + seasonal + residual
        return {
            "level": level,
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "bias": self.lin_bias[h0, o0],
            "total_linear": total_linear,
        }
