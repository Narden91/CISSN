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
        self.refinement_scale = nn.Parameter(torch.tensor(0.1))

    def _validate_state(self, state: torch.Tensor, caller: str) -> None:
        if state.ndim != 2:
            raise ValueError(f"{caller} expects state shape (batch, state_dim); got {tuple(state.shape)}.")
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"{caller} expects trailing state dimension {self.state_dim}; got {state.shape[-1]}."
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (batch, state_dim) -> (batch, horizon, output_dim)"""
        self._validate_state(state, "forward")
        lin = torch.einsum("bs,sho->bho", state, self.lin_weight) + self.lin_bias
        ref = self.refinement_scale * self.refine(state).view(-1, self.horizon, self.output_dim)
        return lin + ref

    def get_contributions(
        self,
        state: torch.Tensor,
        horizon_idx: int = 0,
        output_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose a selected forecast coordinate into structured linear terms plus refinement.

        Args:
            state: (batch, state_dim)
            horizon_idx: Forecast horizon index to explain.
            output_idx: Output channel index to explain.
        """
        self._validate_state(state, "get_contributions")
        if self.state_dim != self.STRUCTURED_STATE_DIM:
            raise ValueError(
                f"get_contributions requires state_dim={self.STRUCTURED_STATE_DIM}; got {self.state_dim}."
            )
        if not 0 <= horizon_idx < self.horizon:
            raise IndexError(f"horizon_idx must be in [0, {self.horizon}); got {horizon_idx}.")
        if not 0 <= output_idx < self.output_dim:
            raise IndexError(f"output_idx must be in [0, {self.output_dim}); got {output_idx}.")

        w = self.lin_weight[:, horizon_idx, output_idx]
        refinement = self.refinement_scale * self.refine(state).view(-1, self.horizon, self.output_dim)[:, horizon_idx, output_idx]

        level = state[:, 0] * w[0]
        trend = state[:, 1] * w[1]
        seasonal = state[:, 2] * w[2] + state[:, 3] * w[3]
        residual = state[:, 4] * w[4]
        total_linear = level + trend + seasonal + residual
        bias = self.lin_bias[horizon_idx, output_idx].expand_as(total_linear)
        linear_prediction = total_linear + bias
        return {
            "level": level,
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "bias": bias,
            "total_linear": total_linear,
            "linear_prediction": linear_prediction,
            "refinement_contribution": refinement,
            "total_prediction": linear_prediction + refinement,
        }

    def get_refinement_ratio(self, state: torch.Tensor) -> float:
        """Fraction of total prediction magnitude coming from the refinement path.

        A ratio near 0 means the linear interpretable path dominates.
        A ratio near 1 means the non-linear refinement dominates,
        which weakens the interpretability guarantee.
        """
        self._validate_state(state, "get_refinement_ratio")
        lin = torch.einsum("bs,sho->bho", state, self.lin_weight) + self.lin_bias
        ref = self.refinement_scale * self.refine(state).view(-1, self.horizon, self.output_dim)
        lin_mag = lin.abs().mean()
        ref_mag = ref.abs().mean()
        total = lin_mag + ref_mag + 1e-8
        return (ref_mag / total).item()
