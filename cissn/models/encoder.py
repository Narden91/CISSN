"""
Disentangled state encoder with structured transition (level, trend, rotation, residual).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DisentangledStateEncoder(nn.Module):
    """
    Maps sequences to a 5-dimensional structural latent state with constrained dynamics.
    """

    STRUCTURED_STATE_DIM = 5

    def __init__(
        self,
        input_dim: int,
        state_dim: int = STRUCTURED_STATE_DIM,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        if state_dim != self.STRUCTURED_STATE_DIM:
            raise ValueError(
                f"DisentangledStateEncoder requires state_dim={self.STRUCTURED_STATE_DIM}; got {state_dim}."
            )
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.innovation = nn.Linear(hidden_dim, state_dim)
        self.correction_mlp = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.correction_scale = nn.Parameter(torch.tensor(0.01))

        self.raw_alpha_L = nn.Parameter(torch.zeros(1))
        self.raw_alpha_T = nn.Parameter(torch.zeros(1))
        self.raw_gamma = nn.Parameter(torch.zeros(1))
        self.raw_alpha_R = nn.Parameter(torch.zeros(1))
        self.omega = nn.Parameter(torch.zeros(1))

    def _level_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha_L) * 0.15 + 0.85

    def _trend_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha_T) * 0.25 + 0.70

    def _gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_gamma) * 0.20 + 0.80

    def _residual_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha_R) * 0.40

    def _structured_dynamics(self):
        a_l = self._level_scale()
        a_t = self._trend_scale()
        g = self._gamma()
        w = self.omega
        c, sn = torch.cos(w), torch.sin(w)
        return a_l, a_t, g * c, -g * sn, g * sn, g * c, self._residual_scale()

    def apply_structured_A(self, s: torch.Tensor, dynamics=None) -> torch.Tensor:
        """Apply block-diagonal A: level, trend, 2D rotation (seasonal), residual."""
        if dynamics is None:
            dynamics = self._structured_dynamics()
        a_l, a_t, rot00, rot01, rot10, rot11, a_r = dynamics
        s0, s1 = s[:, 2:3], s[:, 3:4]
        s_season0 = s0 * rot00 + s1 * rot10
        s_season1 = s0 * rot01 + s1 * rot11
        s_season = torch.cat([s_season0, s_season1], dim=-1)
        s_level = s[:, 0:1] * a_l
        s_trend = s[:, 1:2] * a_t
        s_res = s[:, 4:5] * a_r
        return torch.cat([s_level, s_trend, s_season, s_res], dim=-1)

    def _step_from_hidden(self, h_t: torch.Tensor, s_prev: torch.Tensor, dynamics) -> torch.Tensor:
        b_x = self.innovation(h_t)
        s_linear = self.apply_structured_A(s_prev, dynamics=dynamics) + b_x
        corr_in = torch.cat([s_linear, h_t], dim=-1)
        correction = self.correction_scale * torch.tanh(self.correction_mlp(corr_in))
        return s_linear + correction

    def step(self, x_t: torch.Tensor, s_prev: torch.Tensor) -> torch.Tensor:
        h_t = self.input_proj(x_t)
        return self._step_from_hidden(h_t, s_prev, dynamics=self._structured_dynamics())

    def forward(self, x: torch.Tensor, return_all_states: bool = False):
        """
        Args:
            x: (batch, seq_len, input_dim)
            return_all_states: if True, return (batch, seq_len, state_dim)

        Returns:
            Final state (batch, state_dim) or all states (batch, seq_len, state_dim)
        """
        batch, seq_len, _ = x.shape
        s = torch.zeros(batch, self.state_dim, device=x.device, dtype=x.dtype)
        projected = self.input_proj(x)
        dynamics = self._structured_dynamics()
        if return_all_states:
            outs = x.new_empty(batch, seq_len, self.state_dim)
            for t in range(seq_len):
                s = self._step_from_hidden(projected[:, t, :], s, dynamics)
                outs[:, t, :] = s
            return outs
        for t in range(seq_len):
            s = self._step_from_hidden(projected[:, t, :], s, dynamics)
        return s
