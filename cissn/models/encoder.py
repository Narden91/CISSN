"""
Disentangled state encoder with structured transition (level, trend, rotation, residual).
"""
from __future__ import annotations

import math
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cissn.constants import STRUCTURED_STATE_DIM
from cissn.models._dynamics import StructuredDecayMixin


class DisentangledStateEncoder(StructuredDecayMixin):
    """
    Maps sequences to a 5-dimensional structural latent state with constrained dynamics.
    """

    STRUCTURED_STATE_DIM = STRUCTURED_STATE_DIM

    def __init__(
        self,
        input_dim: int,
        state_dim: int = STRUCTURED_STATE_DIM,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        seasonal_period: Optional[int] = None,
    ):
        """
        Args:
            seasonal_period: If given, initialises the seasonal rotation
                frequency omega at 2*pi/seasonal_period instead of 0, so the
                seasonal block starts rotating at the dataset's actual cycle
                length rather than as a non-rotating decay indistinguishable
                from the residual coordinate. omega remains learnable
                (contrast AnchoredStateEncoder, which also freezes it). None
                preserves the historical zero-init behaviour.
        """
        super().__init__()
        if state_dim != self.STRUCTURED_STATE_DIM:
            raise ValueError(
                f"DisentangledStateEncoder requires state_dim={self.STRUCTURED_STATE_DIM}; got {state_dim}."
            )
        if seasonal_period is not None and seasonal_period <= 1:
            raise ValueError(f"seasonal_period must exceed 1; got {seasonal_period}.")
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.seasonal_period = seasonal_period

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.innovation = nn.Linear(hidden_dim, state_dim)
        # Spectral norm bounds each linear layer to ||W||_2 <= 1, but GELU has
        # Lipschitz constant L_G ~= 1.13, so the composite MLP Lipschitz constant
        # is L_G (not 1). The per-step Jacobian bound is 1 + L_G*beta, not 1+beta.
        self.correction_mlp = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(state_dim + hidden_dim, hidden_dim)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, state_dim)),
        )
        self.raw_correction_scale = nn.Parameter(torch.tensor(-4.6))  # softplus⁻¹(0.01)
        omega_init = 2.0 * math.pi / seasonal_period if seasonal_period is not None else 0.0
        self._register_decay_params(n_dims=1, include_residual=True, omega_init=omega_init)

        if sys.platform != 'win32' and hasattr(torch, 'compile'):
            self._run_sequence = torch.compile(self._run_sequence)

        # CUDA-graph cache for inference. The recurrence is launch-bound, so
        # replaying a captured graph removes nearly all per-step launch cost.
        # Opt-in via enable_cuda_graph(): capture requires static shapes.
        self._cuda_graph_enabled = False
        self._cuda_graphs: dict = {}

    def enable_cuda_graph(self, enabled: bool = True) -> "DisentangledStateEncoder":
        """Enable CUDA-graph replay for no-grad inference forwards.

        Only affects eval-mode forwards under ``torch.no_grad`` on CUDA with a
        fixed input shape; every other path runs eagerly. Call again after
        loading new weights, which invalidates captured graphs.
        """
        self._cuda_graph_enabled = enabled
        self._cuda_graphs.clear()
        return self

    def _graphed_forward(self, x: torch.Tensor, return_all_states: bool):
        """Replay (or capture) a CUDA graph for this input shape."""
        key = (tuple(x.shape), x.dtype, return_all_states)
        entry = self._cuda_graphs.get(key)

        if entry is None:
            # Warm up on a side stream before capture, as required by the CUDA
            # graph API; capture would otherwise record allocator/cuBLAS setup.
            static_in = x.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._forward_eager(static_in, return_all_states)
            torch.cuda.current_stream().wait_stream(side)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = self._forward_eager(static_in, return_all_states)
            entry = (graph, static_in, static_out)
            self._cuda_graphs[key] = entry

        graph, static_in, static_out = entry
        static_in.copy_(x)
        graph.replay()
        # The graph always writes into the same buffer, so hand back a copy.
        return static_out.clone()

    def _forward_eager(self, x: torch.Tensor, return_all_states: bool):
        projected = self.input_proj(x)
        dynamics = self._structured_dynamics()
        return self._run_sequence(projected, dynamics, return_all_states)

    def _correction_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_correction_scale)

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
        return s @ self._dynamics_matrix(dynamics).T

    @staticmethod
    def _dynamics_matrix(dynamics) -> torch.Tensor:
        """Assemble the block-diagonal A as a single (5, 5) matrix.

        A single matmul replaces the per-slice stack, cutting ~10 kernel launches
        per timestep to one. The state dimension is 5, so building A costs far
        less than the launches it saves across a 96-step sequence.
        """
        # Decay params carry a trailing singleton dim; reshape to scalars so the
        # assembled matrix is exactly (state_dim, state_dim).
        a_l, a_t, rot00, rot01, rot10, rot11, a_r = (d.reshape(()) for d in dynamics)
        zero = torch.zeros_like(a_l)
        return torch.stack([
            torch.stack([a_l, zero, zero, zero, zero]),
            torch.stack([zero, a_t, zero, zero, zero]),
            torch.stack([zero, zero, rot00, rot10, zero]),
            torch.stack([zero, zero, rot01, rot11, zero]),
            torch.stack([zero, zero, zero, zero, a_r]),
        ])

    def _step_from_hidden(self, h_t: torch.Tensor, s_prev: torch.Tensor, dynamics) -> torch.Tensor:
        b_x = self.innovation(h_t)
        s_linear = self.apply_structured_A(s_prev, dynamics=dynamics) + b_x
        corr_in = torch.cat([s_linear, h_t], dim=-1)
        correction = self._correction_scale() * torch.tanh(self.correction_mlp(corr_in))
        return s_linear + correction

    @staticmethod
    def _resolved_weight(module: nn.Module) -> torch.Tensor:
        """Return a linear layer's effective weight, applying spectral norm once.

        ``spectral_norm`` installs a forward pre-hook that recomputes ``weight``
        from ``weight_orig``; reading ``module.weight`` directly would bypass it
        and use a stale tensor. Firing the hooks explicitly resolves the weight
        exactly once so it can be reused across the whole sequence.
        """
        for hook in module._forward_pre_hooks.values():
            hook(module, None)
        return module.weight

    def step(self, x_t: torch.Tensor, s_prev: torch.Tensor) -> torch.Tensor:
        h_t = self.input_proj(x_t)
        return self._step_from_hidden(h_t, s_prev, dynamics=self._structured_dynamics())

    def _run_sequence(self, projected: torch.Tensor, dynamics: tuple, return_all_states: bool) -> torch.Tensor:
        batch, seq_len, _ = projected.shape
        s = torch.zeros(batch, self.state_dim, device=projected.device, dtype=projected.dtype)

        # Hoist everything that does not depend on s out of the loop. The inner
        # loop is launch-bound (per-step kernels do ~1us of work), so each op
        # removed from it saves seq_len kernel launches.
        #
        # 1. innovation() is applied to every timestep independently: run it once
        #    as a single batched matmul instead of seq_len small ones.
        b_x_all = self.innovation(projected)                    # (B, T, state_dim)
        # 2. spectral_norm recomputes the normalised weight in a forward pre-hook,
        #    so calling correction_mlp per step re-runs the power iteration every
        #    timestep. That is both wasteful and wrong: it re-normalises
        #    mid-sequence, breaking the fixed ||W||_2 <= 1 assumption behind the
        #    1 + L_G*beta Lipschitz bound. Resolve the weights exactly once here.
        w0, b0 = self._resolved_weight(self.correction_mlp[0]), self.correction_mlp[0].bias
        w1, b1 = self._resolved_weight(self.correction_mlp[2]), self.correction_mlp[2].bias
        a_mat = self._dynamics_matrix(dynamics).T
        scale = self._correction_scale()

        outs = [] if return_all_states else None
        for t in range(seq_len):
            h_t = projected[:, t, :]
            s_linear = s @ a_mat + b_x_all[:, t, :]
            corr_in = torch.cat([s_linear, h_t], dim=-1)
            hidden = F.gelu(F.linear(corr_in, w0, b0))
            s = s_linear + scale * torch.tanh(F.linear(hidden, w1, b1))
            if return_all_states:
                outs.append(s)

        if return_all_states:
            return torch.stack(outs, dim=1)
        return s

    def forward(self, x: torch.Tensor, return_all_states: bool = False):
        """
        Args:
            x: (batch, seq_len, input_dim)
            return_all_states: if True, return (batch, seq_len, state_dim)

        Returns:
            Final state (batch, state_dim) or all states (batch, seq_len, state_dim)
        """
        # Graph replay is only valid when nothing can change between replays:
        # eval mode (no dropout randomness, no spectral-norm power iteration),
        # no autograd, and a CUDA input. Anything else runs eagerly.
        if (
            self._cuda_graph_enabled
            and not self.training
            and not torch.is_grad_enabled()
            and x.is_cuda
        ):
            return self._graphed_forward(x, return_all_states)
        return self._forward_eager(x, return_all_states)
