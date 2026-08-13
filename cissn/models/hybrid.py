"""
Balanced hybrid forecaster: DLinear base plus an additive five-state correction.

    base       = DLinear(raw history)
    correction = LinearCorrectionHead(StateEncoder(history))
    total      = base + correction

The motivation is an information bottleneck. The legacy CISSN routes a
(seq_len, input_dim) history through a 5-dimensional state before forecasting,
so the forecast map has rank <= 5 and most of the raw history is unrecoverable.
DLinear reads the raw history directly. Keeping DLinear as the base and giving
the state only an additive *correction* preserves the structured five-state
interpretation without paying the bottleneck on the whole signal.

Two properties make this safe to publish, and both are enforced by tests:

1. The correction head is zero-initialised, so at correction-stage epoch 0 the
   hybrid is *exactly* the frozen DLinear -- bit-for-bit, not approximately.
2. The base is frozen during correction training, so the hybrid can only
   deviate from DLinear through the correction path, and validation selection
   can always fall back to the epoch-0 checkpoint.

Together these bound the downside: the hybrid cannot do worse than its own base
except by a training choice that checkpoint selection is free to reject.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from cissn.baselines.dlinear import DLinear
from cissn.constants import STRUCTURED_STATE_DIM
from cissn.models.encoder import DisentangledStateEncoder


class LinearCorrectionHead(nn.Module):
    """Zero-initialised linear map from structured state to a forecast correction.

    Deliberately linear: the correction must stay attributable to individual
    state coordinates, so every output is an exact sum of per-coordinate terms
    plus a bias. A non-linear head would forfeit that decomposition.
    """

    def __init__(self, state_dim: int, horizon: int, output_dim: int):
        super().__init__()
        if state_dim != STRUCTURED_STATE_DIM:
            raise ValueError(
                f"LinearCorrectionHead requires state_dim={STRUCTURED_STATE_DIM}; got {state_dim}."
            )
        self.state_dim = state_dim
        self.horizon = horizon
        self.output_dim = output_dim

        # Zero init is the load-bearing choice: it makes epoch 0 exactly DLinear.
        self.weight = nn.Parameter(torch.zeros(state_dim, horizon, output_dim))
        self.bias = nn.Parameter(torch.zeros(horizon, output_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (batch, state_dim) -> (batch, horizon, output_dim)"""
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(
                f"expected state shape (batch, {self.state_dim}); got {tuple(state.shape)}."
            )
        return torch.einsum("bs,sho->bho", state, self.weight) + self.bias

    def contributions(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-coordinate correction terms, each (batch, horizon, output_dim).

        The returned terms sum exactly to ``forward(state)``.
        """
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(
                f"expected state shape (batch, {self.state_dim}); got {tuple(state.shape)}."
            )
        # (batch, state_dim, horizon, output_dim) before reduction over state.
        per_coord = state[:, :, None, None] * self.weight[None, :, :, :]
        return {
            "state_level": per_coord[:, 0],
            "state_trend": per_coord[:, 1],
            # Coordinates 2 and 3 are the cos/sin pair of one rotation block and
            # are only jointly meaningful, so they are reported as one term.
            "state_seasonal": per_coord[:, 2] + per_coord[:, 3],
            "state_residual": per_coord[:, 4],
            "bias": self.bias.expand(state.shape[0], -1, -1),
        }


class HybridCISSN(nn.Module):
    """DLinear base with an additive, frozen-base five-state correction."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        output_dim: int = 1,
        state_dim: int = STRUCTURED_STATE_DIM,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        kernel_size: int = 25,
        state_revin: bool = False,
        state_dynamics: str = "legacy",
        seasonal_period: Optional[int] = None,
    ):
        super().__init__()
        if state_dynamics not in {"legacy", "anchored"}:
            raise ValueError(
                f"state_dynamics must be 'legacy' or 'anchored'; got {state_dynamics!r}."
            )
        if state_dynamics == "anchored" and not seasonal_period:
            raise ValueError("anchored state dynamics require a positive seasonal_period.")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.state_revin = state_revin
        self.state_dynamics = state_dynamics

        self.base = DLinear(
            input_dim=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            output_dim=output_dim,
            kernel_size=kernel_size,
        )
        if state_dynamics == "anchored":
            # Imported here: cissn.models.anchored imports the encoder module,
            # so a module-level import would be circular.
            from cissn.models.anchored import AnchoredStateEncoder

            self.encoder = AnchoredStateEncoder(
                input_dim=input_dim,
                seasonal_period=seasonal_period,
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            self.encoder = DisentangledStateEncoder(
                input_dim=input_dim,
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        self.correction = LinearCorrectionHead(
            state_dim=state_dim, horizon=pred_len, output_dim=output_dim
        )

    def freeze_base(self) -> "HybridCISSN":
        """Freeze the DLinear base for correction-stage training."""
        self.base.requires_grad_(False)
        self.base.eval()
        return self

    def correction_parameters(self):
        """Parameters trained during the correction stage (base excluded)."""
        return list(self.encoder.parameters()) + list(self.correction.parameters())

    def train(self, mode: bool = True) -> "HybridCISSN":
        """Keep a frozen base in eval mode even when the hybrid is set to train.

        Without this, ``model.train()`` would re-enable any base-side train-mode
        behaviour after freezing, so the "base never changes" guarantee would
        hold for parameters but not for the forward pass.
        """
        super().train(mode)
        if not any(p.requires_grad for p in self.base.parameters()):
            self.base.eval()
        return self

    def _state_from_history(self, x: torch.Tensor) -> torch.Tensor:
        """Encode history to a final structured state, optionally instance-normed.

        With ``state_revin`` the state branch sees a per-instance, per-feature
        standardised window and the resulting correction is rescaled back by the
        window scale. Zero-init exactness is unaffected (zero times any scale is
        still zero), but the gradient scale at epoch 0 differs from the
        non-RevIN variants, so the two are not comparable at equal learning rate.
        """
        if not self.state_revin:
            return self.encoder(x)

        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        return self.encoder((x - mean) / std)

    def _correction_scale(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Per-instance feature scale used to rescale the RevIN correction."""
        if not self.state_revin:
            return None
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        if std.shape[-1] != self.output_dim:
            std = std[..., : self.output_dim]
        return std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) -> (batch, pred_len, output_dim)"""
        base = self.base(x)
        correction = self.correction(self._state_from_history(x))
        scale = self._correction_scale(x)
        if scale is not None:
            correction = correction * scale
        return base + correction

    def forward_components(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full additive decomposition of the forecast.

        ``base + total_correction`` equals ``total`` exactly, and the individual
        state terms sum to ``total_correction`` exactly (up to the rescaling
        applied under ``state_revin``, which is applied uniformly to each term).
        """
        base = self.base(x)
        state = self._state_from_history(x)
        terms = self.correction.contributions(state)
        scale = self._correction_scale(x)
        if scale is not None:
            terms = {name: term * scale for name, term in terms.items()}

        total_correction = (
            terms["state_level"]
            + terms["state_trend"]
            + terms["state_seasonal"]
            + terms["state_residual"]
            + terms["bias"]
        )
        return {
            "base": base,
            **terms,
            "state": state,
            "total_correction": total_correction,
            "total": base + total_correction,
        }

    def get_contributions(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """State-only contributions, for compatibility with the legacy API."""
        return self.correction.contributions(state)
