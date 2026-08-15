"""
Anchored state dynamics for the hybrid correction branch.

The legacy encoder learns every transition parameter freely, including the
seasonal frequency omega. That is flexible but weakly identified: nothing ties
omega to the dataset's actual sampling period, so the "seasonal" coordinates
are only nominally seasonal.

The anchored variant constrains the transition to a classical local-linear-trend
plus damped-seasonal structure:

    level_t  = a_L * level_{t-1} + a_T * trend_{t-1} + innovation
    trend_t  = a_T * trend_{t-1}                     + innovation
    seasonal = damped rotation at a FIXED dataset frequency
    residual = fast-decaying                         + innovation

Two differences from legacy matter:

* The level/trend block is *coupled* -- trend feeds level, which is what makes
  it a local linear trend rather than two independent decays.
* omega is fixed from the dataset's seasonal period, not learned, so the
  rotation coordinates correspond to a known cycle length.

This is a stronger structural prior, not a more expressive model. It is a
prespecified variant to be selected on validation, not a claim of physical
identification: see the naming rule in RUNBOOK.md ("level-like", not "level",
unless synthetic recovery checks pass).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cissn.constants import STRUCTURED_STATE_DIM
from cissn.models.encoder import DisentangledStateEncoder


class AnchoredStateEncoder(DisentangledStateEncoder):
    """DisentangledStateEncoder with a coupled level/trend block and fixed omega.

    Reuses the parent's projection, innovation, and correction MLP so the only
    difference between variants is the transition matrix.
    """

    def __init__(
        self,
        input_dim: int,
        seasonal_period: int,
        state_dim: int = STRUCTURED_STATE_DIM,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        if seasonal_period <= 1:
            raise ValueError(f"seasonal_period must exceed 1; got {seasonal_period}.")
        super().__init__(
            input_dim=input_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            seasonal_period=seasonal_period,
        )

        # omega is fixed, not learned: one full rotation per seasonal period,
        # never updated. The parent's seasonal_period init already sets the
        # starting value; freeze it here so it never receives a gradient.
        self.omega.requires_grad_(False)

    def _structured_dynamics(self):
        """Coupled local-linear-trend + damped seasonal rotation + fast residual."""
        a_l = self._level_scale()
        a_t = self._trend_scale()
        g = self._gamma()
        w = self.omega
        c, sn = torch.cos(w), torch.sin(w)
        return a_l, a_t, g * c, -g * sn, g * sn, g * c, self._residual_scale()

    @staticmethod
    def _dynamics_matrix(dynamics) -> torch.Tensor:
        """Assemble A with the level<-trend coupling that defines a local linear trend.

        Differs from the parent only in the (0, 1) entry: level integrates trend.
        """
        a_l, a_t, rot00, rot01, rot10, rot11, a_r = (d.reshape(()) for d in dynamics)
        zero = torch.zeros_like(a_l)
        return torch.stack([
            # level_t = a_L * level + a_T * trend   <- the coupling term
            torch.stack([a_l, a_t, zero, zero, zero]),
            torch.stack([zero, a_t, zero, zero, zero]),
            torch.stack([zero, zero, rot00, rot10, zero]),
            torch.stack([zero, zero, rot01, rot11, zero]),
            torch.stack([zero, zero, zero, zero, a_r]),
        ])

    def _residual_scale(self) -> torch.Tensor:
        """alpha_R in [0.00, 0.20] -- faster decay than the legacy [0, 0.40].

        The residual coordinate should absorb only short-lived deviations; a
        slower decay lets it shadow the level coordinate and blur the
        decomposition.
        """
        return torch.sigmoid(self.raw_alpha_R) * 0.20
