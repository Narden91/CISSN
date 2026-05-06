"""
Shared block-diagonal structured dynamics for CISSN models.

Both DisentangledStateEncoder and DeepState implement the same sigmoid-gated
decay parameters (level, trend, seasonal rotation, residual).  This mixin
centralises the parameter ranges so a change propagates to both models.

Parameter ranges (all constrained via sigmoid):
    alpha_L  ∈ [0.85, 1.00]   — level (near unit-root)
    alpha_T  ∈ [0.70, 0.95]   — trend (persistent)
    gamma    ∈ [0.80, 1.00]   — seasonal rotation amplitude
    alpha_R  ∈ [0.00, 0.40]   — residual (fast-decaying)   [encoder only]
    omega    — learnable seasonal frequency                 [both]
"""
import torch
import torch.nn as nn


class StructuredDecayMixin(nn.Module):
    """Mixin that registers sigmoid-constrained decay parameters and exposes
    helper properties returning their effective (constrained) values.

    Subclasses must call ``super().__init__()`` before using these methods,
    and may optionally pass ``scalar=True`` to create 1-d parameters (encoder)
    or ``scalar=False`` with a given ``n_dims`` to create per-output-dim
    parameters (DeepState with output_dim > 1).
    """

    def _register_decay_params(self, n_dims: int = 1, include_residual: bool = True) -> None:
        """Register raw (unconstrained) decay parameters.

        Args:
            n_dims: Parameter vector length.  Use 1 for a shared scalar; use
                ``output_dim`` for per-variate parameters (DeepState).
            include_residual: Whether to register raw_alpha_R.  The DeepState
                baseline has no residual component, so it passes False.
        """
        self.raw_alpha_L = nn.Parameter(torch.zeros(n_dims))
        self.raw_alpha_T = nn.Parameter(torch.zeros(n_dims))
        self.raw_gamma   = nn.Parameter(torch.zeros(n_dims))
        self.omega       = nn.Parameter(torch.zeros(n_dims))
        if include_residual:
            self.raw_alpha_R = nn.Parameter(torch.zeros(n_dims))

    # ------------------------------------------------------------------
    # Constrained accessors
    # ------------------------------------------------------------------

    def _level_scale(self) -> torch.Tensor:
        """alpha_L ∈ [0.85, 1.00]"""
        return torch.sigmoid(self.raw_alpha_L) * 0.15 + 0.85

    def _trend_scale(self) -> torch.Tensor:
        """alpha_T ∈ [0.70, 0.95]"""
        return torch.sigmoid(self.raw_alpha_T) * 0.25 + 0.70

    def _gamma(self) -> torch.Tensor:
        """gamma ∈ [0.80, 1.00]"""
        return torch.sigmoid(self.raw_gamma) * 0.20 + 0.80

    def _residual_scale(self) -> torch.Tensor:
        """alpha_R ∈ [0.00, 0.40]"""
        return torch.sigmoid(self.raw_alpha_R) * 0.40
