"""
Reversible instance normalisation (RevIN).

Reference: Kim et al., "Reversible Instance Normalization for Accurate
Time-Series Forecasting against Distribution Shift", ICLR 2022.
https://openreview.net/forum?id=cGDAkQo1C0p

Each input window is standardised per instance and per feature before the model
sees it, and the statistics are re-applied to the forecast:

    x_norm = (x - mean) / std          mean, std from THIS window only
    y_hat  = model(x_norm) * std + mean

The point is to remove the level-tracking burden. Without it, a model forecasting
a non-stationary series must spend capacity representing where the series
currently sits, and under MSE the safe way to do that is to shrink toward the
training mean -- which produces an under-dispersed, near-constant forecast that
still lowers training loss. Normalising per window makes the level an explicit,
non-learned quantity, so the model only has to predict shape.

Statistics come exclusively from the input window, never from the target, so
this leaks no future information.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Per-instance, per-feature normalisation with a matching denormalisation.

    Args:
        num_features: Channel count of the input window.
        eps: Floor on the standard deviation, guarding constant channels.
        affine: Learn a per-feature scale and shift applied in normalised space.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if num_features <= 0:
            raise ValueError(f"num_features must be positive; got {num_features}.")
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self._mean: torch.Tensor | None = None
        self._stdev: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._store_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"mode must be 'norm' or 'denorm'; got {mode!r}.")

    def _store_statistics(self, x: torch.Tensor) -> None:
        """Record per-instance statistics from the input window only."""
        if x.ndim != 3:
            raise ValueError(f"expected (batch, seq_len, features); got {tuple(x.shape)}.")
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"expected {self.num_features} features; got {x.shape[-1]}."
            )
        # detach(): these are treated as constants of the transform, so gradients
        # flow through the normalised forecast rather than through the statistics.
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._stdev = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._mean) / self._stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._stdev is None:
            raise RuntimeError("call forward(x, 'norm') before denormalising.")
        if x.shape[-1] != self.num_features:
            # Broadcasting would silently rescale with the WRONG channel's
            # statistics -- under --features MS a single-column forecast would
            # pick up feature 0's mean/std instead of the target's. Refuse it;
            # the caller must select statistics with select_channels() instead.
            raise ValueError(
                f"denorm expects {self.num_features} channels to match the stored "
                f"statistics; got {x.shape[-1]}. Use select_channels() to restrict "
                "RevIN to the forecast's target channels."
            )
        if self.affine:
            # Reverse the affine map before restoring scale; eps guards a weight
            # that has been driven to zero during training.
            x = (x - self.affine_bias) / (self.affine_weight + self.eps**2)
        # Statistics are (batch, 1, features) and broadcast over any horizon,
        # so the forecast length need not match the input length.
        return x * self._stdev + self._mean

    def select_channels(self, index: int) -> "RevIN":
        """Restrict stored statistics to a single channel, for MS forecasting.

        ``--features MS`` predicts one target column from many inputs, so the
        forecast has one channel while the statistics have ``num_features``.
        This returns a lightweight view carrying only that channel's statistics
        so denormalisation uses the target's own mean and scale.
        """
        if self._mean is None or self._stdev is None:
            raise RuntimeError("call forward(x, 'norm') before selecting channels.")
        # Normalise the index first: a bare index+1 slice yields an empty tensor
        # for index=-1, which would silently produce a zero-width forecast.
        if not -self.num_features <= index < self.num_features:
            raise IndexError(
                f"channel index {index} is out of range for {self.num_features} features."
            )
        index %= self.num_features
        channel = slice(index, index + 1)

        view = RevIN.__new__(RevIN)
        nn.Module.__init__(view)
        view.num_features = 1
        view.eps = self.eps
        view.affine = self.affine
        if self.affine:
            view.affine_weight = self.affine_weight[channel]
            view.affine_bias = self.affine_bias[channel]
        view._mean = self._mean[..., channel]
        view._stdev = self._stdev[..., channel]
        return view
