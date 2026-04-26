"""
DLinear baseline: single linear layer forecasting from last input observation.

Reference: Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DLinear(nn.Module):
    """
    Decomposition-Linear: applies a moving-average decomposition followed by
    two independent linear projections for trend and residual components.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        output_dim: int = 1,
        kernel_size: int = 25,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.kernel_size = kernel_size

        self.decompose = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
                                       padding=kernel_size // 2)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_residual = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            forecast: (batch, pred_len, output_dim) — last output_dim channels
        """
        x = x[:, :, -1:]  # (B, L, 1) — use last feature for univariate
        x = x.permute(0, 2, 1)  # (B, 1, L)

        trend = self.decompose(x)
        residual = x - trend

        trend_out = self.linear_trend(trend).permute(0, 2, 1)   # (B, pred_len, 1)
        residual_out = self.linear_residual(residual).permute(0, 2, 1)
        return trend_out + residual_out

    def get_contributions(self, state=None) -> dict:
        """DLinear has no interpretable state; returns empty dict."""
        return {}
