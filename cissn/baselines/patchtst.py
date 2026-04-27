"""
PatchTST baseline: patch-based Transformer for long-term time-series forecasting.

Divides the input sequence into non-overlapping (or strided) patches, embeds each
patch with a linear projection, then applies a standard Transformer encoder.
Prediction is made via a flat linear head over all patch embeddings.

Channel-independent by default (one model shared across all variates).

Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
    with Transformers", ICLR 2023.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Projects non-overlapping or strided patches to d_model."""

    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) — single variate

        Returns:
            (batch, n_patches, d_model)
        """
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return self.dropout(self.norm(self.proj(x)))


class PatchTST(nn.Module):
    """
    Channel-independent PatchTST.

    Each variate is processed independently through the same patch embedding
    and Transformer encoder. Outputs are projected back to pred_len per variate.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Number of patches
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.d_model = d_model

        self.patch_embed = PatchEmbedding(patch_len, stride, d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.head = nn.Linear(self.n_patches * d_model, pred_len)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            forecast: (batch, pred_len, input_dim)
        """
        B, L, D = x.shape

        # Process each variate independently: reshape to (B*D, L)
        x = x.permute(0, 2, 1).reshape(B * D, L)

        patches = self.patch_embed(x)                         # (B*D, n_patches, d_model)
        enc_out = self.transformer(patches)                   # (B*D, n_patches, d_model)
        flat = enc_out.reshape(B * D, -1)                     # (B*D, n_patches * d_model)
        out = self.head(flat)                                 # (B*D, pred_len)

        return out.reshape(B, D, self.pred_len).permute(0, 2, 1)  # (B, pred_len, D)

    def get_contributions(self, state=None) -> dict:
        """PatchTST has no interpretable state decomposition; returns empty dict."""
        return {}
