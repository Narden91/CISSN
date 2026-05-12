import torch
import torch.nn as nn

from cissn.constants import STRUCTURED_STATE_DIM


class DisentanglementLoss(nn.Module):
    """
    Unsupervised loss to encourage disentangled state representations.

    Components:
    1. Covariance regularization (independence)
    2. Temporal behavior regularization (smooth level/trend, stable seasonal radius,
       low-autocorrelation residual)

    Expects a 5-dimensional structured state (level, trend, seasonal pair, residual).
    """

    STRUCTURED_STATE_DIM = STRUCTURED_STATE_DIM

    def __init__(self, lambda_cov: float = 1.0, lambda_temporal: float = 0.5):
        super().__init__()
        self.lambda_cov = lambda_cov
        self.lambda_temporal = lambda_temporal
        self.register_buffer(
            "off_diag_mask",
            torch.ones(self.STRUCTURED_STATE_DIM, self.STRUCTURED_STATE_DIM)
            - torch.eye(self.STRUCTURED_STATE_DIM),
        )

    def covariance_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Penalize off-diagonal covariance (encourage independence).

        Args:
            states: (batch, seq_len, state_dim)
        """
        batch_size, seq_len, state_dim = states.shape

        centered = states - states.mean(dim=1, keepdim=True)
        centered = centered.reshape(-1, state_dim)  # (batch*seq_len, state_dim)

        n = centered.size(0)
        if n < 2:
            return torch.tensor(0.0, device=states.device)

        cov = torch.mm(centered.t(), centered) / (n - 1)
        off_diag_mask = self.off_diag_mask
        return torch.norm(cov * off_diag_mask, p="fro") ** 2

    def temporal_consistency_loss(self, states: torch.Tensor, dynamics: tuple = None) -> torch.Tensor:
        """
        Encourage appropriate temporal behavior per component.

        The encoder transition includes both input innovation and a bounded
        correction term, so this loss deliberately avoids penalizing deviation
        from A @ s_prev directly. That direct penalty conflicts with legitimate
        input-driven state changes.
        """
        if states.shape[1] < 2:
            return torch.tensor(0.0, device=states.device)

        diffs = states[:, 1:, :] - states[:, :-1, :]
        level_loss = torch.mean(diffs[:, :, 0] ** 2)

        if states.shape[1] > 2:
            diff2_trend = diffs[:, 1:, 1] - diffs[:, :-1, 1]
            trend_loss = torch.mean(diff2_trend ** 2)
        else:
            trend_loss = torch.mean(diffs[:, :, 1] ** 2)

        seasonal_mag = torch.norm(states[:, :, 2:4], dim=-1)
        mag_diff = seasonal_mag[:, 1:] - seasonal_mag[:, :-1]
        seasonal_loss = torch.mean(mag_diff ** 2) * 0.1

        residual = states[:, :, 4]
        residual = residual - residual.mean(dim=1, keepdim=True)
        if residual.shape[1] > 1:
            residual_num = torch.mean(residual[:, 1:] * residual[:, :-1], dim=1)
            residual_den = torch.mean(residual[:, 1:] ** 2 + residual[:, :-1] ** 2, dim=1).clamp_min(1e-8)
            residual_loss = torch.mean((2.0 * residual_num / residual_den) ** 2)
        else:
            residual_loss = torch.tensor(0.0, device=states.device)
        return level_loss + trend_loss + seasonal_loss + residual_loss

    def forward(self, states: torch.Tensor, dynamics: tuple = None) -> torch.Tensor:
        """
        Compute total disentanglement loss.

        Args:
            states: (batch, seq_len, state_dim) with state_dim == 5
            dynamics: (tuple, optional) structural dynamics scales from the encoder
        """
        if states.shape[-1] != self.STRUCTURED_STATE_DIM:
            raise ValueError(
                f"DisentanglementLoss expects state_dim={self.STRUCTURED_STATE_DIM}; got {states.shape[-1]}."
            )
        return self.lambda_cov * self.covariance_loss(states) + self.lambda_temporal * self.temporal_consistency_loss(
            states, dynamics
        )

    @staticmethod
    def get_metrics(states: torch.Tensor) -> dict:
        """Compute disentanglement quality diagnostics from state sequences.

        Args:
            states: (batch, seq_len, state_dim)

        Returns:
            dict with mean_abs_off_diag_corr and per_dim_variance.
        """
        flat = states.reshape(-1, states.shape[-1])
        corr = torch.nan_to_num(torch.corrcoef(flat.T), nan=0.0, posinf=0.0, neginf=0.0)
        eye = torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
        off_diag_abs = (corr * (1 - eye)).abs()
        return {
            "mean_abs_off_diag_corr": float(off_diag_abs.mean().item()),
            "per_dim_variance": [float(v) for v in flat.var(dim=0).detach().cpu()],
        }
