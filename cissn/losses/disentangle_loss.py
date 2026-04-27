import torch
import torch.nn as nn


class DisentanglementLoss(nn.Module):
    """
    Unsupervised loss to encourage disentangled state representations.

    Components:
    1. Covariance regularization (independence)
    2. Temporal consistency (slow vs fast dynamics)

    Expects a 5-dimensional structured state (level, trend, seasonal pair, residual).
    """

    EXPECTED_STATE_DIM = 5

    def __init__(self, lambda_cov: float = 1.0, lambda_temporal: float = 0.5):
        super().__init__()
        self.lambda_cov = lambda_cov
        self.lambda_temporal = lambda_temporal
        self.register_buffer(
            "off_diag_mask",
            torch.ones(self.EXPECTED_STATE_DIM, self.EXPECTED_STATE_DIM)
            - torch.eye(self.EXPECTED_STATE_DIM),
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

        if state_dim == self.EXPECTED_STATE_DIM:
            off_diag_mask = self.off_diag_mask
        else:
            off_diag_mask = (
                torch.ones(state_dim, state_dim, device=states.device, dtype=states.dtype)
                - torch.eye(state_dim, device=states.device, dtype=states.dtype)
            )

        return torch.norm(cov * off_diag_mask, p="fro") ** 2

    def temporal_consistency_loss(self, states: torch.Tensor, dynamics: tuple = None) -> torch.Tensor:
        """
        Encourage appropriate temporal dynamics per dimension.

        If dynamics (from DisentangledStateEncoder) is provided, penalizes
        deviation from expected structural transitions. Otherwise, uses
        simple finite differences.
        """
        if states.shape[1] < 2:
            return torch.tensor(0.0, device=states.device)

        if dynamics is not None:
            a_l, a_t, rot00, rot01, rot10, rot11, a_r = dynamics
            s_prev = states[:, :-1, :]
            s_curr = states[:, 1:, :]

            level_loss = torch.mean((s_curr[:, :, 0] - s_prev[:, :, 0] * a_l) ** 2)
            trend_loss = torch.mean((s_curr[:, :, 1] - s_prev[:, :, 1] * a_t) ** 2)
            seasonal_loss = torch.mean(
                (s_curr[:, :, 2] - (s_prev[:, :, 2] * rot00 + s_prev[:, :, 3] * rot10)) ** 2
                + (s_curr[:, :, 3] - (s_prev[:, :, 2] * rot01 + s_prev[:, :, 3] * rot11)) ** 2
            ) * 0.1
            residual_loss = torch.mean(states[:, :, 4] ** 2)
            return level_loss + trend_loss + seasonal_loss + residual_loss

        # Fallback: finite differences
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

        residual_loss = torch.mean(states[:, :, 4] ** 2)
        return level_loss + trend_loss + seasonal_loss + residual_loss

    def forward(self, states: torch.Tensor, dynamics: tuple = None) -> torch.Tensor:
        """
        Compute total disentanglement loss.

        Args:
            states: (batch, seq_len, state_dim) with state_dim == 5
            dynamics: (tuple, optional) structural dynamics scales from the encoder
        """
        if states.shape[-1] != self.EXPECTED_STATE_DIM:
            raise ValueError(
                f"DisentanglementLoss expects state_dim={self.EXPECTED_STATE_DIM}; got {states.shape[-1]}."
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
        corr = torch.corrcoef(flat.T)
        eye = torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
        off_diag_abs = (corr * (1 - eye)).abs()
        return {
            "mean_abs_off_diag_corr": float(off_diag_abs.mean().item()),
            "per_dim_variance": [float(v) for v in flat.var(dim=0).detach().cpu()],
        }
