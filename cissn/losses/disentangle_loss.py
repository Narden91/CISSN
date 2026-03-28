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

    def __init__(
        self,
        lambda_cov: float = 1.0,
        lambda_temporal: float = 0.5
    ):
        super().__init__()
        self.lambda_cov = lambda_cov
        self.lambda_temporal = lambda_temporal
        
    def covariance_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Penalize off-diagonal covariance (encourage independence).
        
        Args:
            states: (batch, seq_len, state_dim)
        """
        # Flatten batch and time
        batch_size, seq_len, state_dim = states.shape
        flat_states = states.reshape(-1, state_dim)  # (batch*seq_len, state_dim)
        
        # Center the states
        mean = flat_states.mean(dim=0, keepdim=True)
        centered = flat_states - mean
        
        # Compute covariance matrix
        n = flat_states.size(0)
        if n < 2:
            return torch.tensor(0.0, device=states.device)
            
        cov = torch.mm(centered.t(), centered) / (n - 1)
        
        # Penalize off-diagonal elements
        # Create mask for off-diagonal
        eye = torch.eye(state_dim, device=states.device)
        off_diag_cov = cov * (1 - eye)
        
        return torch.norm(off_diag_cov, p='fro') ** 2
    
    def temporal_consistency_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Encourage appropriate temporal dynamics per dimension.
        
        - Level (dim 0): slow-varying
        - Trend (dim 1): smooth (second-difference penalty when seq_len > 2)
        - Seasonal (dims 2–3): weak magnitude drift penalty
        - Residual (dim 4): small magnitude
        """
        if states.shape[1] < 2:
            return torch.tensor(0.0, device=states.device)
            
        # Diff across time
        diffs = states[:, 1:, :] - states[:, :-1, :]
        
        # Level (dim 0): slow varying -> minimize diff
        level_loss = torch.mean(diffs[:, :, 0] ** 2)
        
        # Trend (dim 1): smooth -> minimize second derivative (diff of diffs)
        if states.shape[1] > 2:
            diff2_trend = diffs[:, 1:, 1] - diffs[:, :-1, 1]
            trend_loss = torch.mean(diff2_trend ** 2)
        else:
            trend_loss = torch.mean(diffs[:, :, 1] ** 2)
            
        # Seasonal (dims 2, 3): 
        # Ideally we want them to follow the rotation dynamics, but that's hard to enforce via loss 
        # without knowing omega perfectly. The structural prior (A matrix) does most of the work.
        # We can add a weak regularization to keep magnitude consistent or just ignore for now.
        # Let's add a small penalty on magnitude changes to avoid explosion.
        seasonal_mag = torch.norm(states[:, :, 2:4], dim=-1)
        if states.shape[1] > 1:
            mag_diff = seasonal_mag[:, 1:] - seasonal_mag[:, :-1]
            seasonal_loss = torch.mean(mag_diff ** 2) * 0.1
        else:
            seasonal_loss = torch.tensor(0.0, device=states.device)

        # Residual (dim 4): minimize magnitude (sparsity)
        residual_loss = torch.mean(states[:, :, 4] ** 2)
        
        return level_loss + trend_loss + seasonal_loss + residual_loss

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute total disentanglement loss.
        
        Args:
            states: (batch, seq_len, state_dim) with state_dim == 5
        """
        _, _, state_dim = states.shape
        if state_dim != self.EXPECTED_STATE_DIM:
            raise ValueError(
                f"DisentanglementLoss expects state_dim={self.EXPECTED_STATE_DIM}; got {state_dim}."
            )
        l_cov = self.covariance_loss(states)
        l_temp = self.temporal_consistency_loss(states)
        
        return self.lambda_cov * l_cov + self.lambda_temporal * l_temp
