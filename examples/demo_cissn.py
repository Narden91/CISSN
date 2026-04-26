import torch
import numpy as np
import sys
import os

# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cissn.models import DisentangledStateEncoder, ForecastHead
from cissn.losses.disentangle_loss import DisentanglementLoss
from cissn.conformal import StateConditionalConformal
from cissn.explanations import ForecastExplainer

def test_cissn_flow():
    print("Initializing CISSN components...")
    
    # Hyperparameters
    input_dim = 10
    state_dim = 5 # Updated to 5
    hidden_dim = 32
    batch_size = 16
    seq_len = 20
    horizon = 5
    
    # 1. Initialize Models
    encoder = DisentangledStateEncoder(input_dim, state_dim, hidden_dim)
    head = ForecastHead(state_dim, output_dim=1, horizon=horizon)
    loss_fn = DisentanglementLoss()
    conformal = StateConditionalConformal()
    explainer = ForecastExplainer(head)
    
    print("Models initialized successfully.")
    
    # 2. Forward Pass
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Encode
    states = encoder(x, return_all_states=True)  # (batch, seq_len, state_dim)
    final_state = states[:, -1, :]  # (batch, state_dim)
    
    print(f"Encoded state shape: {final_state.shape}")
    
    # Forecast
    forecast = head(final_state)  # (batch, horizon, 1)
    print(f"Forecast shape: {forecast.shape}")
    
    # 3. Compute Loss
    dynamics = encoder._structured_dynamics()
    d_loss = loss_fn(states, dynamics)
    print(f"Disentanglement Loss: {d_loss.item():.4f}")
    
    # 4. Conformal Prediction (Calibration)
    print("Calibrating conformal predictor...")
    calibration_states = torch.randn(100, state_dim) # Dummy states
    calibration_residuals = torch.abs(torch.randn(100)) # Dummy residuals
    conformal.fit(calibration_states, calibration_residuals)
    
    # Predict intervals for the full forecast tensor
    lower, upper = conformal.predict(final_state, forecast)
    print(f"Prediction Interval width: {(upper - lower).mean().item():.4f}")
    
    # 5. Explainability
    print("Generating explanations...")
    explanations = explainer.explain(final_state, horizon_idx=0, output_idx=0)
    
    print(f"Explanation for first sample:")
    e = explanations[0]
    print(f"  Level: {e.level_contribution:.4f}")
    print(f"  Trend: {e.trend_contribution:.4f}")
    print(f"  Seasonal: {e.seasonal_contribution:.4f}")
    print(f"  Residual: {e.residual_contribution:.4f}")
    print(f"  Bias: {e.bias:.4f}")
    print(f"  Linear Branch: {e.linear_prediction:.4f}")
    print(f"  Refinement: {e.refinement_contribution:.4f}")
    print(f"  Total Forecast: {e.total_prediction:.4f}")
    
    print("\nSUCCESS: All components integrated and functioning.")

if __name__ == "__main__":
    test_cissn_flow()
