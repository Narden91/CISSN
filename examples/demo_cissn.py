import torch
import numpy as np


from cissn.models import DisentangledStateEncoder, ForecastHead
from cissn.losses.disentangle_loss import DisentanglementLoss
from cissn.conformal import StateConditionalConformal
from cissn.explanations import ForecastExplainer

def test_cissn_flow():
    print("Initializing CISSN components...")
    
    input_dim = 10
    state_dim = 5
    hidden_dim = 32
    batch_size = 16
    seq_len = 20
    horizon = 5

    encoder = DisentangledStateEncoder(input_dim, state_dim, hidden_dim)
    head = ForecastHead(state_dim, output_dim=1, horizon=horizon)
    loss_fn = DisentanglementLoss()
    conformal = StateConditionalConformal()
    explainer = ForecastExplainer(head)
    
    print("Models initialized successfully.")
    
    x = torch.randn(batch_size, seq_len, input_dim)
    states = encoder(x, return_all_states=True)
    final_state = states[:, -1, :]
    
    print(f"Encoded state shape: {final_state.shape}")
    
    forecast = head(final_state)
    print(f"Forecast shape: {forecast.shape}")

    dynamics = encoder._structured_dynamics()
    d_loss = loss_fn(states, dynamics)
    print(f"Disentanglement Loss: {d_loss.item():.4f}")
    
    print("Calibrating conformal predictor...")
    calibration_states = torch.randn(100, state_dim)
    calibration_residuals = torch.abs(torch.randn(100))
    conformal.fit(calibration_states, calibration_residuals)
    
    lower, upper = conformal.predict(final_state, forecast)
    print(f"Prediction Interval width: {(upper - lower).mean().item():.4f}")

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
