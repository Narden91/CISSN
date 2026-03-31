from dataclasses import dataclass
from typing import List
import torch

@dataclass
class ExplanationResult:
    level_contribution: float
    trend_contribution: float
    seasonal_contribution: float
    residual_contribution: float
    bias: float
    linear_prediction: float
    refinement_contribution: float
    total_prediction: float

class ForecastExplainer:
    """
    Explains forecasts by decomposing them into interpretable state components.
    """
    
    def __init__(self, forecast_head):
        self.forecast_head = forecast_head
        
    def explain(
        self,
        state: torch.Tensor,
        horizon_idx: int = 0,
        output_idx: int = 0,
    ) -> List[ExplanationResult]:
        """
        Explain a selected forecast coordinate for every sample in a batch.
        
        Args:
            state: (batch, state_dim)
            horizon_idx: Forecast horizon index to explain.
            output_idx: Output channel index to explain.
            
        Returns:
            List of ExplanationResult objects
        """
        contributions = self.forecast_head.get_contributions(
            state,
            horizon_idx=horizon_idx,
            output_idx=output_idx,
        )

        level = contributions['level'].detach().cpu()
        trend = contributions['trend'].detach().cpu()
        seasonal = contributions['seasonal'].detach().cpu()
        residual = contributions['residual'].detach().cpu()
        bias = contributions['bias'].detach().cpu()
        linear_prediction = contributions['linear_prediction'].detach().cpu()
        refinement = contributions['refinement_contribution'].detach().cpu()
        total_prediction = contributions['total_prediction'].detach().cpu()
        
        batch_size = state.size(0)
        explanations = []
        
        for i in range(batch_size):
            res = ExplanationResult(
                level_contribution=float(level[i].item()),
                trend_contribution=float(trend[i].item()),
                seasonal_contribution=float(seasonal[i].item()),
                residual_contribution=float(residual[i].item()),
                bias=float(bias[i].item()),
                linear_prediction=float(linear_prediction[i].item()),
                refinement_contribution=float(refinement[i].item()),
                total_prediction=float(total_prediction[i].item()),
            )
            explanations.append(res)
            
        return explanations
