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
            state, horizon_idx=horizon_idx, output_idx=output_idx
        )

        keys = ("level", "trend", "seasonal", "residual", "bias", "linear_prediction",
                "refinement_contribution", "total_prediction")
        tensors = {k: contributions[k].detach().cpu() for k in keys}

        return [
            ExplanationResult(
                level_contribution=float(tensors["level"][i]),
                trend_contribution=float(tensors["trend"][i]),
                seasonal_contribution=float(tensors["seasonal"][i]),
                residual_contribution=float(tensors["residual"][i]),
                bias=float(tensors["bias"][i]),
                linear_prediction=float(tensors["linear_prediction"][i]),
                refinement_contribution=float(tensors["refinement_contribution"][i]),
                total_prediction=float(tensors["total_prediction"][i]),
            )
            for i in range(state.size(0))
        ]
