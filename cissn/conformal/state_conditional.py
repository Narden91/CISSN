import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple

class StateConditionalConformal:
    """
    State-Conditional Conformal Prediction (SCCP).
    
    Uses latent states to cluster time steps and compute adaptive prediction intervals.
    """
    
    VALID_MULTIVARIATE_STRATEGIES = {'per_feature', 'max', 'mean'}

    def __init__(self, alpha: float = 0.1, n_clusters: int = 5, multivariate_strategy: str = 'per_feature'):
        """
        Args:
            alpha: Significance level (coverage = 1 - alpha)
            n_clusters: Number of state clusters
            multivariate_strategy: How to calibrate multi-output residuals.
                - 'per_feature': keep the residual trailing dimensions and learn one quantile per element.
                - 'max': reduce every sample to a single worst-case residual.
                - 'mean': reduce every sample to a single mean residual.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must lie strictly between 0 and 1; got {alpha}.")
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be a positive integer; got {n_clusters}.")
        if multivariate_strategy not in self.VALID_MULTIVARIATE_STRATEGIES:
            supported = ', '.join(sorted(self.VALID_MULTIVARIATE_STRATEGIES))
            raise ValueError(f"Unknown multivariate strategy {multivariate_strategy!r}. Supported values: {supported}.")

        self.alpha = alpha
        self.n_clusters = n_clusters
        self.multivariate_strategy = multivariate_strategy
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.quantiles = {}  # {cluster_id: quantile_value}
        self.quantile_shape = ()
        self.calibrated = False

    @staticmethod
    def _to_numpy(value: Union[torch.Tensor, np.ndarray], name: str) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.size == 0:
            raise ValueError(f"{name} must contain at least one sample.")
        return array

    @staticmethod
    def _validate_states(states: np.ndarray) -> np.ndarray:
        if states.ndim != 2:
            raise ValueError(f"states must have shape (n_samples, state_dim); got {states.shape}.")
        return states

    def _prepare_residuals(self, residuals: np.ndarray, n_samples: int) -> Tuple[np.ndarray, Tuple[int, ...]]:
        if residuals.ndim == 0:
            raise ValueError("residuals must include a sample axis as the first dimension.")
        if residuals.shape[0] != n_samples:
            raise ValueError(
                f"states and residuals must share the same number of samples; got {n_samples} and {residuals.shape[0]}."
            )
        if np.any(residuals < 0):
            raise ValueError("residuals must be absolute non-negative errors.")

        if residuals.ndim == 1:
            return residuals, ()

        flattened = residuals.reshape(n_samples, -1)
        if self.multivariate_strategy == 'max':
            return flattened.max(axis=1), ()
        if self.multivariate_strategy == 'mean':
            return flattened.mean(axis=1), ()
        return residuals, tuple(residuals.shape[1:])

    @staticmethod
    def _compute_quantile(residuals: np.ndarray, q_level: float):
        return np.quantile(residuals, q_level, axis=0, method='higher')

    def _build_quantile_tensor(self, q_values, point_forecasts: torch.Tensor) -> torch.Tensor:
        if self.quantile_shape:
            q_array = np.stack(q_values, axis=0)
        else:
            q_array = np.asarray(q_values)

        q_tensor = torch.as_tensor(q_array, device=point_forecasts.device, dtype=point_forecasts.dtype)
        forecast_shape = tuple(point_forecasts.shape[1:])

        if not self.quantile_shape:
            while q_tensor.ndim < point_forecasts.ndim:
                q_tensor = q_tensor.unsqueeze(-1)
            return q_tensor

        if forecast_shape == self.quantile_shape:
            return q_tensor

        if len(self.quantile_shape) == 1 and point_forecasts.ndim == 3 and forecast_shape[-1:] == self.quantile_shape:
            return q_tensor.unsqueeze(1)

        raise ValueError(
            "point_forecasts trailing shape is incompatible with calibrated quantiles: "
            f"expected {self.quantile_shape} or (horizon, {self.quantile_shape[0]}) for per-feature output calibration, got {forecast_shape}."
        )
        
    def fit(self, states: Union[torch.Tensor, np.ndarray], residuals: Union[torch.Tensor, np.ndarray]):
        """
        Calibrate the conformal predictor.
        
        Args:
            states: (n_samples, state_dim) - Latent states from calibration set
            residuals: Absolute residuals with leading sample axis. Supported shapes:
                - (n_samples,) for scalar calibration
                - (n_samples, output_dim) for per-output calibration
                - (n_samples, horizon, output_dim) for per-horizon, per-output calibration
        """
        states = self._validate_states(self._to_numpy(states, 'states'))
        residuals = self._to_numpy(residuals, 'residuals')

        n_samples = states.shape[0]
        residuals, self.quantile_shape = self._prepare_residuals(residuals, n_samples)
        n_clusters = min(self.n_clusters, n_samples)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.quantiles = {}
        # 1. Cluster states
        scaled_states = self.scaler.fit_transform(states)
        self.kmeans.fit(scaled_states)
        cluster_labels = self.kmeans.predict(scaled_states)

        # 2. Compute per-cluster quantiles
        for k in range(self.kmeans.n_clusters):
            # Get residuals for this cluster
            mask = (cluster_labels == k)
            cluster_residuals = residuals[mask]
            
            if cluster_residuals.shape[0] == 0:
                cluster_residuals = residuals

            n_k = cluster_residuals.shape[0]
            q_level = np.ceil((n_k + 1) * (1 - self.alpha)) / n_k
            q_level = min(q_level, 1.0)
            self.quantiles[k] = self._compute_quantile(cluster_residuals, q_level)
                
        self.calibrated = True
        
    def predict(self, states: Union[torch.Tensor, np.ndarray], point_forecasts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate prediction intervals.
        
        Args:
            states: (n_samples, state_dim)
            point_forecasts: Forecast tensor with leading sample axis. Supported shapes:
                - (n_samples,)
                - (n_samples, output_dim)
                - (n_samples, horizon, output_dim)
            
        Returns:
            lower_bound, upper_bound with the same shape as point_forecasts.
        """
        if not self.calibrated:
            raise RuntimeError("Conformal predictor not calibrated. Call fit() first.")
        if not isinstance(point_forecasts, torch.Tensor):
            raise TypeError("point_forecasts must be a torch.Tensor.")
        if point_forecasts.ndim == 0:
            raise ValueError("point_forecasts must include a sample axis as the first dimension.")
            
        states_np = self._validate_states(self._to_numpy(states, 'states'))
        if states_np.shape[0] != point_forecasts.shape[0]:
            raise ValueError(
                "states and point_forecasts must share the same number of samples; "
                f"got {states_np.shape[0]} and {point_forecasts.shape[0]}."
            )
            
        # Assign clusters
        scaled_states = self.scaler.transform(states_np)
        cluster_labels = self.kmeans.predict(scaled_states)
        
        # Retrieve quantiles
        q_values = [self.quantiles[k] for k in cluster_labels]
        q_tensor = self._build_quantile_tensor(q_values, point_forecasts)
        
        lower = point_forecasts - q_tensor
        upper = point_forecasts + q_tensor
        
        return lower, upper
