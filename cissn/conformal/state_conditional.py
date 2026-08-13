import logging

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple, Optional

logger = logging.getLogger(__name__)


def split_conformal_q_level(n: int, alpha: float) -> float:
    """Compute exact finite-sample split-conformal quantile level: ceil((n+1)(1-alpha)) / (n+1)."""
    if n <= 0:
        raise ValueError(f"n must be a positive integer; got {n}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie strictly between 0 and 1; got {alpha}.")
    return min(np.ceil((n + 1) * (1 - alpha)) / (n + 1), 1.0)


def _to_numpy(value: Union[torch.Tensor, np.ndarray], name: str) -> np.ndarray:
    """Shared by StateConditionalConformal and StateScaledConformal."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    return array


def _validate_states(states: np.ndarray) -> np.ndarray:
    """Shared by StateConditionalConformal and StateScaledConformal."""
    if states.ndim != 2:
        raise ValueError(f"states must have shape (n_samples, state_dim); got {states.shape}.")
    return states


def _compute_quantile(residuals: np.ndarray, q_level: float):
    """Shared by StateConditionalConformal and StateScaledConformal."""
    return np.quantile(residuals, q_level, axis=0, method="higher")


class StateConditionalConformal:
    """
    State-Conditional Conformal Prediction (SCCP).

    Uses latent states to cluster time steps and compute adaptive prediction intervals.
    """

    VALID_MULTIVARIATE_STRATEGIES = {"per_feature", "max"}

    def __init__(
        self,
        alpha: float = 0.1,
        n_clusters: int = 5,
        multivariate_strategy: str = "per_feature",
        random_state: int = 42,
        calibration_stride: int = 1,
    ):
        """
        Args:
            alpha: Significance level (coverage = 1 - alpha)
            n_clusters: Number of state clusters
            multivariate_strategy: 'per_feature' calibrates a separate score for
                every horizon-feature cell; 'max' calibrates one score for a
                simultaneous horizon-feature block.
            random_state: Seed for KMeans reproducibility.
            calibration_stride: Keep every kth chronological calibration origin.
                This does not create an unconditional guarantee; it is recorded
                for dependence-aware analysis under the study assumptions.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must lie strictly between 0 and 1; got {alpha}.")
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be a positive integer; got {n_clusters}.")
        if multivariate_strategy not in self.VALID_MULTIVARIATE_STRATEGIES:
            supported = ", ".join(sorted(self.VALID_MULTIVARIATE_STRATEGIES))
            raise ValueError(f"Unknown multivariate strategy {multivariate_strategy!r}. Supported values: {supported}.")
        if calibration_stride <= 0:
            raise ValueError(f"calibration_stride must be positive; got {calibration_stride}.")

        self.alpha = alpha
        self.n_clusters = n_clusters
        self.multivariate_strategy = multivariate_strategy
        if multivariate_strategy == "max":
            self.coverage_scope = "simultaneous"
        elif multivariate_strategy == "per_feature":
            self.coverage_scope = "marginal"
        else:
            self.coverage_scope = "marginal"
        self.random_state = random_state
        self.calibration_stride = calibration_stride
        self._reset_fit_state()

    def _reset_fit_state(self) -> None:
        self.kmeans: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.quantiles: dict = {}
        self.quantile_shape: tuple = ()
        self.cluster_sizes_: dict = {}
        self.cluster_fallbacks_: dict = {}
        self.calibration_samples_ = 0
        self.partition_fitted = False
        self.last_predicted_clusters_: Optional[np.ndarray] = None
        self.calibrated = False

    _to_numpy = staticmethod(_to_numpy)
    _validate_states = staticmethod(_validate_states)
    _compute_quantile = staticmethod(_compute_quantile)

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
        if self.multivariate_strategy == "max":
            return flattened.max(axis=1), ()
        return residuals, tuple(residuals.shape[1:])

    @staticmethod
    def _compute_acf1(residuals: np.ndarray) -> Optional[float]:
        """Compute lag-1 autocorrelation. Returns None if < 5 samples."""
        if residuals.shape[0] < 5:
            return None
        r = residuals
        if r.ndim > 1:
            r = r.reshape(r.shape[0], -1).mean(axis=1)
        diff_r = r - r.mean()
        denom = float(np.linalg.norm(diff_r[:-1]) * np.linalg.norm(diff_r[1:]))
        if denom <= 1e-12:
            return 0.0
        rho = float(np.dot(diff_r[:-1], diff_r[1:]) / denom)
        return 0.0 if not np.isfinite(rho) else rho

    def _build_quantile_tensor(self, q_values, point_forecasts: torch.Tensor) -> torch.Tensor:
        q_array = np.stack(q_values, axis=0) if self.quantile_shape else np.asarray(q_values)
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
            f"expected {self.quantile_shape} or (horizon, {self.quantile_shape[0]}) "
            f"for per-feature output calibration, got {forecast_shape}."
        )

    def fit_partition(self, reference_states: Union[torch.Tensor, np.ndarray]) -> None:
        """Learn a state partition from training data before calibration."""
        states = self._validate_states(self._to_numpy(reference_states, "reference_states"))
        self._reset_fit_state()
        minimum = max(5, int(np.ceil(1.0 / self.alpha)))
        n_clusters = min(self.n_clusters, max(1, states.shape[0] // minimum))
        self.scaler = StandardScaler()
        scaled_states = self.scaler.fit_transform(states)

        while True:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(scaled_states)
            if n_clusters == 1 or np.bincount(labels, minlength=n_clusters).min() >= minimum:
                break
            n_clusters -= 1

        self.kmeans = kmeans
        self.partition_fitted = True

    def calibrate(
        self,
        states: Union[torch.Tensor, np.ndarray],
        residuals: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Calibrate a frozen state partition on chronological residuals."""
        if not self.partition_fitted:
            raise RuntimeError("Call fit_partition() before calibrate().")

        states_np = self._validate_states(self._to_numpy(states, "states"))
        residuals_np = self._to_numpy(residuals, "residuals")
        indices = np.arange(0, states_np.shape[0], self.calibration_stride)
        states_np = states_np[indices]
        residuals_np = residuals_np[indices]
        residuals_np, self.quantile_shape = self._prepare_residuals(residuals_np, states_np.shape[0])
        labels = self.assign_clusters(states_np)
        minimum = max(5, int(np.ceil(1.0 / self.alpha)))
        global_quantile = self._compute_quantile(
            residuals_np, split_conformal_q_level(residuals_np.shape[0], self.alpha)
        )

        self.quantiles = {}
        self.cluster_sizes_ = {}
        self.cluster_fallbacks_ = {}
        self.calibration_samples_ = int(states_np.shape[0])
        for cluster in range(self.kmeans.n_clusters):
            cluster_residuals = residuals_np[labels == cluster]
            count = cluster_residuals.shape[0]
            self.cluster_sizes_[cluster] = int(count)
            if count < minimum:
                self.quantiles[cluster] = global_quantile
                self.cluster_fallbacks_[cluster] = "global_quantile"
                continue
            self.quantiles[cluster] = self._compute_quantile(
                cluster_residuals, split_conformal_q_level(count, self.alpha)
            )
            self.cluster_fallbacks_[cluster] = None

        self.calibrated = True
        logger.info(
            "SCCP calibrated: clusters=%d, stride=%d, samples=%d",
            self.kmeans.n_clusters, self.calibration_stride, self.calibration_samples_,
        )

    def fit(self, states: Union[torch.Tensor, np.ndarray], residuals: Union[torch.Tensor, np.ndarray]) -> None:
        """Convenience API for IID tests; experiments use separate partition and calibration data."""
        self.fit_partition(states)
        self.calibrate(states, residuals)

    def assign_clusters(self, states: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Assign states to fitted conformal clusters."""
        if not self.partition_fitted or self.kmeans is None or self.scaler is None:
            raise RuntimeError("State partition is not fitted. Call fit_partition() first.")
        states_np = self._validate_states(self._to_numpy(states, "states"))
        # sklearn requires the predict dtype to match the fitted centroids, so a
        # caller passing float32 states to a float64-fitted KMeans (or vice
        # versa) would hit a buffer dtype mismatch.
        scaled = np.asarray(
            self.scaler.transform(states_np), dtype=self.kmeans.cluster_centers_.dtype
        )
        return self.kmeans.predict(scaled)

    def get_cluster_stats(self) -> dict:
        """Return JSON-serializable fitted cluster diagnostics."""
        if not self.calibrated:
            raise RuntimeError("Conformal predictor not calibrated. Call fit() first.")
        stats = {}
        for k, n_k in self.cluster_sizes_.items():
            q = self.quantiles.get(k)
            stats[int(k)] = {
                "n_samples": int(n_k),
                "quantile_shape": list(np.asarray(q).shape),
                "quantile_mean": float(np.asarray(q).mean()) if q is not None else None,
                "fallback": self.cluster_fallbacks_.get(k),
            }
        return {
            "alpha": float(self.alpha),
            "requested_n_clusters": int(self.n_clusters),
            "fitted_n_clusters": int(self.kmeans.n_clusters),
            "multivariate_strategy": self.multivariate_strategy,
            "coverage_scope": self.coverage_scope,
            "calibration_stride": int(self.calibration_stride),
            "calibration_samples": int(self.calibration_samples_),
            "clusters": stats,
        }

    def predict(
        self,
        states: Union[torch.Tensor, np.ndarray],
        point_forecasts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate prediction intervals.

        Args:
            states: (n_samples, state_dim)
            point_forecasts: Forecast tensor with leading sample axis.

        Returns:
            lower_bound, upper_bound with the same shape as point_forecasts.
        """
        if not self.calibrated:
            raise RuntimeError("Conformal predictor not calibrated. Call fit() first.")
        if not isinstance(point_forecasts, torch.Tensor):
            raise TypeError("point_forecasts must be a torch.Tensor.")
        if point_forecasts.ndim == 0:
            raise ValueError("point_forecasts must include a sample axis as the first dimension.")

        states_np = self._validate_states(self._to_numpy(states, "states"))
        if states_np.shape[0] != point_forecasts.shape[0]:
            raise ValueError(
                "states and point_forecasts must share the same number of samples; "
                f"got {states_np.shape[0]} and {point_forecasts.shape[0]}."
            )

        cluster_labels = self.assign_clusters(states_np)
        self.last_predicted_clusters_ = cluster_labels
        q_values = [self.quantiles[k] for k in cluster_labels]
        q_tensor = self._build_quantile_tensor(q_values, point_forecasts)
        return point_forecasts - q_tensor, point_forecasts + q_tensor

    def diagnose_dependence(
        self,
        states: Union[torch.Tensor, np.ndarray],
        residuals: Union[torch.Tensor, np.ndarray],
    ) -> dict:
        """
        Report within-cluster lag-1 autocorrelation without modifying intervals.

        Returns a dict mapping cluster_id -> dict with keys:
            acf_lag1, n_samples, and a warning when serial dependence is high.
        """
        states = self._validate_states(self._to_numpy(states, "states"))
        residuals = self._to_numpy(residuals, "residuals")
        # Reduce residuals the same way fit() does so ACF values are comparable.
        n_samples = states.shape[0]
        residuals, _ = self._prepare_residuals(residuals, n_samples)
        if not self.calibrated:
            raise RuntimeError("Conformal predictor not calibrated. Call fit() first.")
        cluster_labels = self.assign_clusters(states)

        results = {}
        for k in range(self.kmeans.n_clusters):
            mask = cluster_labels == k
            n_k = mask.sum()
            entry = {"n_samples": int(n_k)}

            if n_k < 5:
                entry["acf_lag1"] = None
                entry["warning"] = "too few samples for ACF computation"
                results[k] = entry
                continue

            rho = self._compute_acf1(residuals[mask])
            entry["acf_lag1"] = rho

            if rho is not None and abs(rho) > 0.3:
                entry["warning"] = "substantial serial dependence"

            results[k] = entry
        return results


class StateScaledConformal:
    """
    State-Scaled Conformal Prediction.

    Uses the learned latent state as a continuous difficulty estimator: a
    log-linear regression of per-sample residual magnitude on the (scaled)
    state gives a scale sigma(s), and the conformal score is the residual
    normalized by that scale (the standard normalized/locally-weighted
    conformal score of Papadopoulos et al., applied with a learned structured
    state as the difficulty estimator).

    This differs from StateConditionalConformal, which discretizes the state
    into K-Means clusters and calibrates one quantile per cluster. Measured on
    this repository's diagnostic runs, cluster membership captures only a
    fraction of the state's relationship to residual scale relative to using
    the continuous state directly (R^2 ~0.17 vs ~0.73), which the continuous
    scale is designed to recover.
    """

    VALID_MULTIVARIATE_STRATEGIES = {"per_feature", "max"}

    def __init__(
        self,
        alpha: float = 0.1,
        multivariate_strategy: str = "per_feature",
        ridge: float = 1e-3,
        sigma_floor: float = 1e-3,
    ):
        """
        Args:
            alpha: Significance level (coverage = 1 - alpha)
            multivariate_strategy: 'per_feature' calibrates a separate
                normalized score for every horizon-feature cell; 'max'
                calibrates one normalized score for a simultaneous
                horizon-feature block.
            ridge: L2 penalty added to the scale regression for numerical
                stability; the regression target is 1-D (mean log-residual),
                so this is a small stabilizer, not a tuned hyperparameter.
            sigma_floor: Minimum allowed value of the fitted scale, guarding
                against division by ~0 for states far from the fitted range.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must lie strictly between 0 and 1; got {alpha}.")
        if multivariate_strategy not in self.VALID_MULTIVARIATE_STRATEGIES:
            supported = ", ".join(sorted(self.VALID_MULTIVARIATE_STRATEGIES))
            raise ValueError(f"Unknown multivariate strategy {multivariate_strategy!r}. Supported values: {supported}.")
        if ridge < 0:
            raise ValueError(f"ridge must be non-negative; got {ridge}.")
        if sigma_floor <= 0:
            raise ValueError(f"sigma_floor must be positive; got {sigma_floor}.")

        self.alpha = alpha
        self.multivariate_strategy = multivariate_strategy
        self.coverage_scope = "simultaneous" if multivariate_strategy == "max" else "marginal"
        self.ridge = ridge
        self.sigma_floor = sigma_floor
        self._reset_fit_state()

    def _reset_fit_state(self) -> None:
        self.scaler: Optional[StandardScaler] = None
        self.beta_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.quantiles: dict = {}
        self.quantile_shape: tuple = ()
        self.calibration_samples_ = 0
        self.scale_fitted = False
        self.calibrated = False

    _to_numpy = staticmethod(_to_numpy)
    _validate_states = staticmethod(_validate_states)
    _compute_quantile = staticmethod(_compute_quantile)

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
        if self.multivariate_strategy == "max":
            return flattened.max(axis=1), ()
        return residuals, tuple(residuals.shape[1:])

    def fit_scale(
        self,
        reference_states: Union[torch.Tensor, np.ndarray],
        reference_residuals: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Fit sigma(s) = exp(beta . scale(s) + intercept) from training data.

        The regression target is log(mean absolute residual + eps) per
        sample, averaged over every non-sample axis of the residual tensor,
        so a single scalar scale is learned per state even when residuals
        are multi-dimensional (horizon x feature). Must be called before
        calibrate(), and must be fit on data disjoint from the calibration
        split -- mirroring StateConditionalConformal.fit_partition().
        """
        states = self._validate_states(self._to_numpy(reference_states, "reference_states"))
        residuals = self._to_numpy(reference_residuals, "reference_residuals")
        if residuals.shape[0] != states.shape[0]:
            raise ValueError(
                "reference_states and reference_residuals must share the same number of samples; "
                f"got {states.shape[0]} and {residuals.shape[0]}."
            )
        if np.any(residuals < 0):
            raise ValueError("reference_residuals must be absolute non-negative errors.")
        self._reset_fit_state()

        self.scaler = StandardScaler()
        scaled_states = self.scaler.fit_transform(states)

        flat = residuals.reshape(residuals.shape[0], -1)
        target = np.log(flat.mean(axis=1) + 1e-8)

        design = np.concatenate([scaled_states, np.ones((scaled_states.shape[0], 1))], axis=1)
        n_features = design.shape[1]
        gram = design.T @ design + self.ridge * np.eye(n_features)
        coefs = np.linalg.solve(gram, design.T @ target)
        self.beta_ = coefs[:-1]
        self.intercept_ = float(coefs[-1])
        self.scale_fitted = True

    def _sigma(self, states: np.ndarray) -> np.ndarray:
        if not self.scale_fitted:
            raise RuntimeError("Call fit_scale() before computing sigma.")
        scaled = self.scaler.transform(states)
        log_sigma = scaled @ self.beta_ + self.intercept_
        return np.maximum(np.exp(log_sigma), self.sigma_floor)

    def calibrate(
        self,
        states: Union[torch.Tensor, np.ndarray],
        residuals: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Calibrate a frozen state scale on chronological calibration residuals."""
        if not self.scale_fitted:
            raise RuntimeError("Call fit_scale() before calibrate().")

        states_np = self._validate_states(self._to_numpy(states, "states"))
        residuals_np = self._to_numpy(residuals, "residuals")
        residuals_np, self.quantile_shape = self._prepare_residuals(residuals_np, states_np.shape[0])

        sigma = self._sigma(states_np)
        normalized = residuals_np / sigma.reshape(-1, *([1] * (residuals_np.ndim - 1)))

        q_level = split_conformal_q_level(normalized.shape[0], self.alpha)
        self.quantiles = self._compute_quantile(normalized, q_level)
        self.calibration_samples_ = int(states_np.shape[0])
        self.calibrated = True
        logger.info(
            "State-scaled CP calibrated: samples=%d, strategy=%s",
            self.calibration_samples_, self.multivariate_strategy,
        )

    def fit(self, states: Union[torch.Tensor, np.ndarray], residuals: Union[torch.Tensor, np.ndarray]) -> None:
        """Convenience API for IID tests; experiments use separate scale-fit and calibration data."""
        self.fit_scale(states, residuals)
        self.calibrate(states, residuals)

    def get_scale_stats(self) -> dict:
        """Return JSON-serializable fitted scale diagnostics."""
        if not self.calibrated:
            raise RuntimeError("Conformal predictor not calibrated. Call fit() first.")
        return {
            "alpha": float(self.alpha),
            "multivariate_strategy": self.multivariate_strategy,
            "coverage_scope": self.coverage_scope,
            "calibration_samples": int(self.calibration_samples_),
            "beta": [float(b) for b in self.beta_],
            "intercept": float(self.intercept_),
            "ridge": float(self.ridge),
            "sigma_floor": float(self.sigma_floor),
            "quantile_shape": list(np.asarray(self.quantiles).shape),
        }

    def predict(
        self,
        states: Union[torch.Tensor, np.ndarray],
        point_forecasts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate prediction intervals scaled by the fitted per-sample sigma(s).

        Args:
            states: (n_samples, state_dim)
            point_forecasts: Forecast tensor with leading sample axis.

        Returns:
            lower_bound, upper_bound with the same shape as point_forecasts.
        """
        if not self.calibrated:
            raise RuntimeError("Conformal predictor not calibrated. Call fit() first.")
        if not isinstance(point_forecasts, torch.Tensor):
            raise TypeError("point_forecasts must be a torch.Tensor.")
        if point_forecasts.ndim == 0:
            raise ValueError("point_forecasts must include a sample axis as the first dimension.")

        states_np = self._validate_states(self._to_numpy(states, "states"))
        if states_np.shape[0] != point_forecasts.shape[0]:
            raise ValueError(
                "states and point_forecasts must share the same number of samples; "
                f"got {states_np.shape[0]} and {point_forecasts.shape[0]}."
            )

        sigma = self._sigma(states_np)
        sigma_tensor = torch.as_tensor(sigma, device=point_forecasts.device, dtype=point_forecasts.dtype)
        while sigma_tensor.ndim < point_forecasts.ndim:
            sigma_tensor = sigma_tensor.unsqueeze(-1)

        # Broadcast the single fitted quantile array to the forecast shape,
        # then scale per-sample by sigma(s) -- the quantile is shared across
        # samples (fit on normalized residuals), sigma is not.
        q_tensor = torch.as_tensor(
            np.asarray(self.quantiles), device=point_forecasts.device, dtype=point_forecasts.dtype
        )
        forecast_shape = tuple(point_forecasts.shape[1:])
        if not self.quantile_shape:
            while q_tensor.ndim < point_forecasts.ndim - 1:
                q_tensor = q_tensor.unsqueeze(-1)
            q_tensor = q_tensor.unsqueeze(0)
        elif forecast_shape == self.quantile_shape:
            q_tensor = q_tensor.unsqueeze(0)
        elif len(self.quantile_shape) == 1 and point_forecasts.ndim == 3 and forecast_shape[-1:] == self.quantile_shape:
            q_tensor = q_tensor.reshape(1, 1, -1)
        else:
            raise ValueError(
                "point_forecasts trailing shape is incompatible with calibrated quantiles: "
                f"expected {self.quantile_shape} or (horizon, {self.quantile_shape[0] if self.quantile_shape else None}) "
                f"for per-feature output calibration, got {forecast_shape}."
            )

        width = q_tensor * sigma_tensor
        return point_forecasts - width, point_forecasts + width
