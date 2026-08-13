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
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    return array


def _validate_states(states: np.ndarray) -> np.ndarray:
    if states.ndim != 2:
        raise ValueError(f"states must have shape (n_samples, state_dim); got {states.shape}.")
    return states


def _compute_quantile(residuals: np.ndarray, q_level: float):
    return np.quantile(residuals, q_level, axis=0, method="higher")


class _StateConformalBase:
    """Input handling and score geometry shared by both state-conditional predictors.

    Both subclasses take the same inputs (states plus absolute residuals),
    support the same two score geometries, and differ only in how they turn a
    state into a quantile: StateConditionalConformal via a K-Means partition,
    StateScaledConformal via a continuous sigma(state).
    """

    VALID_MULTIVARIATE_STRATEGIES = {"per_feature", "max"}

    _to_numpy = staticmethod(_to_numpy)
    _validate_states = staticmethod(_validate_states)
    _compute_quantile = staticmethod(_compute_quantile)

    def _validate_common_args(self, alpha: float, multivariate_strategy: str) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must lie strictly between 0 and 1; got {alpha}.")
        if multivariate_strategy not in self.VALID_MULTIVARIATE_STRATEGIES:
            supported = ", ".join(sorted(self.VALID_MULTIVARIATE_STRATEGIES))
            raise ValueError(
                f"Unknown multivariate strategy {multivariate_strategy!r}. Supported values: {supported}."
            )
        self.alpha = alpha
        self.multivariate_strategy = multivariate_strategy
        # 'max' reduces the whole horizon-feature block to one score per sample,
        # so what it guarantees is simultaneous rather than marginal coverage.
        self.coverage_scope = "simultaneous" if multivariate_strategy == "max" else "marginal"

    def _prepare_residuals(self, residuals: np.ndarray, n_samples: int) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """Reduce residuals to conformal scores and report their trailing shape."""
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
        if self.multivariate_strategy == "max":
            return residuals.reshape(n_samples, -1).max(axis=1), ()
        return residuals, tuple(residuals.shape[1:])

    def _check_predict_inputs(
        self, states: Union[torch.Tensor, np.ndarray], point_forecasts: torch.Tensor
    ) -> np.ndarray:
        """Validate a predict() call and return the states as a 2-D array."""
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
        return states_np

    def _min_calibration_samples(self) -> int:
        """Smallest group size that can support an exact 1-alpha quantile."""
        return max(5, int(np.ceil(1.0 / self.alpha)))


class StateConditionalConformal(_StateConformalBase):
    """
    State-Conditional Conformal Prediction (SCCP).

    Partitions the latent state with K-Means and calibrates one quantile array
    per cluster, so each cluster gets its own full horizon-feature quantile
    surface rather than a single scale factor.
    """

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
        self._validate_common_args(alpha, multivariate_strategy)
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be a positive integer; got {n_clusters}.")
        if calibration_stride <= 0:
            raise ValueError(f"calibration_stride must be positive; got {calibration_stride}.")

        self.n_clusters = n_clusters
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

    def _build_quantile_tensor(self, cluster_labels: np.ndarray, point_forecasts: torch.Tensor) -> torch.Tensor:
        """Gather each sample's cluster quantile, broadcast to the forecast shape.

        Indexes a stacked (n_clusters, *quantile_shape) table by label rather
        than materialising one quantile array per sample: at ETTh1-h336 scale
        the per-sample form allocated ~48 MB through a Python list, while the
        table holds only n_clusters rows.
        """
        table = torch.as_tensor(
            np.stack([np.asarray(self.quantiles[k]) for k in sorted(self.quantiles)], axis=0),
            device=point_forecasts.device,
            dtype=point_forecasts.dtype,
        )
        index = torch.as_tensor(cluster_labels, device=point_forecasts.device, dtype=torch.long)
        q_tensor = table[index]
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
        minimum = self._min_calibration_samples()
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
        minimum = self._min_calibration_samples()
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
        states_np = self._check_predict_inputs(states, point_forecasts)
        cluster_labels = self.assign_clusters(states_np)
        self.last_predicted_clusters_ = cluster_labels
        q_tensor = self._build_quantile_tensor(cluster_labels, point_forecasts)
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


class StateScaledConformal(_StateConformalBase):
    """
    State-Scaled Conformal Prediction.

    Uses the learned latent state as a continuous difficulty estimator: a
    log-linear regression of per-sample residual magnitude on the (scaled)
    state gives a scale sigma(s), and the conformal score is the residual
    normalized by that scale (the standard normalized/locally-weighted
    conformal score of Papadopoulos et al., applied with a learned structured
    state as the difficulty estimator).

    This differs from StateConditionalConformal, which discretizes the state
    into K-Means clusters and calibrates one quantile per cluster. Cluster
    membership captures less of the state's relationship to residual *scale*
    than the continuous state does (R^2 ~0.17 vs ~0.73), but that is not the
    axis that decides interval quality: because each cluster gets its own full
    quantile surface, the cluster predictor captures the state x cell
    interaction that a scalar sigma cannot (see scale_geometry below).

    `scale_geometry` selects what shape sigma takes, and it is the difference
    between the mechanism working and not working:

    - ``'scalar'`` (default, original behaviour): one sigma per sample. Only
      ~1% of residual variance lives on the per-sample axis, and an oracle
      using the *true* per-sample test scale gains under 1% Winkler, so this
      geometry has almost no headroom under `per_feature`. Measured over three
      ETTh1-h336 RevIN seeds x four chronological cuts it is +0.011 Winkler
      against flat CP (better on 5/12), i.e. no improvement.
    - ``'per_cell'``: one sigma per horizon-feature cell, from an independent
      log-linear regression per cell. This lets the state reshape the whole
      quantile surface rather than only rescale its level, which is where the
      conditioning signal actually is: per-cluster quantile *shape* deviates by
      ~0.20-0.26 while its *level* varies by only ~0.03. Over the same three
      seeds x four cuts it is -0.237 Winkler against flat CP (better on 12/12,
      against -0.124 for cluster SCCP) at matched coverage.

    Both results are ETTh1-h336 development measurements, not protocol runs.
    `'per_cell'` wins only in the RevIN regime; on a pre-RevIN run it loses to
    flat CP substantially. See docs/methodology.md and
    scripts/diagnose_conditioning_headroom.py.
    """

    VALID_SCALE_GEOMETRIES = {"scalar", "per_cell"}

    def __init__(
        self,
        alpha: float = 0.1,
        multivariate_strategy: str = "per_feature",
        ridge: float = 1e-3,
        sigma_floor: float = 1e-3,
        scale_geometry: str = "scalar",
    ):
        """
        Args:
            alpha: Significance level (coverage = 1 - alpha)
            multivariate_strategy: 'per_feature' calibrates a separate
                normalized score for every horizon-feature cell; 'max'
                calibrates one normalized score for a simultaneous
                horizon-feature block.
            ridge: L2 penalty added to the scale regression for numerical
                stability. Under 'scalar' the regression target is 1-D so this
                is a small stabilizer; under 'per_cell' one regression is
                solved per cell against the same design matrix, and the fit is
                insensitive to this value over ridge in [1e-4, 1e2].
            sigma_floor: Minimum allowed value of the fitted scale, guarding
                against division by ~0 for states far from the fitted range.
            scale_geometry: 'scalar' fits one sigma per sample (default,
                original behaviour); 'per_cell' fits one sigma per
                horizon-feature cell. See the class docstring -- 'per_cell' is
                the geometry with measured headroom. Ignored when
                multivariate_strategy='max', which reduces residuals to one
                scalar per sample before any scale is applied.
        """
        self._validate_common_args(alpha, multivariate_strategy)
        if scale_geometry not in self.VALID_SCALE_GEOMETRIES:
            supported = ", ".join(sorted(self.VALID_SCALE_GEOMETRIES))
            raise ValueError(f"Unknown scale geometry {scale_geometry!r}. Supported values: {supported}.")
        if ridge < 0:
            raise ValueError(f"ridge must be non-negative; got {ridge}.")
        if sigma_floor <= 0:
            raise ValueError(f"sigma_floor must be positive; got {sigma_floor}.")

        self.ridge = ridge
        self.sigma_floor = sigma_floor
        # 'max' collapses residuals to one scalar per sample before scaling, so
        # a per-cell scale has nothing to attach to; fall back to scalar.
        self.scale_geometry = "scalar" if multivariate_strategy == "max" else scale_geometry
        self._reset_fit_state()

    def _reset_fit_state(self) -> None:
        self.scaler: Optional[StandardScaler] = None
        # Under 'scalar', beta_ is (state_dim,) and intercept_ a float. Under
        # 'per_cell', beta_ is (state_dim, n_cells) and intercept_ (n_cells,),
        # one independent regression per horizon-feature cell.
        self.beta_: Optional[np.ndarray] = None
        self.intercept_: Union[float, np.ndarray] = 0.0
        self.sigma_shape_: tuple = ()
        self.quantiles: dict = {}
        self.quantile_shape: tuple = ()
        self.calibration_samples_ = 0
        self.scale_fitted = False
        self.calibrated = False

    def fit_scale(
        self,
        reference_states: Union[torch.Tensor, np.ndarray],
        reference_residuals: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Fit sigma(s) = exp(beta . scale(s) + intercept) from training data.

        Under scale_geometry='scalar' the regression target is log(mean
        absolute residual + eps) per sample, averaged over every non-sample
        axis, so one scalar scale is learned per state.

        Under 'per_cell' the target is the log residual of each
        horizon-feature cell and one regression is solved per cell against the
        same design matrix, so sigma has the residual's own trailing shape.
        This lets the state reshape the quantile surface rather than only
        rescale its level; all cells share one Gram factorisation, so the cost
        over 'scalar' is one extra matrix multiply, not one solve per cell.

        Must be called before calibrate(), and must be fit on data disjoint
        from the calibration split -- mirroring
        StateConditionalConformal.fit_partition().
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
        geometry = self.scale_geometry
        self._reset_fit_state()

        self.scaler = StandardScaler()
        scaled_states = self.scaler.fit_transform(states)

        flat = residuals.reshape(residuals.shape[0], -1)
        if geometry == "per_cell" and residuals.ndim > 1:
            target = np.log(flat + 1e-8)
            self.sigma_shape_ = tuple(residuals.shape[1:])
        else:
            # 'scalar', or a 1-D residual where per-cell and scalar coincide.
            target = np.log(flat.mean(axis=1) + 1e-8)[:, None]
            self.sigma_shape_ = ()

        design = np.concatenate([scaled_states, np.ones((scaled_states.shape[0], 1))], axis=1)
        n_features = design.shape[1]
        gram = design.T @ design + self.ridge * np.eye(n_features)
        coefs = np.linalg.solve(gram, design.T @ target)
        if self.sigma_shape_:
            self.beta_ = coefs[:-1]
            self.intercept_ = coefs[-1]
        else:
            self.beta_ = coefs[:-1, 0]
            self.intercept_ = float(coefs[-1, 0])
        self.scale_fitted = True

    def sigma(self, states: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Fitted difficulty scale for the given states.

        Shape is (n_samples,) under scale_geometry='scalar' and
        (n_samples, *residual_trailing_shape) under 'per_cell'. Use
        difficulty_score() when a single scalar per sample is required.
        """
        if not self.scale_fitted:
            raise RuntimeError("Call fit_scale() before computing sigma.")
        states = self._validate_states(self._to_numpy(states, "states"))
        scaled = self.scaler.transform(states)
        log_sigma = scaled @ self.beta_ + self.intercept_
        sigma = np.maximum(np.exp(log_sigma), self.sigma_floor)
        if self.sigma_shape_:
            return sigma.reshape(states.shape[0], *self.sigma_shape_)
        return sigma

    def difficulty_score(self, states: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """One scalar difficulty per sample, whatever the scale geometry.

        This is the shared, method-agnostic score used for conditional-coverage
        binning: the runner fits bin edges from it on train states and reuses
        them for every conditioning mechanism, so flat CP, cluster SCCP, and
        state-scaled CP are scored on the SAME slices of state-space (see
        cissn/evaluation/metrics.py). Binning needs a 1-D score, so a per-cell
        sigma is reduced to its per-sample mean.
        """
        sigma = self.sigma(states)
        if sigma.ndim > 1:
            return sigma.reshape(sigma.shape[0], -1).mean(axis=1)
        return sigma

    def _sigma_for(self, states: np.ndarray, target_ndim: int) -> np.ndarray:
        """sigma(s) broadcast against a residual/forecast tensor of target_ndim.

        A scalar sigma gains trailing singleton axes. A per-cell sigma already
        carries the residual's trailing shape and is returned unchanged; it
        must match, because it was fitted against that exact cell layout.
        """
        sigma = self.sigma(states)
        if self.sigma_shape_:
            if sigma.ndim != target_ndim:
                raise ValueError(
                    f"per-cell sigma has trailing shape {self.sigma_shape_}, which is "
                    f"incompatible with a {target_ndim}-dimensional target; the scale was "
                    "fitted against a different horizon-feature layout."
                )
            return sigma
        return sigma.reshape(-1, *([1] * (target_ndim - 1)))

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

        normalized = residuals_np / self._sigma_for(states_np, residuals_np.ndim)

        q_level = split_conformal_q_level(normalized.shape[0], self.alpha)
        self.quantiles = self._compute_quantile(normalized, q_level)
        self.calibration_samples_ = int(states_np.shape[0])
        self.calibrated = True
        logger.info(
            "State-scaled CP calibrated: samples=%d, strategy=%s, geometry=%s",
            self.calibration_samples_, self.multivariate_strategy, self.scale_geometry,
        )

    def fit(self, states: Union[torch.Tensor, np.ndarray], residuals: Union[torch.Tensor, np.ndarray]) -> None:
        """Convenience API for IID tests; experiments use separate scale-fit and calibration data."""
        self.fit_scale(states, residuals)
        self.calibrate(states, residuals)

    def get_scale_stats(self) -> dict:
        """Return JSON-serializable fitted scale diagnostics."""
        if not self.calibrated:
            raise RuntimeError("Conformal predictor not calibrated. Call fit() first.")
        beta = np.asarray(self.beta_)
        stats = {
            "alpha": float(self.alpha),
            "multivariate_strategy": self.multivariate_strategy,
            "coverage_scope": self.coverage_scope,
            "scale_geometry": self.scale_geometry,
            "calibration_samples": int(self.calibration_samples_),
            "ridge": float(self.ridge),
            "sigma_floor": float(self.sigma_floor),
            "quantile_shape": list(np.asarray(self.quantiles).shape),
            "sigma_shape": list(self.sigma_shape_),
        }
        if self.sigma_shape_:
            # One regression per cell: summarise rather than dump H*C
            # coefficients into an artifact that has to stay readable.
            stats["beta_mean_per_state_dim"] = [float(b) for b in beta.mean(axis=1)]
            stats["beta_std_per_state_dim"] = [float(b) for b in beta.std(axis=1)]
            stats["intercept_mean"] = float(np.mean(self.intercept_))
        else:
            stats["beta"] = [float(b) for b in beta]
            stats["intercept"] = float(self.intercept_)
        return stats

    def predict(
        self,
        states: Union[torch.Tensor, np.ndarray],
        point_forecasts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate prediction intervals scaled by the fitted sigma(state).

        Args:
            states: (n_samples, state_dim)
            point_forecasts: Forecast tensor with leading sample axis.

        Returns:
            lower_bound, upper_bound with the same shape as point_forecasts.
        """
        states_np = self._check_predict_inputs(states, point_forecasts)
        sigma = self._sigma_for(states_np, point_forecasts.ndim)
        sigma_tensor = torch.as_tensor(sigma, device=point_forecasts.device, dtype=point_forecasts.dtype)

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
