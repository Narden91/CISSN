import json
import unittest
import warnings
from types import SimpleNamespace

import numpy as np
import torch


from cissn.conformal import StateConditionalConformal, StateScaledConformal


class TestConformalContracts(unittest.TestCase):
    def test_scalar_quantiles_broadcast_over_horizon_and_output(self):
        conformal = StateConditionalConformal(alpha=0.1, n_clusters=1, multivariate_strategy='max')
        # 11 samples >= ceil(1/alpha)=10 to avoid spurious small-cluster warning
        states = torch.linspace(0.0, 1.0, 11).unsqueeze(1).expand(-1, 2)
        residuals = torch.linspace(1.0, 4.0, 11)
        forecasts = torch.zeros(11, 3, 2, dtype=torch.float32)

        conformal.fit(states, residuals)
        lower, upper = conformal.predict(states, forecasts)
        widths = (upper - lower) / 2.0

        self.assertEqual(lower.shape, forecasts.shape)
        self.assertTrue(torch.allclose(widths[:, :, 0], widths[:, :, 1]))
        self.assertTrue(torch.allclose(widths[:, 0, :], widths[:, -1, :]))

    def test_per_feature_quantiles_broadcast_across_horizon(self):
        conformal = StateConditionalConformal(alpha=0.1, n_clusters=1, multivariate_strategy='per_feature')
        # 11 samples >= ceil(1/alpha)=10 to avoid spurious small-cluster warning
        t = torch.linspace(0.0, 1.0, 11)
        states = t.unsqueeze(1).expand(-1, 2)
        residuals = torch.stack([t + 1.0, (t + 1.0) * 10.0], dim=1)
        forecasts = torch.zeros(11, 3, 2, dtype=torch.float32)

        conformal.fit(states, residuals)
        lower, upper = conformal.predict(states, forecasts)
        widths = (upper - lower) / 2.0

        self.assertEqual(lower.shape, forecasts.shape)
        self.assertTrue(torch.allclose(widths[:, 0, :], widths[:, 1, :]))
        self.assertFalse(torch.allclose(widths[:, :, 0], widths[:, :, 1]))

    def test_predict_rejects_incompatible_forecast_shape(self):
        conformal = StateConditionalConformal(alpha=0.1, n_clusters=1, multivariate_strategy='per_feature')
        # 11 samples >= ceil(1/alpha)=10 to avoid spurious small-cluster warning
        states = torch.linspace(0.0, 1.0, 11).unsqueeze(1).expand(-1, 2)
        residuals = torch.ones(11, 2, 2, dtype=torch.float32)

        conformal.fit(states, residuals)

        with self.assertRaisesRegex(ValueError, 'incompatible'):
            conformal.predict(states, torch.zeros(11, 2, dtype=torch.float32))

    def test_requested_single_cluster_is_respected(self):
        conformal = StateConditionalConformal(alpha=0.1, n_clusters=1, multivariate_strategy='max')
        states = torch.randn(12, 2)
        residuals = torch.ones(12)

        conformal.fit(states, residuals)

        self.assertEqual(conformal.kmeans.n_clusters, 1)
        self.assertEqual(conformal.get_cluster_stats()["fitted_n_clusters"], 1)

    def test_no_cluster_falls_below_finite_sample_threshold(self):
        """Every fitted cluster must retain enough samples for a valid quantile.

        Budgeting clusters as n // min_samples only bounds the average size;
        K-Means on imbalanced states could still leave a cluster below 1/alpha,
        where the finite-sample quantile has no coverage guarantee. Reproduces
        exchange_rate at pred_len=720, which yields 39 calibration samples.
        """
        rng = np.random.default_rng(1)
        states = torch.tensor(
            np.vstack([
                rng.normal(0.0, 0.1, size=(9, 5)),
                rng.normal(5.0, 0.1, size=(10, 5)),
                rng.normal(-5.0, 2.0, size=(20, 5)),
            ]),
            dtype=torch.float32,
        )
        residuals = torch.tensor(np.abs(rng.normal(size=(39, 4, 2))), dtype=torch.float32)

        conformal = StateConditionalConformal(
            alpha=0.1, n_clusters=5, multivariate_strategy='max', random_state=42
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            conformal.fit(states, residuals)

        min_required = max(5, int(np.ceil(1.0 / 0.1)))
        smallest = min(conformal.cluster_sizes_.values())
        self.assertGreaterEqual(
            smallest, min_required,
            f"cluster of {smallest} samples is below the {min_required} needed for alpha=0.1",
        )

    def test_constant_residual_acf_is_zero_not_nan(self):
        rho = StateConditionalConformal._compute_acf1(torch.ones(12).numpy())

        self.assertEqual(rho, 0.0)

    def test_refit_resets_cluster_state(self):
        conformal = StateConditionalConformal(alpha=0.1, n_clusters=3, multivariate_strategy='max')
        states = torch.randn(30, 2)
        residuals = torch.arange(30, dtype=torch.float32).abs()
        # Suppress small-cluster warnings — KMeans can produce uneven splits on small fixtures
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            conformal.fit(states, residuals)
            first_clusters = conformal.kmeans.n_clusters
            conformal.fit(states[:8], torch.ones(8))

        self.assertLessEqual(conformal.kmeans.n_clusters, first_clusters)
        self.assertEqual(set(conformal.cluster_sizes_), set(range(conformal.kmeans.n_clusters)))
        self.assertEqual(conformal.cluster_fallbacks_, {0: "global_quantile"})

    def test_quantile_level_matches_textbook_and_empirical_coverage_holds(self):
        """q_level must equal ceil((n+1)(1-a))/(n+1); empirical coverage on i.i.d. draw >= 1-alpha."""
        import math, numpy as np

        alpha = 0.1
        n = 99  # chosen so (n+1)(1-alpha)=90 is exact integer
        rng = np.random.default_rng(0)

        conformal = StateConditionalConformal(alpha=alpha, n_clusters=1, multivariate_strategy='max')
        states = torch.as_tensor(rng.standard_normal((n, 2)), dtype=torch.float32)
        # i.i.d. U[0,1] residuals — known quantile is analytic
        residuals = torch.as_tensor(rng.uniform(0, 1, n), dtype=torch.float32)
        conformal.fit(states, residuals)

        # Verify q_level formula: ceil((n+1)(1-alpha))/(n+1)
        expected_q_level = math.ceil((n + 1) * (1 - alpha)) / (n + 1)
        cal_residuals_np = residuals.numpy()
        recomputed_q = float(np.quantile(cal_residuals_np, expected_q_level, method="higher"))
        stored_q = float(list(conformal.quantiles.values())[0])
        self.assertAlmostEqual(stored_q, recomputed_q, places=6,
                               msg="Stored quantile does not match textbook split-conformal level")

        # Empirical coverage on held-out i.i.d. draw must be >= 1-alpha
        n_test = 2000
        test_states = torch.as_tensor(rng.standard_normal((n_test, 2)), dtype=torch.float32)
        test_residuals = torch.as_tensor(rng.uniform(0, 1, n_test), dtype=torch.float32)
        forecasts = torch.zeros(n_test, 1, 1)
        targets = test_residuals.unsqueeze(1).unsqueeze(1)
        lower, upper = conformal.predict(test_states, forecasts)
        covered = ((lower <= targets) & (targets <= upper)).float().mean().item()
        self.assertGreaterEqual(covered, 1 - alpha,
                                msg=f"Empirical coverage {covered:.4f} < 1-alpha={1-alpha}")

    def test_flat_vs_sccp_parity(self):
        """FlatConformal and single-cluster SCCP must produce identical quantiles on same 1D residuals."""
        import numpy as np
        from cissn.baselines.flat_conformal import FlatConformal

        rng = np.random.default_rng(42)
        residuals = rng.uniform(0.1, 5.0, 50)
        states = rng.standard_normal((50, 2))

        flat = FlatConformal(alpha=0.1)
        flat.fit(residuals)

        sccp = StateConditionalConformal(alpha=0.1, n_clusters=1, multivariate_strategy="max")
        sccp.fit(states, residuals)

        stored_sccp_q = float(list(sccp.quantiles.values())[0])
        self.assertAlmostEqual(flat.quantile_, stored_sccp_q, places=6,
                               msg="FlatConformal and SCCP single-cluster quantiles differ")

    def test_compute_joint_picp(self):
        """compute_joint_picp requires all elements in sample window to be covered."""
        import numpy as np
        from cissn.evaluation.metrics import compute_joint_picp, compute_picp

        lower = np.zeros((4, 2, 2))
        upper = np.ones((4, 2, 2))
        y_true = np.zeros((4, 2, 2))
        # Set 1 element out of bounds in sample 0
        y_true[0, 1, 1] = 2.0

        # Element-wise marginal coverage: 15 out of 16 covered = 15/16 = 0.9375
        self.assertEqual(compute_picp(lower, upper, y_true), 15.0 / 16.0)
        # Joint sample coverage: 3 out of 4 samples fully covered = 0.75
        self.assertEqual(compute_joint_picp(lower, upper, y_true), 0.75)

    def test_partition_is_frozen_before_calibration_and_stride_is_recorded(self):
        rng = np.random.default_rng(9)
        reference_states = rng.normal(size=(40, 2))
        calibration_states = rng.normal(size=(20, 2))
        residuals = np.abs(rng.normal(size=(20, 3, 2)))
        conformal = StateConditionalConformal(
            alpha=0.1, n_clusters=2, multivariate_strategy="per_feature", calibration_stride=2
        )

        conformal.fit_partition(reference_states)
        centers_before = conformal.kmeans.cluster_centers_.copy()
        conformal.calibrate(calibration_states, residuals)

        self.assertTrue(np.array_equal(centers_before, conformal.kmeans.cluster_centers_))
        stats = conformal.get_cluster_stats()
        self.assertEqual(stats["calibration_stride"], 2)
        self.assertEqual(stats["calibration_samples"], 10)

    def test_state_scaled_conformal_also_accepts_and_records_calibration_stride(self):
        """StateScaledConformal must expose the identical calibration_stride
        constructor argument and calibrate()-time behaviour as
        StateConditionalConformal (see test_partition_is_frozen_before_
        calibration_and_stride_is_recorded above) -- otherwise a caller that
        wires --calibration_stride into one mechanism and not the other
        reintroduces a fitting-set-size asymmetry between them."""
        rng = np.random.default_rng(9)
        reference_states = rng.normal(size=(40, 2))
        calibration_states = rng.normal(size=(20, 2))
        residuals = np.abs(rng.normal(size=(20, 3, 2)))
        scaled = StateScaledConformal(alpha=0.1, multivariate_strategy="per_feature", calibration_stride=2)

        scaled.fit_scale(reference_states, np.abs(rng.normal(size=(40, 3, 2))))
        scaled.calibrate(calibration_states, residuals)

        stats = scaled.get_scale_stats()
        self.assertEqual(stats["calibration_stride"], 2)
        self.assertEqual(stats["calibration_samples"], 10)

    def test_dependence_diagnostic_uses_real_adjacent_origins(self):
        conformal = StateConditionalConformal(alpha=0.1, n_clusters=2)
        conformal.calibrated = True
        conformal.kmeans = SimpleNamespace(n_clusters=2)
        conformal.assign_clusters = lambda states: np.array([0, 1] * 5)
        states = np.zeros((10, 2))
        residuals = np.arange(10, dtype=float)

        result = conformal.diagnose_dependence(states, residuals, origin_indices=np.arange(10))

        self.assertEqual(result[0]["n_lag1_pairs"], 0)
        self.assertIsNone(result[0]["acf_lag1"])

    def test_flat_conformal_uses_horizon_feature_quantiles(self):
        from cissn.baselines.flat_conformal import FlatConformal

        residuals = np.stack([
            np.ones((2, 2)),
            np.full((2, 2), 2.0),
            np.full((2, 2), 3.0),
            np.full((2, 2), 4.0),
        ])
        flat = FlatConformal(alpha=0.2, multivariate_strategy="per_feature")
        flat.fit(residuals)
        lower, upper = flat.predict(torch.zeros(3, 2, 2))

        self.assertEqual(tuple(flat.quantile_.shape), (2, 2))
        self.assertEqual(tuple(lower.shape), (3, 2, 2))
        self.assertTrue(torch.allclose((upper - lower) / 2, torch.full((3, 2, 2), 4.0)))

    def test_dlinear_matches_replicate_endpoint_moving_average(self):
        from cissn.baselines import DLinear

        model = DLinear(input_dim=1, seq_len=5, pred_len=2, kernel_size=3)
        x = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
        with torch.no_grad():
            padded = torch.tensor([[[1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0]]])
            expected = torch.nn.functional.avg_pool1d(padded, kernel_size=3, stride=1)
            actual = model.decompose(torch.cat([x.permute(0, 2, 1)[:, :, :1], x.permute(0, 2, 1), x.permute(0, 2, 1)[:, :, -1:]], dim=2))
        self.assertTrue(torch.allclose(actual, expected))


class TestStateScaledConformal(unittest.TestCase):
    """StateScaledConformal: continuous state-conditional scale, calibrated on
    normalized residuals, as an alternative to K-Means cluster quantiles."""

    def test_fit_scale_before_calibrate_is_enforced(self):
        conformal = StateScaledConformal(alpha=0.1)
        states = torch.randn(20, 5)
        residuals = torch.abs(torch.randn(20, 3, 2))

        with self.assertRaises(RuntimeError):
            conformal.calibrate(states, residuals)

    def test_converges_to_flat_conformal_when_state_carries_no_signal(self):
        """With a state independent of residual scale, beta -> ~0 and the
        fitted quantile*sigma width should match FlatConformal closely --
        the scaled predictor must not invent structure that isn't there."""
        from cissn.baselines.flat_conformal import FlatConformal

        rng = np.random.default_rng(0)
        n_fit, n_cal = 300, 300
        states = rng.standard_normal((n_fit + n_cal, 5))
        residuals = np.abs(rng.standard_normal((n_fit + n_cal, 3, 2))) + 1.0

        scaled = StateScaledConformal(alpha=0.1, multivariate_strategy='per_feature')
        scaled.fit_scale(states[:n_fit], residuals[:n_fit])
        scaled.calibrate(states[n_fit:], residuals[n_fit:])

        flat = FlatConformal(alpha=0.1, multivariate_strategy='per_feature')
        flat.fit(residuals[n_fit:])

        forecasts = torch.zeros(50, 3, 2)
        test_states = states[n_fit:n_fit + 50]
        lower, upper = scaled.predict(test_states, forecasts)
        flat_lower, flat_upper = flat.predict(forecasts)

        scaled_width = float((upper - lower).mean())
        flat_width = float((flat_upper - flat_lower).mean())
        self.assertAlmostEqual(scaled_width, flat_width, delta=0.15 * flat_width)

    def test_recovers_state_dependent_scale(self):
        """When residual scale is truly a function of the state, the fitted
        sigma(s) must track it: predicted intervals should be wider for
        states associated with larger residuals."""
        rng = np.random.default_rng(1)
        n = 800
        states = rng.standard_normal((n, 5))
        # Residual scale grows with state[:, 0]; state[:, 0] in [-3, 3] roughly.
        true_sigma = np.exp(0.5 * states[:, 0])
        residuals = np.abs(rng.standard_normal((n, 2)) * true_sigma[:, None])

        n_fit = n // 2
        scaled = StateScaledConformal(alpha=0.1, multivariate_strategy='per_feature')
        scaled.fit_scale(states[:n_fit], residuals[:n_fit])
        scaled.calibrate(states[n_fit:], residuals[n_fit:])

        low_state = np.tile(np.array([-2.0, 0.0, 0.0, 0.0, 0.0]), (10, 1))
        high_state = np.tile(np.array([2.0, 0.0, 0.0, 0.0, 0.0]), (10, 1))
        forecasts = torch.zeros(10, 2)
        low_lower, low_upper = scaled.predict(low_state, forecasts)
        high_lower, high_upper = scaled.predict(high_state, forecasts)

        self.assertGreater(
            float((high_upper - high_lower).mean()),
            float((low_upper - low_lower).mean()),
            "sigma(s) did not track state-dependent residual scale",
        )

    def test_max_strategy_produces_scalar_quantile(self):
        conformal = StateScaledConformal(alpha=0.1, multivariate_strategy='max')
        rng = np.random.default_rng(2)
        states = rng.standard_normal((60, 3))
        residuals = np.abs(rng.standard_normal((60, 4, 2)))

        conformal.fit_scale(states[:30], residuals[:30])
        conformal.calibrate(states[30:], residuals[30:])

        self.assertEqual(conformal.quantile_shape, ())
        lower, upper = conformal.predict(states[30:40], torch.zeros(10, 4, 2))
        self.assertEqual(tuple(lower.shape), (10, 4, 2))

    def test_per_feature_strategy_preserves_horizon_feature_shape(self):
        conformal = StateScaledConformal(alpha=0.1, multivariate_strategy='per_feature')
        rng = np.random.default_rng(3)
        states = rng.standard_normal((60, 3))
        residuals = np.abs(rng.standard_normal((60, 4, 2)))

        conformal.fit_scale(states[:30], residuals[:30])
        conformal.calibrate(states[30:], residuals[30:])

        self.assertEqual(conformal.quantile_shape, (4, 2))
        lower, upper = conformal.predict(states[30:40], torch.zeros(10, 4, 2))
        self.assertEqual(tuple(lower.shape), (10, 4, 2))

    def test_rejects_negative_residuals(self):
        conformal = StateScaledConformal(alpha=0.1)
        states = torch.randn(20, 3)
        residuals = torch.randn(20, 2)  # can be negative

        with self.assertRaises(ValueError):
            conformal.fit_scale(states, residuals)


class TestStateScaledPerCellGeometry(unittest.TestCase):
    """scale_geometry='per_cell' fits one sigma per horizon-feature cell, so
    the state can reshape the quantile surface rather than only rescale its
    level -- the geometry where the conditioning signal was measured to be."""

    def test_sigma_carries_residual_trailing_shape(self):
        rng = np.random.default_rng(20)
        states = rng.standard_normal((80, 4))
        residuals = np.abs(rng.standard_normal((80, 6, 3)))

        scaled = StateScaledConformal(alpha=0.1, scale_geometry='per_cell')
        scaled.fit_scale(states[:40], residuals[:40])

        self.assertEqual(scaled.sigma_shape_, (6, 3))
        self.assertEqual(tuple(scaled.sigma(states[40:]).shape), (40, 6, 3))
        # difficulty_score() must stay 1-D for conditional-coverage binning.
        self.assertEqual(tuple(scaled.difficulty_score(states[40:]).shape), (40,))

    def test_scalar_geometry_remains_the_default(self):
        rng = np.random.default_rng(21)
        states = rng.standard_normal((60, 4))
        residuals = np.abs(rng.standard_normal((60, 5, 2)))

        scaled = StateScaledConformal(alpha=0.1)
        scaled.fit_scale(states, residuals)

        self.assertEqual(scaled.scale_geometry, 'scalar')
        self.assertEqual(scaled.sigma_shape_, ())
        self.assertEqual(tuple(scaled.sigma(states).shape), (60,))

    def test_max_strategy_forces_scalar_geometry(self):
        """'max' reduces residuals to one scalar per sample before scaling, so
        a per-cell scale has nothing to attach to."""
        scaled = StateScaledConformal(alpha=0.1, multivariate_strategy='max', scale_geometry='per_cell')
        self.assertEqual(scaled.scale_geometry, 'scalar')

    def test_per_cell_recovers_cell_specific_scale(self):
        """When residual scale depends on the state differently per cell, the
        per-cell geometry must track it and the scalar geometry must not."""
        rng = np.random.default_rng(22)
        n = 600
        states = rng.standard_normal((n, 3))
        # Cell (0,0) grows with state[:,0]; cell (1,0) shrinks with it. A
        # scalar sigma averages these to ~no signal; per-cell sees both.
        scale = np.stack([np.exp(0.6 * states[:, 0]), np.exp(-0.6 * states[:, 0])], axis=1)
        residuals = np.abs(rng.standard_normal((n, 2)) * scale)

        n_fit = n // 2
        scaled = StateScaledConformal(alpha=0.1, scale_geometry='per_cell')
        scaled.fit_scale(states[:n_fit], residuals[:n_fit])
        scaled.calibrate(states[n_fit:], residuals[n_fit:])

        high = np.tile(np.array([2.0, 0.0, 0.0]), (10, 1))
        sigma_high = scaled.sigma(high)
        self.assertGreater(
            float(sigma_high[:, 0].mean()), float(sigma_high[:, 1].mean()),
            "per-cell sigma did not separate cells with opposite state dependence",
        )

    def test_per_cell_predict_preserves_forecast_shape(self):
        rng = np.random.default_rng(23)
        states = rng.standard_normal((80, 3))
        residuals = np.abs(rng.standard_normal((80, 4, 2)))

        scaled = StateScaledConformal(alpha=0.1, scale_geometry='per_cell')
        scaled.fit_scale(states[:40], residuals[:40])
        scaled.calibrate(states[40:], residuals[40:])

        lower, upper = scaled.predict(states[40:50], torch.zeros(10, 4, 2))
        self.assertEqual(tuple(lower.shape), (10, 4, 2))
        self.assertTrue(bool((upper >= lower).all()))

    def test_per_cell_scale_stats_are_json_serializable(self):
        rng = np.random.default_rng(24)
        states = rng.standard_normal((80, 3))
        residuals = np.abs(rng.standard_normal((80, 4, 2)))

        scaled = StateScaledConformal(alpha=0.1, scale_geometry='per_cell')
        scaled.fit_scale(states[:40], residuals[:40])
        scaled.calibrate(states[40:], residuals[40:])

        stats = scaled.get_scale_stats()
        json.dumps(stats)  # must not raise
        self.assertEqual(stats["scale_geometry"], "per_cell")
        self.assertEqual(stats["sigma_shape"], [4, 2])
        # H*C coefficients are summarised, not dumped verbatim.
        self.assertEqual(len(stats["beta_mean_per_state_dim"]), 3)

    def test_rejects_unknown_scale_geometry(self):
        with self.assertRaises(ValueError):
            StateScaledConformal(alpha=0.1, scale_geometry='per_horizon')


class TestConditioningComparisonFairness(unittest.TestCase):
    """A conditioning mechanism may only be compared against a comparator
    calibrated on the SAME calibration window.

    A prior development diagnostic scored state-scaled CP after fitting sigma(s)
    on the first half of a window and calibrating on the second, but scored flat
    CP on the whole window -- giving flat CP twice the calibration data and the
    conditioned method an apparent advantage. These tests lock the invariant
    that a fair comparison shares the calibration split, and that a scalar
    per-sample sigma cannot exploit variation that lives on the per-cell axis.
    """

    def _winkler(self, lower, upper, y_true, alpha=0.1):
        from cissn.evaluation.metrics import winkler_score
        return winkler_score(lower, upper, y_true, alpha=alpha)

    def test_state_scaled_matches_flat_when_calibrated_on_same_window(self):
        """With an uninformative state, sharing the calibration window makes
        state-scaled CP and flat CP agree. Any large gap under this setup
        indicates the two were calibrated on different data, not that
        conditioning helped."""
        from cissn.baselines.flat_conformal import FlatConformal

        rng = np.random.default_rng(11)
        n_fit, n_cal, n_test = 200, 200, 100
        states = rng.standard_normal((n_fit + n_cal + n_test, 4))
        residuals = np.abs(rng.standard_normal((n_fit + n_cal + n_test, 3, 2))) + 1.0

        cal_slice = slice(n_fit, n_fit + n_cal)
        scaled = StateScaledConformal(alpha=0.1, multivariate_strategy='per_feature')
        scaled.fit_scale(states[:n_fit], residuals[:n_fit])
        scaled.calibrate(states[cal_slice], residuals[cal_slice])

        flat = FlatConformal(alpha=0.1, multivariate_strategy='per_feature')
        flat.fit(residuals[cal_slice])

        test_states = states[n_fit + n_cal:]
        forecasts = torch.zeros(n_test, 3, 2)
        y_true = np.zeros((n_test, 3, 2))

        lower, upper = scaled.predict(test_states, forecasts)
        flat_lower, flat_upper = flat.predict(forecasts)

        scaled_winkler = self._winkler(lower.numpy(), upper.numpy(), y_true)
        flat_winkler = self._winkler(flat_lower.numpy(), flat_upper.numpy(), y_true)
        self.assertAlmostEqual(scaled_winkler, flat_winkler, delta=0.15 * flat_winkler)

    def test_per_sample_sigma_cannot_exploit_per_cell_variation(self):
        """sigma(s) is one scalar per window. When residual scale varies only
        across horizon-feature cells and not across samples, the scalar has
        nothing to condition on and must not beat flat CP -- `per_feature`
        cell-wise quantiles already handle that axis."""
        from cissn.baselines.flat_conformal import FlatConformal

        rng = np.random.default_rng(12)
        n_fit, n_cal, n_test = 300, 300, 200
        n = n_fit + n_cal + n_test
        states = rng.standard_normal((n, 4))
        # Scale depends on the cell only, identically for every sample.
        cell_scale = np.array([[0.5, 3.0], [1.0, 2.0], [1.5, 1.0]])
        residuals = np.abs(rng.standard_normal((n, 3, 2))) * cell_scale

        cal_slice = slice(n_fit, n_fit + n_cal)
        scaled = StateScaledConformal(alpha=0.1, multivariate_strategy='per_feature')
        scaled.fit_scale(states[:n_fit], residuals[:n_fit])
        scaled.calibrate(states[cal_slice], residuals[cal_slice])

        flat = FlatConformal(alpha=0.1, multivariate_strategy='per_feature')
        flat.fit(residuals[cal_slice])

        test_states = states[n_fit + n_cal:]
        forecasts = torch.zeros(n_test, 3, 2)
        y_true = residuals[n_fit + n_cal:] * rng.choice([-1.0, 1.0], size=(n_test, 3, 2))

        lower, upper = scaled.predict(test_states, forecasts)
        flat_lower, flat_upper = flat.predict(forecasts)

        scaled_winkler = self._winkler(lower.numpy(), upper.numpy(), y_true)
        flat_winkler = self._winkler(flat_lower.numpy(), flat_upper.numpy(), y_true)
        self.assertGreater(
            scaled_winkler, 0.95 * flat_winkler,
            "state-scaled CP appeared to beat flat CP on purely per-cell variation, "
            "which a per-sample scalar sigma cannot explain",
        )


class TestConditionalCoverageMetric(unittest.TestCase):
    def test_fit_coverage_bin_edges_produces_equal_frequency_bins(self):
        from cissn.evaluation.metrics import fit_coverage_bin_edges

        rng = np.random.default_rng(4)
        scores = rng.uniform(0, 1, 500)
        edges = fit_coverage_bin_edges(scores, n_bins=5)

        self.assertEqual(len(edges), 4)
        self.assertTrue(np.all(np.diff(edges) > 0))

    def test_conditional_coverage_by_bin_detects_a_starved_slab(self):
        """A method with good marginal coverage but a starved bin must be
        caught by worst_slab_coverage even though marginal PICP looks fine."""
        from cissn.evaluation.metrics import fit_coverage_bin_edges, conditional_coverage_by_bin

        rng = np.random.default_rng(5)
        n = 400
        scores = rng.uniform(0, 1, n)
        edges = fit_coverage_bin_edges(scores, n_bins=4)

        y_true = np.zeros(n)
        lower = np.full(n, -1.0)
        upper = np.full(n, 1.0)
        # Starve the top score bin: half its samples fall outside [-1, 1].
        top_bin_mask = scores > np.quantile(scores, 0.75)
        y_true[top_bin_mask] = rng.choice([0.0, 5.0], size=top_bin_mask.sum())

        result = conditional_coverage_by_bin(lower, upper, y_true, scores, edges, alpha=0.1)

        overall_coverage = float(((y_true >= lower) & (y_true <= upper)).mean())
        self.assertGreater(overall_coverage, 0.85)
        self.assertLess(result["worst_slab_coverage"], overall_coverage)
        self.assertEqual(
            result["worst_prespecified_bin_coverage"], result["worst_slab_coverage"]
        )
        self.assertGreater(result["max_coverage_deviation"], 0.05)

    def test_same_bin_edges_produce_comparable_results_across_methods(self):
        """Two different interval methods scored on the SAME bin_edges must
        report per-bin sample counts that sum to the same total -- this is
        what makes the comparison fair rather than each method grading
        itself on its own partition."""
        from cissn.evaluation.metrics import fit_coverage_bin_edges, conditional_coverage_by_bin

        rng = np.random.default_rng(6)
        n = 200
        scores = rng.uniform(0, 1, n)
        edges = fit_coverage_bin_edges(scores, n_bins=5)
        y_true = rng.standard_normal(n)

        narrow = conditional_coverage_by_bin(
            np.full(n, -0.5), np.full(n, 0.5), y_true, scores, edges, alpha=0.1
        )
        wide = conditional_coverage_by_bin(
            np.full(n, -3.0), np.full(n, 3.0), y_true, scores, edges, alpha=0.1
        )

        total_narrow = sum(b["n_samples"] for b in narrow["bins"].values())
        total_wide = sum(b["n_samples"] for b in wide["bins"].values())
        self.assertEqual(total_narrow, total_wide)
        self.assertEqual(set(narrow["bins"].keys()), set(wide["bins"].keys()))
        self.assertGreaterEqual(wide["worst_slab_coverage"], narrow["worst_slab_coverage"])


class TestMeanScaledIntervalScore(unittest.TestCase):
    def test_multivariate_msis_uses_time_axis_per_feature(self):
        from cissn.evaluation.metrics import mean_scaled_interval_score

        y_train = np.column_stack((np.arange(12, dtype=float), np.arange(0, 120, 10, dtype=float)))
        y_true = np.array([[[4.0, 40.0]], [[6.0, 60.0]]])
        lower = y_true - np.array([[[1.0, 10.0]]])
        upper = y_true + np.array([[[1.0, 10.0]]])

        actual = mean_scaled_interval_score(lower, upper, y_true, y_train, seasonal_period=1, alpha=0.1)
        self.assertAlmostEqual(actual, 2.0, places=7)

    def test_msis_rejects_mismatched_feature_count(self):
        from cissn.evaluation.metrics import mean_scaled_interval_score

        y_train = np.arange(12, dtype=float).reshape(6, 2)
        lower = np.zeros((2, 1, 1))
        with self.assertRaisesRegex(ValueError, "Forecast feature count"):
            mean_scaled_interval_score(lower, lower + 1, lower, y_train, seasonal_period=1)


class TestPerOriginIntervalScores(unittest.TestCase):
    def test_preserves_chronological_origin_rows(self):
        from cissn.evaluation.metrics import per_origin_interval_scores

        y_true = np.array([[[0.0]], [[2.0]]])
        lower = np.array([[[-1.0]], [[-1.0]]])
        upper = np.array([[[1.0]], [[1.0]]])

        scores = per_origin_interval_scores(lower, upper, y_true, alpha=0.1)

        self.assertEqual(scores.shape, (2, 3))
        self.assertTrue(np.allclose(scores[:, 0], [1.0, 0.0]))
        self.assertTrue(np.allclose(scores[:, 1], [2.0, 2.0]))
        self.assertLess(scores[0, 2], scores[1, 2])


if __name__ == '__main__':
    unittest.main()
 
