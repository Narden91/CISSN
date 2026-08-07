import unittest
import warnings

import torch


from cissn.conformal import StateConditionalConformal


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
        self.assertEqual(conformal.acf_corrections_, {})

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


if __name__ == '__main__':
    unittest.main()
 
