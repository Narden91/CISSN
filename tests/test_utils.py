import unittest

import torch


from cissn.conformal import StateConditionalConformal


class TestConformalContracts(unittest.TestCase):
    def test_scalar_quantiles_broadcast_over_horizon_and_output(self):
        conformal = StateConditionalConformal(alpha=0.1, n_clusters=1, multivariate_strategy='max')
        states = torch.tensor([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=torch.float32)
        residuals = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        forecasts = torch.zeros(4, 3, 2, dtype=torch.float32)

        conformal.fit(states, residuals)
        lower, upper = conformal.predict(states, forecasts)
        widths = (upper - lower) / 2.0

        self.assertEqual(lower.shape, forecasts.shape)
        self.assertTrue(torch.allclose(widths[:, :, 0], widths[:, :, 1]))
        self.assertTrue(torch.allclose(widths[:, 0, :], widths[:, -1, :]))

    def test_per_feature_quantiles_broadcast_across_horizon(self):
        conformal = StateConditionalConformal(alpha=0.1, n_clusters=1, multivariate_strategy='per_feature')
        states = torch.tensor([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=torch.float32)
        residuals = torch.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            dtype=torch.float32,
        )
        forecasts = torch.zeros(4, 3, 2, dtype=torch.float32)

        conformal.fit(states, residuals)
        lower, upper = conformal.predict(states, forecasts)
        widths = (upper - lower) / 2.0

        self.assertEqual(lower.shape, forecasts.shape)
        self.assertTrue(torch.allclose(widths[:, 0, :], widths[:, 1, :]))
        self.assertFalse(torch.allclose(widths[:, :, 0], widths[:, :, 1]))

    def test_predict_rejects_incompatible_forecast_shape(self):
        conformal = StateConditionalConformal(alpha=0.1, n_clusters=1, multivariate_strategy='per_feature')
        states = torch.tensor([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=torch.float32)
        residuals = torch.ones(4, 2, 2, dtype=torch.float32)

        conformal.fit(states, residuals)

        with self.assertRaisesRegex(ValueError, 'incompatible'):
            conformal.predict(states, torch.zeros(4, 2, dtype=torch.float32))

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
        conformal.fit(states, residuals)
        first_clusters = conformal.kmeans.n_clusters

        conformal.fit(states[:8], torch.ones(8))

        self.assertLessEqual(conformal.kmeans.n_clusters, first_clusters)
        self.assertEqual(set(conformal.cluster_sizes_), set(range(conformal.kmeans.n_clusters)))
        self.assertEqual(conformal.acf_corrections_, {})


if __name__ == '__main__':
    unittest.main()
 
