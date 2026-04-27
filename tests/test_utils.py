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


if __name__ == '__main__':
    unittest.main()
 
