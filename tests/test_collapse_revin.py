"""
Contracts for the forecast-collapse diagnostic and RevIN.

Both exist to address one failure: under MSE a model can lower loss by shrinking
its forecast toward the mean instead of tracking the signal. The diagnostic has
to make that visible, and RevIN has to remove the level-tracking burden that
causes it without leaking future information.
"""
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from cissn.evaluation.collapse import DispersionAccumulator, dispersion_summary
from cissn.models.revin import RevIN


class TestDispersionDiagnostic(unittest.TestCase):
    def test_matched_dispersion_scores_near_one(self):
        rng = np.random.default_rng(0)
        trues = rng.normal(size=(200, 24, 7))
        preds = trues + rng.normal(scale=0.01, size=trues.shape)

        summary = dispersion_summary(preds, trues)

        self.assertAlmostEqual(summary["variance_ratio"], 1.0, delta=0.05)
        self.assertGreater(summary["corr"], 0.99)

    def test_collapsed_forecast_scores_near_zero(self):
        rng = np.random.default_rng(0)
        trues = rng.normal(size=(200, 24, 7))
        preds = np.full_like(trues, trues.mean())

        summary = dispersion_summary(preds, trues)

        self.assertAlmostEqual(summary["variance_ratio"], 0.0, places=6)

    def test_shrunk_forecast_reports_the_shrink_factor(self):
        """A forecast scaled by a recovers variance ratio a**2."""
        rng = np.random.default_rng(0)
        trues = rng.normal(size=(500, 24, 7))
        preds = 0.5 * trues

        summary = dispersion_summary(preds, trues)

        self.assertAlmostEqual(summary["variance_ratio"], 0.25, delta=0.01)
        # Correlation is unaffected by scaling: direction is right, amplitude is not.
        self.assertGreater(summary["corr"], 0.99)

    def test_streaming_matches_one_shot(self):
        rng = np.random.default_rng(0)
        trues = rng.normal(size=(400, 12, 3))
        preds = 0.7 * trues + rng.normal(scale=0.1, size=trues.shape)

        accumulator = DispersionAccumulator()
        for start in range(0, 400, 64):
            accumulator.update(preds[start:start + 64], trues[start:start + 64])

        streamed = accumulator.summary()
        one_shot = dispersion_summary(preds, trues)
        for key in ("variance_ratio", "pred_std", "true_std", "corr"):
            self.assertAlmostEqual(streamed[key], one_shot[key], places=8)

    def test_empty_accumulator_reports_none(self):
        self.assertIsNone(DispersionAccumulator().summary()["variance_ratio"])

    def test_accepts_torch_tensors(self):
        trues = torch.randn(50, 8, 3)

        summary = dispersion_summary(trues, trues)

        self.assertAlmostEqual(summary["variance_ratio"], 1.0, places=5)

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            dispersion_summary(np.zeros((10, 3)), np.zeros((10, 4)))


class TestRevIN(unittest.TestCase):
    def test_denorm_inverts_norm(self):
        revin = RevIN(num_features=7, affine=False)
        x = torch.randn(8, 96, 7) * 3.0 + 5.0

        restored = revin(revin(x, "norm"), "denorm")

        torch.testing.assert_close(restored, x, rtol=1e-4, atol=1e-4)

    def test_denorm_inverts_norm_with_affine(self):
        revin = RevIN(num_features=7, affine=True)
        with torch.no_grad():
            revin.affine_weight.normal_(1.0, 0.1)
            revin.affine_bias.normal_(0.0, 0.1)
        x = torch.randn(8, 96, 7) * 3.0 + 5.0

        restored = revin(revin(x, "norm"), "denorm")

        torch.testing.assert_close(restored, x, rtol=1e-3, atol=1e-3)

    def test_normalised_window_is_standardised_per_instance(self):
        revin = RevIN(num_features=4, affine=False)
        # Each instance gets a different level and scale.
        x = torch.randn(6, 96, 4) * torch.tensor([1.0, 5.0, 10.0, 0.5]) + 20.0

        normalized = revin(x, "norm")

        torch.testing.assert_close(
            normalized.mean(dim=1), torch.zeros(6, 4), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            normalized.std(dim=1, unbiased=False), torch.ones(6, 4), atol=1e-3, rtol=0
        )

    def test_statistics_come_only_from_the_input_window(self):
        """RevIN must not see the target: no future information may leak."""
        revin = RevIN(num_features=3, affine=False)
        x = torch.randn(4, 96, 3)

        revin(x, "norm")
        mean_before = revin._mean.clone()

        # A totally different "future" must not change the stored statistics.
        forecast = torch.randn(4, 336, 3) * 100.0
        revin(forecast, "denorm")

        torch.testing.assert_close(revin._mean, mean_before)

    def test_denorm_broadcasts_over_a_longer_horizon(self):
        revin = RevIN(num_features=7, affine=False)
        revin(torch.randn(8, 96, 7), "norm")

        out = revin(torch.randn(8, 336, 7), "denorm")

        self.assertEqual(out.shape, (8, 336, 7))

    def test_constant_channel_does_not_produce_nan(self):
        revin = RevIN(num_features=2, affine=False)
        x = torch.ones(4, 96, 2)

        normalized = revin(x, "norm")

        self.assertTrue(torch.isfinite(normalized).all())

    def test_gradients_reach_affine_parameters(self):
        """A model sits between norm and denorm, so the affine terms do not cancel.

        Applying denorm directly to a norm output is an exact inverse and yields
        zero affine gradient by construction; the parameters only matter when
        something transforms the normalised window in between.
        """
        revin = RevIN(num_features=3, affine=True)
        model = torch.nn.Linear(3, 3)
        x = torch.randn(4, 96, 3)

        revin(model(revin(x, "norm")), "denorm").pow(2).mean().backward()

        self.assertIsNotNone(revin.affine_weight.grad)
        self.assertTrue(torch.any(revin.affine_weight.grad != 0))

    def test_denorm_refuses_a_mismatched_channel_count(self):
        """Broadcasting here would silently use the wrong channel's statistics.

        Under --features MS the forecast has one column while the statistics
        have num_features; a silent broadcast would rescale the target with
        feature 0's mean and scale.
        """
        revin = RevIN(num_features=7, affine=False)
        revin(torch.randn(4, 96, 7), "norm")

        with self.assertRaises(ValueError):
            revin(torch.randn(4, 336, 1), "denorm")

    def test_select_channels_uses_the_target_statistics(self):
        revin = RevIN(num_features=7, affine=False)
        # Give the last channel a distinctive level and scale.
        x = torch.randn(2, 96, 7)
        x[..., -1] = x[..., -1] * 50.0 + 100.0
        revin(x, "norm")

        restored = revin.select_channels(-1)(torch.zeros(2, 336, 1), "denorm")

        # Denormalising zero must return the target channel's own mean.
        expected = revin._mean[..., -1:].expand_as(restored)
        torch.testing.assert_close(restored, expected)

    def test_select_channels_round_trips_the_target_column(self):
        revin = RevIN(num_features=5, affine=True)
        x = torch.randn(3, 96, 5) * 7.0 + 2.0

        normalized = revin(x, "norm")
        target = normalized[..., -1:]
        restored = revin.select_channels(-1)(target, "denorm")

        torch.testing.assert_close(restored, x[..., -1:], rtol=1e-3, atol=1e-3)

    def test_select_channels_accepts_negative_and_positive_indices(self):
        revin = RevIN(num_features=5, affine=False)
        revin(torch.randn(3, 96, 5), "norm")

        torch.testing.assert_close(
            revin.select_channels(-1)._mean, revin.select_channels(4)._mean
        )

    def test_select_channels_rejects_out_of_range(self):
        revin = RevIN(num_features=5, affine=False)
        revin(torch.randn(3, 96, 5), "norm")

        with self.assertRaises(IndexError):
            revin.select_channels(5)

    def test_rejects_wrong_feature_count(self):
        revin = RevIN(num_features=7)

        with self.assertRaises(ValueError):
            revin(torch.randn(4, 96, 5), "norm")

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            RevIN(num_features=3)(torch.randn(2, 96, 3), "rescale")

    def test_denorm_before_norm_raises(self):
        with self.assertRaises(RuntimeError):
            RevIN(num_features=3)(torch.randn(2, 96, 3), "denorm")


class TestPairedFlatComparison(unittest.TestCase):
    """SCCP's contribution is state conditioning, so the flat-CP comparator must
    be fitted on the same residuals from the same model. Fitting it in a separate
    training run would confound the comparison with training variance."""

    def _experiment(self):
        from types import SimpleNamespace

        from experiments.run_benchmark import Experiment

        experiment = Experiment.__new__(Experiment)
        experiment.args = SimpleNamespace(conformal_alpha=0.1)
        return experiment

    def test_reports_nothing_when_flat_cp_is_absent(self):
        experiment = self._experiment()

        self.assertEqual(experiment._compare_against_flat_conformal(None, None), {})

    def test_paired_comparison_scores_the_same_forecasts(self):
        from cissn.baselines import FlatConformal

        experiment = self._experiment()
        rng = np.random.default_rng(0)
        cal_residuals = np.abs(rng.normal(size=(300, 8, 3)))
        flat = FlatConformal(alpha=0.1, multivariate_strategy="per_feature")
        flat.fit(cal_residuals)
        experiment.flat_conformal = flat

        preds = rng.normal(size=(120, 8, 3)).astype(np.float32)
        trues = preds + rng.normal(scale=0.3, size=preds.shape).astype(np.float32)

        report = experiment._compare_against_flat_conformal(preds, trues)

        for key in ("coverage", "mean_width", "winkler", "calibration_error"):
            self.assertIn(key, report)
        self.assertGreater(report["mean_width"], 0.0)
        self.assertEqual(report["coverage_scope"], "marginal")


if __name__ == "__main__":
    unittest.main()
