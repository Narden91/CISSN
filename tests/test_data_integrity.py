import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cissn.data.dataset import Dataset_ETT_hour
from cissn.data.registry import DatasetIntegrityError, verify_dataset
from cissn.evaluation.sanity import check_forecast_sanity


class TestDatasetIntegrity(unittest.TestCase):
    """Regression coverage for the D1 failure mode: data/ETT/ETTh1.csv was
    replaced with i.i.d. noise (wrong row count, wrong date range, no
    autocorrelation) and every model trained on it correctly converged to
    the constant predictor -- no exception was ever raised. These tests
    pin the checks that now catch that class of failure before training."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.data_root = Path(self._tmpdir.name)

    def _write_csv(self, filename: str, df: pd.DataFrame) -> None:
        df.to_csv(self.data_root / filename, index=False)

    def test_rejects_wrong_row_count(self):
        dates = pd.date_range("2016-07-01", periods=100, freq="h")
        df = pd.DataFrame({"date": dates, "OT": np.random.randn(100)})
        self._write_csv("ETTh1.csv", df)

        report = verify_dataset("ETTh1", data_root=self.data_root, strict=False)

        self.assertFalse(report["passed"])
        self.assertTrue(any("row count" in f for f in report["failures"]))

    def test_rejects_white_noise_substituted_for_autocorrelated_series(self):
        """The exact D1 failure mode: right row count and date range, but the
        values are i.i.d. noise instead of a real (autocorrelated) series."""
        spec_rows = 17420
        dates = pd.date_range("2016-07-01", periods=spec_rows, freq="h")
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "date": dates,
            **{col: rng.normal(size=spec_rows) for col in ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]},
        })
        self._write_csv("ETTh1.csv", df)

        report = verify_dataset("ETTh1", data_root=self.data_root, strict=False)

        self.assertFalse(report["passed"])
        self.assertTrue(any("autocorrelation" in f for f in report["failures"]))

    def test_strict_mode_raises(self):
        dates = pd.date_range("2016-07-01", periods=5, freq="h")
        df = pd.DataFrame({"date": dates, "OT": np.zeros(5)})
        self._write_csv("ETTh1.csv", df)

        with self.assertRaises(DatasetIntegrityError):
            verify_dataset("ETTh1", data_root=self.data_root, strict=True)

    def test_missing_file_fails_with_clear_message(self):
        report = verify_dataset("ETTh1", data_root=self.data_root, strict=False)
        self.assertFalse(report["passed"])
        self.assertTrue(any("not found" in f for f in report["failures"]))

    def test_solar_has_no_integrity_fingerprint_but_does_not_crash(self):
        (self.data_root / "solar_AL.txt").write_text("0.1,0.2\n0.3,0.4\n", encoding="utf-8")
        report = verify_dataset("solar", data_root=self.data_root, strict=False)
        self.assertTrue(report["passed"])
        self.assertTrue(report["warnings"])


class TestSplitBorderCorrectness(unittest.TestCase):
    """Regression coverage for the D6 fix: cal is carved from the tail of the
    canonical train window so val/test stay literature-comparable, while
    train/cal/val/test remain disjoint and chronologically ordered."""

    def _make_hourly_dataset(self, flag: str, n_rows: int = 20000, seq_len: int = 96,
                              pred_len: int = 96, cal_fraction: float = 0.2):
        dates = pd.date_range("2016-07-01", periods=n_rows, freq="h")
        df = pd.DataFrame({"date": dates, "OT": np.arange(n_rows, dtype=float)})
        import unittest.mock as mock
        with mock.patch("cissn.data.dataset.pd.read_csv", return_value=df):
            return Dataset_ETT_hour(
                root_path=".", data_path="ignored.csv", flag=flag,
                size=[seq_len, 48, pred_len], features="S", target="OT",
                cal_fraction=cal_fraction,
            )

    def test_canonical_ett_window_counts_match_published_ltsf_protocol(self):
        # 12/4/4 months = 8640/2880/2880 hourly rows; cal (20% of train) is
        # carved from train's tail, so val/test stay at the canonical size.
        train = self._make_hourly_dataset("train")
        cal = self._make_hourly_dataset("cal")
        val = self._make_hourly_dataset("val")
        test = self._make_hourly_dataset("test")

        self.assertEqual(len(train.data_x), 8640 - int(round(0.2 * 8640)))
        self.assertEqual(len(cal.data_x), int(round(0.2 * 8640)) + 96)  # + seq_len lookback
        self.assertEqual(len(val.data_x), 2880 + 96)
        self.assertEqual(len(test.data_x), 2880 + 96)

    def test_splits_are_disjoint_and_chronologically_ordered(self):
        n_rows = 20000
        borders = {}
        for flag in ["train", "cal", "val", "test"]:
            ds = self._make_hourly_dataset(flag, n_rows=n_rows)
            border1s, border2s = ds._get_borders(pd.DataFrame({"date": pd.date_range("2016-07-01", periods=n_rows, freq="h")}))
            idx = ds._resolve_split_index(len(border1s))
            borders[flag] = (border1s[idx], border2s[idx])

        # Target rows (border2, excluding the seq_len lookback that's shared
        # input-only context) must be strictly increasing and non-overlapping.
        train_end = borders["train"][1]
        cal_start, cal_end = borders["cal"]
        val_start, val_end = borders["val"]
        test_start, test_end = borders["test"]

        self.assertLessEqual(train_end, cal_end)
        self.assertLessEqual(cal_end, val_end)
        self.assertLessEqual(val_end, test_end)
        # test must never reach back into cal's target region (only into val's).
        self.assertGreaterEqual(test_start, val_start)

    def test_cal_fraction_controls_train_cal_boundary(self):
        small_cal = self._make_hourly_dataset("train", cal_fraction=0.1)
        large_cal = self._make_hourly_dataset("train", cal_fraction=0.3)
        # A larger cal_fraction carves more off the train tail, so train shrinks.
        self.assertGreater(len(small_cal.data_x), len(large_cal.data_x))


class TestForecastReview(unittest.TestCase):
    """Regression coverage for the D2 finding: a model that converges to a
    near-constant output (the Bayes-optimal predictor for white-noise input)
    completes without raising. check_forecast_sanity is the check that
    would have caught it immediately from the saved pred/true arrays."""

    def test_near_constant_prediction_is_flagged_but_stays_valid(self):
        rng = np.random.default_rng(0)
        trues = rng.normal(size=(500, 24, 7))
        preds = np.zeros_like(trues) + 1e-4  # near-constant, mimics white-noise collapse

        report = check_forecast_sanity(preds, trues)

        # Degenerate output is a quality problem, not a structural one: the
        # arrays are finite and well-formed, so the run stays reportable.
        self.assertTrue(report["structural_passed"])
        self.assertTrue(any("constant" in w for w in report["warnings"]))

    def test_poor_forecast_remains_publication_visible(self):
        rng = np.random.default_rng(0)
        trues = rng.normal(size=(500, 24, 7))
        # Full variance but uncorrelated with targets: MSE ~= 2*var(trues).
        preds = rng.normal(size=(500, 24, 7))
        y_train = rng.normal(size=(2000, 7))

        report = check_forecast_sanity(preds, trues, y_train=y_train)

        self.assertTrue(report["passed"])
        self.assertTrue(report["structural_passed"])
        self.assertEqual(report["failures"], [])
        self.assertTrue(
            any("training-split reference" in w for w in report["warnings"])
        )

    def test_quality_references_come_from_train_split_only(self):
        """No test statistic may influence the quality reference."""
        rng = np.random.default_rng(0)
        trues = rng.normal(size=(500, 24, 7))
        preds = trues + rng.normal(scale=0.1, size=trues.shape)
        y_train = rng.normal(size=(2000, 7))

        report = check_forecast_sanity(
            preds, trues, y_train=y_train, seasonal_period=24
        )

        refs = report["quality"]["reference_mse"]
        self.assertIn("train_mean", refs)
        self.assertIn("seasonal_naive", refs)
        self.assertIn("persistence", refs)

    def test_references_are_evaluated_at_the_forecast_horizon(self):
        """A 1-step persistence error must not be used to score an h-step forecast.

        On a trending series the h-step lagged error grows with h; scoring a
        24-step forecast against a 1-step baseline would flag good models as
        weak purely because the baseline solved an easier problem.
        """
        t = np.arange(2000, dtype=float).reshape(-1, 1)
        y_train = np.concatenate([t * 0.01, t * 0.02], axis=1)
        trues = np.zeros((100, 24, 2))
        preds = np.zeros_like(trues)

        h1 = check_forecast_sanity(preds, trues, y_train=y_train, horizon=1)
        h24 = check_forecast_sanity(preds, trues, y_train=y_train, horizon=24)

        self.assertLess(
            h1["quality"]["reference_mse"]["persistence"],
            h24["quality"]["reference_mse"]["persistence"],
        )

    def test_horizon_defaults_to_the_trues_horizon_axis(self):
        t = np.arange(2000, dtype=float).reshape(-1, 1)
        y_train = np.concatenate([t * 0.01, t * 0.02], axis=1)
        trues = np.zeros((100, 24, 2))

        inferred = check_forecast_sanity(np.zeros_like(trues), trues, y_train=y_train)
        explicit = check_forecast_sanity(
            np.zeros_like(trues), trues, y_train=y_train, horizon=24
        )

        self.assertEqual(
            inferred["quality"]["reference_mse"], explicit["quality"]["reference_mse"]
        )

    def test_seasonal_naive_uses_whole_cycles_covering_the_horizon(self):
        rng = np.random.default_rng(0)
        y_train = rng.normal(size=(2000, 3))
        trues = np.zeros((50, 30, 3))

        report = check_forecast_sanity(
            np.zeros_like(trues), trues, y_train=y_train, seasonal_period=24, horizon=30
        )

        # horizon 30 with period 24 needs two full cycles (lag 48), not one.
        expected = float(np.mean((y_train[48:] - y_train[:-48]) ** 2))
        self.assertAlmostEqual(
            report["quality"]["reference_mse"]["seasonal_naive"], expected, places=10
        )

    def test_no_reference_reported_without_train_data(self):
        rng = np.random.default_rng(0)
        trues = rng.normal(size=(200, 24, 7))
        preds = rng.normal(size=(200, 24, 7))

        report = check_forecast_sanity(preds, trues)

        # Without a training split there is no admissible reference; the check
        # must report none rather than fall back to a test-derived baseline.
        self.assertEqual(report["quality"]["reference_mse"], {})

    def test_reasonable_forecast_passes(self):
        rng = np.random.default_rng(0)
        trues = rng.normal(size=(500, 24, 7))
        preds = trues + rng.normal(scale=0.1, size=trues.shape)  # small residual error

        report = check_forecast_sanity(preds, trues)

        self.assertTrue(report["passed"])
        self.assertEqual(report["failures"], [])

    def test_non_finite_predictions_fail_structurally(self):
        trues = np.random.default_rng(0).normal(size=(100, 10))
        preds = trues.copy()
        preds[0, 0] = np.nan

        report = check_forecast_sanity(preds, trues)

        self.assertFalse(report["passed"])
        self.assertFalse(report["structural_passed"])
        self.assertTrue(any("non-finite" in f for f in report["failures"]))

    def test_empty_and_mismatched_arrays_fail_structurally(self):
        empty = check_forecast_sanity(np.array([]), np.array([]))
        self.assertFalse(empty["structural_passed"])
        self.assertTrue(any("empty" in f for f in empty["failures"]))

        rng = np.random.default_rng(0)
        mismatched = check_forecast_sanity(
            rng.normal(size=(100, 10)), rng.normal(size=(100, 9))
        )
        self.assertFalse(mismatched["structural_passed"])
        self.assertTrue(any("shape mismatch" in f for f in mismatched["failures"]))

    def test_inverted_interval_bounds_fail_structurally(self):
        rng = np.random.default_rng(0)
        trues = rng.normal(size=(100, 10))
        preds = trues + 0.1
        lower, upper = preds + 1.0, preds - 1.0  # inverted

        report = check_forecast_sanity(preds, trues, lower=lower, upper=upper)

        self.assertFalse(report["structural_passed"])
        self.assertTrue(any("upper < lower" in f for f in report["failures"]))

    def test_all_nan_bounds_are_valid_point_only_run(self):
        rng = np.random.default_rng(0)
        trues = rng.normal(size=(100, 10))
        preds = trues + 0.1
        nan_bounds = np.full_like(preds, np.nan)

        report = check_forecast_sanity(
            preds, trues, lower=nan_bounds, upper=nan_bounds
        )

        self.assertTrue(report["structural_passed"])

    def test_history_flags_non_improving_validation_loss(self):
        rng = np.random.default_rng(0)
        trues = rng.normal(size=(500, 24, 7))
        preds = trues + rng.normal(scale=0.1, size=trues.shape)
        history = [
            {"epoch": 1, "train_loss": 1.0, "vali_loss": 1.0, "lr": 1e-3},
            {"epoch": 2, "train_loss": 1.0, "vali_loss": 1.0, "lr": 1e-6},
        ]

        report = check_forecast_sanity(preds, trues, history=history)

        self.assertTrue(report["passed"])  # advisory only, never a hard failure
        self.assertTrue(any("did not improve" in w for w in report["warnings"]))
        self.assertTrue(any("learning rate" in w for w in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
