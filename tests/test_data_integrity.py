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

    def test_near_constant_prediction_fails(self):
        rng = np.random.default_rng(0)
        trues = rng.normal(size=(500, 24, 7))
        preds = np.zeros_like(trues) + 1e-4  # near-constant, mimics white-noise collapse

        report = check_forecast_sanity(preds, trues)

        self.assertFalse(report["passed"])
        self.assertTrue(any("constant" in f for f in report["failures"]))

    def test_insufficient_mean_reference_improvement_is_reported(self):
        rng = np.random.default_rng(0)
        trues = rng.normal(size=(500, 24, 7))
        # Predictions with full variance but uncorrelated with targets: MSE ~= 2*var(trues).
        preds = rng.normal(size=(500, 24, 7))

        report = check_forecast_sanity(preds, trues)

        self.assertFalse(report["passed"])
        self.assertTrue(any("10% reduction" in f for f in report["failures"]))

    def test_reasonable_forecast_passes(self):
        rng = np.random.default_rng(0)
        trues = rng.normal(size=(500, 24, 7))
        preds = trues + rng.normal(scale=0.1, size=trues.shape)  # small residual error

        report = check_forecast_sanity(preds, trues)

        self.assertTrue(report["passed"])
        self.assertEqual(report["failures"], [])

    def test_failed_review_is_recorded_without_raising(self):
        trues = np.random.default_rng(0).normal(size=(100, 10))
        preds = np.zeros_like(trues)

        report = check_forecast_sanity(preds, trues)

        self.assertFalse(report["passed"])

    def test_history_flags_non_improving_validation_loss(self):
        rng = np.random.default_rng(0)
        trues = rng.normal(size=(500, 24, 7))
        preds = trues + rng.normal(scale=0.1, size=trues.shape)
        history = [
            {"epoch": 1, "train_loss": 1.0, "vali_loss": 1.0, "lr": 1e-3},
            {"epoch": 2, "train_loss": 1.0, "vali_loss": 1.0, "lr": 1e-6},
        ]

        report = check_forecast_sanity(preds, trues, history=history)

        self.assertTrue(report["passed"])  # only a warning, not a hard failure
        self.assertTrue(any("did not improve" in w for w in report["warnings"]))
        self.assertTrue(any("learning rate" in w for w in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
