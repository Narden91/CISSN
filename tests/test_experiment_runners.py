import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.run_benchmark import (
    Experiment,
    build_setting_name,
    enforce_evidence_contract,
    parse_args as parse_benchmark_args,
    require_clean_source,
)
from experiments.run_baseline import compute_metrics, parse_ensemble_seeds
from experiments.run_multiseed import aggregate_results, build_benchmark_run_argv, parse_multiseed_args


class TestArchitectureSelection(unittest.TestCase):
    """The hybrid must be strictly opt-in and must never share a run directory
    with a legacy run, or the two would overwrite each other's checkpoints."""

    def test_legacy_is_the_default_architecture(self):
        args = parse_benchmark_args([])

        self.assertEqual(args.architecture, 'legacy')
        self.assertEqual(args.state_dynamics, 'legacy')
        self.assertFalse(args.state_revin)

    def test_legacy_setting_name_is_unchanged_by_the_hybrid_option(self):
        args = parse_benchmark_args([])

        self.assertNotIn('hybrid', build_setting_name(args))

    def test_every_architecture_variant_gets_a_distinct_run_directory(self):
        variants = [
            [],
            ['--architecture', 'hybrid'],
            ['--architecture', 'hybrid', '--state_dynamics', 'anchored'],
            ['--architecture', 'hybrid', '--state_dynamics', 'anchored', '--state_revin'],
        ]
        names = [build_setting_name(parse_benchmark_args(v)) for v in variants]

        self.assertEqual(len(set(names)), len(names))

    def test_rejects_unknown_architecture(self):
        with self.assertRaises(SystemExit):
            parse_benchmark_args(['--architecture', 'nonexistent'])


class TestConformalConditioningSelection(unittest.TestCase):
    """--conformal_conditioning must default to the pre-existing behavior and
    never collide a scale-mode run with a cluster-mode run on disk."""

    def test_cluster_is_the_default_conditioning_mode(self):
        args = parse_benchmark_args([])

        self.assertEqual(args.conformal_conditioning, 'cluster')

    def test_default_conditioning_setting_name_is_unchanged(self):
        """The default ('cluster') must produce a setting name with no
        conditioning-mode suffix, so every existing run directory on disk
        stays byte-identical after this flag was added."""
        args = parse_benchmark_args([])

        self.assertNotIn('cond', build_setting_name(args))

    def test_scale_conditioning_gets_a_distinct_run_directory(self):
        cluster_name = build_setting_name(parse_benchmark_args([]))
        scale_name = build_setting_name(parse_benchmark_args(['--conformal_conditioning', 'scale']))

        self.assertNotEqual(cluster_name, scale_name)

    def test_rejects_unknown_conditioning_mode(self):
        with self.assertRaises(SystemExit):
            parse_benchmark_args(['--conformal_conditioning', 'nonexistent'])


class TestExperimentRunners(unittest.TestCase):
    def test_confirmation_requires_sealed_run_safeguards(self):
        args = SimpleNamespace(
            evidence_role="confirmation",
            immutable_artifacts=False,
            strict_determinism=True,
            require_clean_git=True,
        )
        with self.assertRaisesRegex(ValueError, "immutable_artifacts"):
            enforce_evidence_contract(args)

        args.immutable_artifacts = True
        enforce_evidence_contract(args)

    def test_selection_is_refused_by_test_evaluation_runner(self):
        args = SimpleNamespace(evidence_role="selection")
        with self.assertRaisesRegex(ValueError, "validation-only"):
            enforce_evidence_contract(args)

    def test_calibration_stride_selects_one_shared_chronological_index(self):
        experiment = Experiment.__new__(Experiment)
        experiment.args = SimpleNamespace(calibration_stride=3)

        indices = experiment._shared_calibration_indices(10)

        self.assertTrue(np.array_equal(indices, np.array([0, 3, 6, 9])))
        self.assertEqual(len(indices), 4)

    def test_conditioning_and_quantile_calibration_use_disjoint_origins(self):
        selected = np.arange(0, 12, dtype=np.int64)

        conditioning, calibration = Experiment._split_calibration_indices(selected)

        self.assertTrue(np.array_equal(conditioning, np.arange(0, 6)))
        self.assertTrue(np.array_equal(calibration, np.arange(6, 12)))
        self.assertFalse(np.intersect1d(conditioning, calibration).size)

    def test_config_rejects_unknown_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.yaml"
            path.write_text("training:\n  learnng_rate: 0.001\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown config key"):
                parse_benchmark_args(["--config", str(path)])

    def test_config_rejects_invalid_typed_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.yaml"
            path.write_text("training:\n  train_epochs: not-an-int\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid value"):
                parse_benchmark_args(["--config", str(path)])

    def test_clean_source_fails_closed_when_git_is_unavailable(self):
        args = SimpleNamespace(require_clean_git=True)
        with patch("experiments.run_benchmark.environment_snapshot") as snapshot:
            snapshot.return_value = {"git_commit": None, "git_status": None, "git_dirty": None}
            with self.assertRaisesRegex(RuntimeError, "readable committed Git"):
                require_clean_source(args)

    def test_multiseed_wrapper_preserves_benchmark_args(self):
        wrapper_args, benchmark_argv = parse_multiseed_args(
            [
                '--seeds', '7,8',
                '--all_horizons',
                '--config', 'experiments/configs/etth1_smoke.yaml',
                '--dropout', '0.2',
                '--walk_forward',
                '--data', 'ETTh1',
            ]
        )

        self.assertEqual(wrapper_args.seeds, '7,8')
        self.assertTrue(wrapper_args.all_horizons)
        self.assertEqual(
            benchmark_argv,
            [
                '--config', 'experiments/configs/etth1_smoke.yaml',
                '--dropout', '0.2',
                '--walk_forward',
                '--data', 'ETTh1',
            ],
        )

    def test_multiseed_child_args_override_seed_and_horizon(self):
        child_argv = build_benchmark_run_argv(
            [
                '--config', 'experiments/configs/etth1_smoke.yaml',
                '--dropout', '0.2',
                '--walk_forward',
                '--seed', '1',
                '--pred_len', '96',
            ],
            seed=7,
            horizon=24,
        )

        args = parse_benchmark_args(child_argv)

        self.assertEqual(args.seed, 7)
        self.assertEqual(args.pred_len, 24)
        self.assertAlmostEqual(args.dropout, 0.2)
        self.assertTrue(args.walk_forward)
        self.assertEqual(args.lradj, 'cosine')

    def test_test_origin_stride_preserves_trailing_origin(self):
        args = parse_benchmark_args(["--test_origin_stride", "3"])

        self.assertEqual(args.test_origin_stride, 3)
        self.assertFalse(args.walk_forward)

    def test_retired_strict_sanity_argument_remains_accepted(self):
        args = parse_benchmark_args(['--strict_sanity'])

        self.assertFalse(hasattr(args, 'strict_sanity'))

    def test_baseline_interval_metrics_report_marginal_scope(self):
        args = SimpleNamespace(conformal_alpha=0.1)
        preds = np.array([[1.0], [2.0]], dtype=np.float32)
        trues = np.array([[1.5], [2.5]], dtype=np.float32)
        lower = np.array([[0.5], [1.5]], dtype=np.float32)
        upper = np.array([[1.5], [2.5]], dtype=np.float32)

        _point_metrics, interval_metrics = compute_metrics(args, preds, trues, lower=lower, upper=upper)

        self.assertEqual(interval_metrics['coverage_scope'], 'marginal')
        self.assertIsNotNone(interval_metrics['coverage'])

    def test_point_baseline_metrics_keep_empty_coverage_scope(self):
        args = SimpleNamespace(conformal_alpha=0.1)
        preds = np.array([[1.0], [2.0]], dtype=np.float32)
        trues = np.array([[1.5], [2.5]], dtype=np.float32)

        _point_metrics, interval_metrics = compute_metrics(args, preds, trues)

        self.assertIsNone(interval_metrics['coverage_scope'])
        self.assertIsNone(interval_metrics['coverage'])

    def test_default_ensemble_members_change_with_outer_seed(self):
        first = parse_ensemble_seeds(SimpleNamespace(ensemble_seeds='', ensemble_size=3, seed=42))
        second = parse_ensemble_seeds(SimpleNamespace(ensemble_seeds='', ensemble_size=3, seed=123))

        self.assertEqual(len(first), len(set(first)))
        self.assertNotEqual(first, second)

    def test_multiseed_aggregation_rejects_duplicate_seed(self):
        rows = [
            {"seed": 7, "mse": 1.0},
            {"seed": 7, "mse": 2.0},
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate seeds"):
            aggregate_results(rows, n_seeds_requested=2)


class TestConditioningCalibrationDataSource(unittest.TestCase):
    """Pins the actual data source `_calibrate_conformal` uses to fit each
    conditioning mechanism and the coverage bin edges, so a refactor cannot
    silently reintroduce the fitting-set asymmetry between `fit_partition`
    and `fit_scale`, or move bin-edge fitting onto a different split than
    documented, without a test failing.
    """

    @staticmethod
    def _build_experiment(n_cal_batches, batch_size, state_dim, horizon, n_features):
        from cissn.evaluation.metrics import fit_coverage_bin_edges

        experiment = Experiment.__new__(Experiment)
        experiment.args = SimpleNamespace(
            conformal_alpha=0.1,
            multivariate_strategy='per_feature',
            conformal_conditioning='cluster',
            n_clusters=2,
            seed=1,
            scale_geometry='scalar',
            calibration_stride=1,
            no_progress=True,
        )
        experiment._set_train_mode = lambda training: None

        rng = np.random.default_rng(0)
        raw_states = [
            torch.from_numpy(rng.standard_normal((batch_size, state_dim)).astype(np.float32))
            for _ in range(n_cal_batches)
        ]
        raw_outputs = [
            torch.from_numpy(rng.standard_normal((batch_size, horizon, n_features)).astype(np.float32))
            for _ in range(n_cal_batches)
        ]
        raw_targets = [
            torch.from_numpy(rng.standard_normal((batch_size, horizon, n_features)).astype(np.float32))
            for _ in range(n_cal_batches)
        ]
        calls = iter(zip(raw_states, raw_outputs, raw_targets))

        def fake_forward_and_slice(batch_x, batch_y, return_all_states=False):
            final_state, outputs, target = next(calls)
            return final_state, outputs, target

        experiment._forward_and_slice = fake_forward_and_slice
        cal_loader = [
            (object(), object(), object(), object()) for _ in range(n_cal_batches)
        ]
        return experiment, cal_loader, torch.cat(raw_states, dim=0), fit_coverage_bin_edges

    def test_fit_partition_and_fit_scale_see_the_same_conditioning_states(self):
        """The cluster partition and the sigma regression must be fit on
        identical data -- if a future change reintroduces separate fitting
        windows (e.g. reverting to train states for one mechanism), the
        state tensors captured by each predictor will diverge and this
        assertion will fail."""
        experiment, cal_loader, _all_states, _fit_edges = self._build_experiment(
            n_cal_batches=8, batch_size=4, state_dim=3, horizon=2, n_features=2,
        )
        experiment._build_conditioning_predictors()

        captured = {}
        original_fit_partition = experiment.conformal.fit_partition
        original_fit_scale = experiment.secondary_conformal.fit_scale

        def spy_fit_partition(states):
            captured['partition_states'] = states.clone()
            return original_fit_partition(states)

        def spy_fit_scale(states, residuals):
            captured['scale_states'] = states.clone()
            return original_fit_scale(states, residuals)

        experiment.conformal.fit_partition = spy_fit_partition
        experiment.secondary_conformal.fit_scale = spy_fit_scale

        experiment._calibrate_conformal(cal_loader, artifact_dir=None)

        self.assertTrue(torch.equal(captured['partition_states'], captured['scale_states']))

    def test_coverage_bin_edges_are_fit_on_the_conditioning_half_not_all_calibration_data(self):
        """Locks the documented contract: bin edges come from the same
        conditioning-half window used by fit_partition/fit_scale, not from
        the full calibration split or the quantile-calibration half."""
        experiment, cal_loader, all_states, fit_edges = self._build_experiment(
            n_cal_batches=8, batch_size=4, state_dim=3, horizon=2, n_features=2,
        )
        experiment._build_conditioning_predictors()
        experiment._calibrate_conformal(cal_loader, artifact_dir=None)

        n_total = all_states.shape[0]
        selected = experiment._shared_calibration_indices(n_total)
        conditioning_indices, _calibration_indices = experiment._split_calibration_indices(selected)
        expected_states = all_states[conditioning_indices]
        expected_edges = fit_edges(np.linalg.norm(expected_states.numpy(), axis=1), n_bins=5)

        np.testing.assert_array_equal(experiment._coverage_bin_edges, expected_edges)

    def test_saved_artifacts_show_equal_sized_disjoint_conditioning_and_calibration_states(self):
        """Artifact-level check, independent of internal call structure.

        The two preceding tests spy on `fit_partition`/`fit_scale` and read
        `experiment._coverage_bin_edges` directly -- both break (loudly, via
        AttributeError, or silently, via an untriggered spy) if a refactor
        moves conditioning-fit logic to a different method or object. This
        test instead reads back the `.npy` artifacts `_calibrate_conformal`
        is contracted (CLAUDE.md, "Each final result must contain...") to
        write for every run, so it survives internal restructuring as long
        as the on-disk artifact contract holds: conditioning_states.npy and
        calibration_states.npy must be equal in length and index-disjoint,
        which is what "both mechanisms share one fitting window, distinct
        from the quantile-calibration window" requires on disk.
        """
        experiment, cal_loader, all_states, _fit_edges = self._build_experiment(
            n_cal_batches=8, batch_size=4, state_dim=3, horizon=2, n_features=2,
        )
        experiment._build_conditioning_predictors()

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment._calibrate_conformal(cal_loader, artifact_dir=Path(tmpdir))

            conditioning_states = np.load(Path(tmpdir) / "conditioning_states.npy")
            calibration_states = np.load(Path(tmpdir) / "calibration_states.npy")
            conditioning_indices = np.load(Path(tmpdir) / "conditioning_indices.npy")
            calibration_indices = np.load(Path(tmpdir) / "calibration_indices.npy")

        self.assertEqual(conditioning_states.shape[0], calibration_states.shape[0])
        self.assertFalse(np.intersect1d(conditioning_indices, calibration_indices).size)
        self.assertEqual(
            set(conditioning_indices.tolist()) | set(calibration_indices.tolist()),
            set(range(all_states.shape[0])),
        )


if __name__ == '__main__':
    unittest.main()
