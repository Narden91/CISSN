import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

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


if __name__ == '__main__':
    unittest.main()
