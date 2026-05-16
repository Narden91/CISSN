import unittest
from types import SimpleNamespace

import numpy as np

from experiments.run_benchmark import parse_args as parse_benchmark_args
from experiments.run_baseline import compute_metrics
from experiments.run_multiseed import build_benchmark_run_argv, parse_multiseed_args


class TestExperimentRunners(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()