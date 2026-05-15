import unittest

from experiments.run_benchmark import parse_args as parse_benchmark_args
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


if __name__ == '__main__':
    unittest.main()