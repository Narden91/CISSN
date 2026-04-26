#!/usr/bin/env python
"""
Multi-seed experiment runner for CISSN Paper 1.

Runs the full CISSN pipeline across multiple random seeds and aggregates
results with mean ± std for publication-quality tables.

Usage:
    python experiments/run_multiseed.py --data ETTh1 --pred_len 96 --seeds 42,123,456
    python experiments/run_multiseed.py --data ETTh1 --all_horizons --seeds 42,123,456
"""
import os
import sys
import argparse
import json
import time
import subprocess
import numpy as np
from pathlib import Path


HORIZONS = [24, 96, 192, 336, 720]


def run_single_experiment(args, seed, horizon):
    """Run a single benchmark experiment via subprocess."""
    setting = f'CISSN_{args.data}_sl{args.seq_len}_pl{horizon}_sd5_seed{seed}'
    cmd = [
        sys.executable, 'experiments/run_benchmark.py',
        '--data', args.data,
        '--root_path', args.root_path,
        '--data_path', args.data_path,
        '--features', args.features,
        '--seq_len', str(args.seq_len),
        '--label_len', str(args.label_len),
        '--pred_len', str(horizon),
        '--enc_in', str(args.enc_in),
        '--c_out', str(args.c_out),
        '--d_model', str(args.d_model),
        '--state_dim', '5',
        '--train_epochs', str(args.train_epochs),
        '--batch_size', str(args.batch_size),
        '--learning_rate', str(args.learning_rate),
        '--patience', str(args.patience),
        '--seed', str(seed),
        '--num_workers', str(args.num_workers),
        '--freq', args.freq,
        '--checkpoints', f'./checkpoints/{setting}/',
    ]
    print(f"\n{'—'*60}")
    print(f"Running: data={args.data} horizon={horizon} seed={seed}")
    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  FAILED with return code {result.returncode}")
        return None

    # Read results
    results_dir = Path('./results') / setting.replace(f'seed{seed}', f'sd{args.state_dim}')
    # The benchmark runner uses the setting string without seed in the path
    # Check all possible paths
    alt_setting = f'CISSN_{args.data}_{args.features}_sl{args.seq_len}_pl{horizon}_sd5'
    for rdir in [Path('./results') / setting, Path('./results') / alt_setting]:
        metrics_file = rdir / 'metrics.npy'
        conformal_file = rdir / 'conformal.npy'
        if metrics_file.exists():
            metrics = np.load(metrics_file)
            result_dict = {
                "data": args.data,
                "horizon": horizon,
                "seed": seed,
                "mae": float(metrics[0]),
                "mse": float(metrics[1]),
                "rmse": float(metrics[2]),
                "mape": float(metrics[3]),
            }
            if conformal_file.exists():
                conf = np.load(conformal_file)
                result_dict["coverage"] = float(conf[0])
                result_dict["mpiw"] = float(conf[1])
            return result_dict
    print(f"  Warning: results not found for setting '{setting}'")
    return None


def aggregate_results(all_results):
    """Aggregate results across seeds into mean ± std."""
    if not all_results:
        return {}
    aggregated = {}
    keys = ["mae", "mse", "rmse", "mape", "coverage", "mpiw"]
    for key in keys:
        values = [r[key] for r in all_results if r and key in r]
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    aggregated["n_seeds"] = len(all_results)
    return aggregated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CISSN Multi-Seed Experiment Runner')
    parser.add_argument('--data', type=str, required=True, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./data/ETT/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seeds', type=str, default='42,123,456',
                        help='Comma-separated random seeds')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Single horizon; use --all_horizons to override')
    parser.add_argument('--all_horizons', action='store_true',
                        help='Run all standard horizons (24,96,192,336,720)')
    parser.add_argument('--output', type=str, default='./results/multiseed_results.json')

    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    horizons = HORIZONS if args.all_horizons else [args.pred_len]

    t0 = time.time()
    all_data = {}

    for horizon in horizons:
        horizon_results = []
        for seed in seeds:
            r = run_single_experiment(args, seed, horizon)
            if r:
                horizon_results.append(r)
        if horizon_results:
            key = f"{args.data}_h{horizon}"
            all_data[key] = {
                "individual_runs": horizon_results,
                "aggregated": aggregate_results(horizon_results),
            }
            agg = all_data[key]["aggregated"]
            print(f"\n{key}: MSE={agg['mse']['mean']:.4f}±{agg['mse']['std']:.4f}, "
                  f"MAE={agg['mae']['mean']:.4f}±{agg['mae']['std']:.4f}, "
                  f"Coverage={agg['coverage']['mean']:.4f}±{agg['coverage']['std']:.4f}")

    elapsed = time.time() - t0
    print(f"\nMulti-seed run complete in {elapsed:.1f}s")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"Results saved to {args.output}")
