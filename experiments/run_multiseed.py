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
import csv
import json
import time
import subprocess
import numpy as np
from pathlib import Path

from cissn.data.registry import get_dataset_spec, supported_datasets


HORIZONS = [24, 96, 192, 336, 720]


def _format_float_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def build_setting_name(args, seed, horizon):
    return (
        f"CISSN_{args.data}_{args.features}"
        f"_sl{args.seq_len}_pl{horizon}_sd{args.state_dim}_dm{args.d_model}"
        f"_lc{_format_float_token(args.lambda_cov)}_lt{_format_float_token(args.lambda_temp)}"
        f"_a{_format_float_token(args.conformal_alpha)}_{args.multivariate_strategy}"
        f"_seed{seed}"
    )


def _to_float(value):
    return float(value) if value is not None else float("nan")


def run_single_experiment(args, seed, horizon):
    """Run a single benchmark experiment via subprocess."""
    setting = build_setting_name(args, seed, horizon)
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
        '--state_dim', str(args.state_dim),
        '--train_epochs', str(args.train_epochs),
        '--batch_size', str(args.batch_size),
        '--learning_rate', str(args.learning_rate),
        '--patience', str(args.patience),
        '--seed', str(seed),
        '--num_workers', str(args.num_workers),
        '--freq', args.freq,
        '--lambda_cov', str(args.lambda_cov),
        '--lambda_temp', str(args.lambda_temp),
        '--conformal_alpha', str(args.conformal_alpha),
        '--n_clusters', str(args.n_clusters),
        '--multivariate_strategy', args.multivariate_strategy,
        '--checkpoints', args.checkpoints,
        '--results_dir', args.results_dir,
    ]
    print(f"\n{'—'*60}")
    print(f"Running: data={args.data} horizon={horizon} seed={seed}")
    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  FAILED with return code {result.returncode}")
        return None

    rdir = Path(args.results_dir) / setting
    metrics_json = rdir / 'metrics.json'
    if metrics_json.exists():
        metrics_payload = json.loads(metrics_json.read_text(encoding='utf-8'))
        point = metrics_payload.get("point", {})
        interval = metrics_payload.get("interval", {})
        return {
            "data": args.data,
            "horizon": horizon,
            "seed": seed,
            "setting": setting,
            "mae": _to_float(point.get("mae", np.nan)),
            "mse": _to_float(point.get("mse", np.nan)),
            "rmse": _to_float(point.get("rmse", np.nan)),
            "mape": _to_float(point.get("mape", np.nan)),
            "coverage": _to_float(interval.get("coverage", np.nan)),
            "mpiw": _to_float(interval.get("mean_width", np.nan)),
            "winkler": _to_float(interval.get("winkler", np.nan)),
            "calibration_error": _to_float(interval.get("calibration_error", np.nan)),
        }
    print(f"  Warning: results not found for setting '{setting}'")
    return None


def aggregate_results(all_results):
    """Aggregate results across seeds into mean ± std."""
    if not all_results:
        return {}
    aggregated = {}
    keys = ["mae", "mse", "rmse", "mape", "coverage", "mpiw", "winkler", "calibration_error"]
    for key in keys:
        values = [r[key] for r in all_results if r and key in r]
        if values:
            values = np.asarray(values, dtype=float)
            aggregated[key] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci95": float(1.96 * np.nanstd(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
            }
    aggregated["n_seeds"] = len(all_results)
    return aggregated


def write_raw_csv(path: str, results_by_horizon: dict) -> None:
    rows = []
    for entry in results_by_horizon.values():
        rows.extend(entry["individual_runs"])
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CISSN Multi-Seed Experiment Runner')
    parser.add_argument('--data', type=str, required=True, choices=supported_datasets(), default='ETTh1')
    parser.add_argument('--root_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--freq', type=str, default=None)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--enc_in', type=int, default=None)
    parser.add_argument('--c_out', type=int, default=None)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--state_dim', type=int, default=5)
    parser.add_argument('--lambda_cov', type=float, default=1.0)
    parser.add_argument('--lambda_temp', type=float, default=0.5)
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
    parser.add_argument('--conformal_alpha', type=float, default=0.1)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--multivariate_strategy', type=str, default='per_feature')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--results_dir', type=str, default='./results/')
    parser.add_argument('--output', type=str, default='./results/multiseed_results.json')
    parser.add_argument('--raw_csv', type=str, default='./results/multiseed_raw.csv')

    args = parser.parse_args()
    spec = get_dataset_spec(args.data)
    for key in ("root_path", "data_path", "freq", "target", "enc_in", "c_out"):
        if getattr(args, key) is None:
            setattr(args, key, spec[key])
    if args.features == "MS" and args.c_out == spec["c_out"]:
        args.c_out = 1
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    horizons = spec["horizons"] if args.all_horizons else [args.pred_len]

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
    write_raw_csv(args.raw_csv, all_data)
    print(f"Results saved to {args.output}")
    print(f"Raw rows saved to {args.raw_csv}")
