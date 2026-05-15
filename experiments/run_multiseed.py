#!/usr/bin/env python
"""
Multi-seed experiment runner for CISSN Paper 1.

Runs the full CISSN pipeline across multiple random seeds and aggregates
results with mean ± std for publication-quality tables.

All non-wrapper CLI arguments are forwarded directly to run_benchmark.py.

Usage:
    python experiments/run_multiseed.py --data ETTh1 --pred_len 96 --seeds 42,123,456
    python experiments/run_multiseed.py --data ETTh1 --all_horizons --seeds 42,123,456
"""
import argparse
import csv
import json
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

from cissn.data.registry import get_dataset_spec

try:
    from .run_benchmark import build_setting_name as build_benchmark_setting_name, parse_args as parse_benchmark_args
except ImportError:
    from run_benchmark import build_setting_name as build_benchmark_setting_name, parse_args as parse_benchmark_args


def _to_float(value):
    return float(value) if value is not None else float("nan")


def parse_multiseed_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description='CISSN Multi-Seed Experiment Runner',
        epilog='Any additional arguments are forwarded to experiments/run_benchmark.py.',
    )
    parser.add_argument('--seeds', type=str, default='42,123,456', help='Comma-separated random seeds')
    parser.add_argument('--all_horizons', action='store_true', help='Run all standard horizons for the chosen dataset')
    parser.add_argument('--output', type=str, default='./results/multiseed_results.json')
    parser.add_argument('--raw_csv', type=str, default='./results/multiseed_raw.csv')
    return parser.parse_known_args(args=argv)


def build_benchmark_run_argv(benchmark_argv: list[str], seed: int, horizon: int) -> list[str]:
    return [*benchmark_argv, '--seed', str(seed), '--pred_len', str(horizon)]


def run_single_experiment(benchmark_argv: list[str], seed: int, horizon: int):
    """Run a single benchmark experiment via subprocess."""
    child_argv = build_benchmark_run_argv(benchmark_argv, seed, horizon)
    effective_args = parse_benchmark_args(child_argv)
    setting = build_benchmark_setting_name(effective_args)
    cmd = [sys.executable, 'experiments/run_benchmark.py', *child_argv]
    print(f"\n{'—'*60}")
    print(f"Running: data={effective_args.data} horizon={horizon} seed={seed}")
    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  FAILED with return code {result.returncode}")
        return None

    rdir = Path(effective_args.results_dir) / setting
    metrics_json = rdir / 'metrics.json'
    if metrics_json.exists():
        metrics_payload = json.loads(metrics_json.read_text(encoding='utf-8'))
        point = metrics_payload.get("point", {})
        interval = metrics_payload.get("interval", {})
        return {
            "data": effective_args.data,
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


def main(argv: list[str] | None = None) -> None:
    wrapper_args, benchmark_argv = parse_multiseed_args(argv)
    base_args = parse_benchmark_args(benchmark_argv)

    spec = get_dataset_spec(base_args.data)
    seeds = [int(s.strip()) for s in wrapper_args.seeds.split(',') if s.strip()]
    horizons = spec["horizons"] if wrapper_args.all_horizons else [base_args.pred_len]

    t0 = time.time()
    all_data = {}

    for horizon in horizons:
        horizon_results = []
        for seed in seeds:
            r = run_single_experiment(benchmark_argv, seed, horizon)
            if r:
                horizon_results.append(r)
        if horizon_results:
            key = f"{base_args.data}_h{horizon}"
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
    output_path = Path(wrapper_args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2)
    write_raw_csv(wrapper_args.raw_csv, all_data)
    print(f"Results saved to {wrapper_args.output}")
    print(f"Raw rows saved to {wrapper_args.raw_csv}")


if __name__ == '__main__':
    main()
