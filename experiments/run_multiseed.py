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
    parser.add_argument('--allow_partial', action='store_true',
                        help='Record failed seed/horizon cells as skipped instead of aborting the whole grid; '
                             'the aggregate is marked "complete": false and failed_seeds are listed')
    return parser.parse_known_args(args=argv)


def build_benchmark_run_argv(benchmark_argv: list[str], seed: int, horizon: int) -> list[str]:
    return [*benchmark_argv, '--seed', str(seed), '--pred_len', str(horizon)]


class ExperimentFailedError(RuntimeError):
    """Raised when a child run_benchmark.py subprocess fails or leaves no metrics.json."""


def run_single_experiment(benchmark_argv: list[str], seed: int, horizon: int):
    """Run a single benchmark experiment via subprocess.

    Raises ExperimentFailedError on a non-zero return code or missing
    metrics.json, so a failed cell cannot be silently dropped from the
    seed aggregate (see --allow_partial to opt back into that behavior).
    """
    child_argv = build_benchmark_run_argv(benchmark_argv, seed, horizon)
    effective_args = parse_benchmark_args(child_argv)
    setting = build_benchmark_setting_name(effective_args)
    cmd = [sys.executable, 'experiments/run_benchmark.py', *child_argv]
    print(f"\n{'—'*60}")
    print(f"Running: data={effective_args.data} horizon={horizon} seed={seed}")
    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise ExperimentFailedError(f"{setting}: subprocess exited with code {result.returncode}")

    rdir = Path(effective_args.results_dir) / setting
    metrics_json = rdir / 'metrics.json'
    if not metrics_json.exists():
        raise ExperimentFailedError(f"{setting}: run exited 0 but {metrics_json} was not written")

    metrics_payload = json.loads(metrics_json.read_text(encoding='utf-8'))
    point = metrics_payload.get("point", {})
    interval = metrics_payload.get("interval", {})
    # coverage_primary is the coverage the fitted conformal strategy actually
    # promises (marginal for per_feature, simultaneous coverage_joint for
    # max) -- see run_benchmark.py's Experiment.test for why plain "coverage"
    # is the wrong number to aggregate under a simultaneous strategy.
    result = {
        "data": effective_args.data,
        "horizon": horizon,
        "seed": seed,
        "setting": setting,
        "mae": _to_float(point.get("mae", np.nan)),
        "mse": _to_float(point.get("mse", np.nan)),
        "rmse": _to_float(point.get("rmse", np.nan)),
        "coverage": _to_float(interval.get("coverage_primary", np.nan)),
        "mpiw": _to_float(interval.get("mean_width", np.nan)),
        "winkler": _to_float(interval.get("winkler", np.nan)),
        "calibration_error": _to_float(interval.get("calibration_error", np.nan)),
        "msis": _to_float(interval.get("msis", np.nan)),
    }
    # Paired comparators: written by run_benchmark.py under fixed, mode-tagged
    # keys regardless of which mode was primary for this run (see
    # Experiment.test's conditioning_mode/cluster_result/scaled_result). Pull
    # Winkler and coverage from each so per-seed deltas can be computed
    # without re-reading raw arrays -- this is the paper's central paired
    # evidence and it must survive into the aggregate, not just metrics.json.
    for comparator_key, prefix in (
        ("interval_flat_cp", "flat"),
        ("interval_cluster_cp", "cluster"),
        ("interval_state_scaled", "scaled"),
    ):
        comparator = metrics_payload.get(comparator_key) or {}
        result[f"{prefix}_winkler"] = _to_float(comparator.get("winkler", np.nan))
        result[f"{prefix}_coverage"] = _to_float(comparator.get("coverage_primary", np.nan))
        result[f"{prefix}_mean_width"] = _to_float(comparator.get("mean_width", np.nan))
    # Signed deltas relative to the primary result reported in "interval":
    # negative means the primary conditioning mechanism won on Winkler.
    for prefix in ("flat", "cluster", "scaled"):
        result[f"winkler_delta_vs_{prefix}"] = result["winkler"] - result[f"{prefix}_winkler"]
    return result


def aggregate_results(all_results, n_seeds_requested: int):
    """Aggregate results across seeds into mean ± std."""
    aggregated = {}
    keys = [
        "mae", "mse", "rmse", "coverage", "mpiw", "winkler", "calibration_error", "msis",
        "flat_winkler", "flat_coverage", "flat_mean_width",
        "cluster_winkler", "cluster_coverage", "cluster_mean_width",
        "scaled_winkler", "scaled_coverage", "scaled_mean_width",
        "winkler_delta_vs_flat", "winkler_delta_vs_cluster", "winkler_delta_vs_scaled",
    ]
    for key in keys:
        values = [r[key] for r in all_results if key in r]
        if values:
            values = np.asarray(values, dtype=float)
            aggregated[key] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci95": float(1.96 * np.nanstd(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
            }
    # Sign-consistency counts: how many seeds the primary conditioning
    # mechanism actually beat each comparator on Winkler score. A mean delta
    # can look favorable while the sign flips seed to seed (as it does for
    # cluster-based SCCP vs flat CP on some cuts, see docs/methodology.md);
    # reporting the count alongside the mean keeps that visible.
    for prefix in ("flat", "cluster", "scaled"):
        deltas = [r[f"winkler_delta_vs_{prefix}"] for r in all_results if f"winkler_delta_vs_{prefix}" in r]
        finite = [d for d in deltas if np.isfinite(d)]
        if finite:
            aggregated[f"primary_beats_{prefix}_count"] = int(sum(1 for d in finite if d < 0))
            aggregated[f"primary_beats_{prefix}_of"] = len(finite)
    aggregated["n_seeds"] = len(all_results)
    aggregated["complete"] = len(all_results) == n_seeds_requested
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
    any_incomplete = False

    for horizon in horizons:
        horizon_results = []
        failed_seeds = []
        for seed in seeds:
            try:
                horizon_results.append(run_single_experiment(benchmark_argv, seed, horizon))
            except ExperimentFailedError as exc:
                if not wrapper_args.allow_partial:
                    raise
                print(f"  SKIPPED (--allow_partial): {exc}")
                failed_seeds.append(seed)

        if not horizon_results:
            any_incomplete = True
            continue

        key = f"{base_args.data}_h{horizon}"
        aggregated = aggregate_results(horizon_results, n_seeds_requested=len(seeds))
        aggregated["failed_seeds"] = failed_seeds
        any_incomplete = any_incomplete or not aggregated["complete"]
        all_data[key] = {"individual_runs": horizon_results, "aggregated": aggregated}
        print(f"\n{key}: MSE={aggregated['mse']['mean']:.4f}±{aggregated['mse']['std']:.4f}, "
              f"MAE={aggregated['mae']['mean']:.4f}±{aggregated['mae']['std']:.4f}, "
              f"Coverage={aggregated['coverage']['mean']:.4f}±{aggregated['coverage']['std']:.4f}"
              + (f"  [INCOMPLETE: {len(failed_seeds)} seed(s) failed]" if failed_seeds else ""))
        for prefix, label in (("flat", "flat CP"), ("cluster", "cluster SCCP"), ("scaled", "state-scaled CP")):
            count_key, of_key, delta_key = f"primary_beats_{prefix}_count", f"primary_beats_{prefix}_of", f"winkler_delta_vs_{prefix}"
            if count_key in aggregated:
                print(
                    f"  primary vs {label}: winkler delta {aggregated[delta_key]['mean']:+.4f}"
                    f"±{aggregated[delta_key]['std']:.4f}, wins {aggregated[count_key]}/{aggregated[of_key]} seeds"
                )

    elapsed = time.time() - t0
    print(f"\nMulti-seed run complete in {elapsed:.1f}s")
    output_path = Path(wrapper_args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2)
    write_raw_csv(wrapper_args.raw_csv, all_data)
    print(f"Results saved to {wrapper_args.output}")
    print(f"Raw rows saved to {wrapper_args.raw_csv}")

    if any_incomplete:
        print("\nWARNING: one or more horizon/seed cells were incomplete (see 'complete': false above).")
        sys.exit(1)


if __name__ == '__main__':
    main()
