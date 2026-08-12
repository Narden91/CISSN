#!/usr/bin/env python
"""Generate publication-ready CSV tables from experiment artifacts.

This script scans metrics artifacts produced by:
- experiments/run_benchmark.py
- experiments/run_baseline.py
- experiments/run_multiseed.py (indirectly via metrics.json in run dirs)
- experiments/run_ablation.py (via JSON output files)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


SETTING_BASELINE_RE = re.compile(
    r"^BASELINE_(?P<model>[^_]+)_(?P<data>[^_]+)_[^_]+_sl\d+_pl(?P<pred>\d+)_seed(?P<seed>\d+)"
)
SETTING_CISSN_RE = re.compile(
    r"^CISSN_(?P<data>[^_]+)_[^_]+_sl\d+_pl(?P<pred>\d+).*_seed(?P<seed>\d+)"
)


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _parse_setting(setting: str) -> dict[str, Any]:
    base = {"model": None, "dataset": None, "pred_len": None, "seed": None, "family": "unknown"}
    m = SETTING_BASELINE_RE.match(setting)
    if m:
        base.update(
            {
                "model": m.group("model"),
                "dataset": m.group("data"),
                "pred_len": int(m.group("pred")),
                "seed": int(m.group("seed")),
                "family": "baseline",
            }
        )
        return base
    if setting.startswith("BASELINE_"):
        base["family"] = "baseline"
    m = SETTING_CISSN_RE.match(setting)
    if m:
        base.update(
            {
                "model": "cissn",
                "dataset": m.group("data"),
                "pred_len": int(m.group("pred")),
                "seed": int(m.group("seed")),
                "family": "cissn",
            }
        )
    return base


def collect_run_metrics(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in results_root.glob("**/metrics.json"):
        metrics = _safe_read_json(metrics_path)
        if not metrics:
            continue
        setting = metrics.get("setting") or metrics_path.parent.name
        parsed = _parse_setting(setting)
        config = _safe_read_json(metrics_path.parent / "config.json") or {}

        point = metrics.get("point", {})
        interval = metrics.get("interval", {})
        rows.append(
            {
                "artifact": str(metrics_path),
                "setting": setting,
                "family": parsed["family"],
                "model": _coalesce(metrics.get("model"), config.get("model"), parsed["model"]),
                "dataset": _coalesce(config.get("data"), parsed["dataset"]),
                "pred_len": _coalesce(config.get("pred_len"), parsed["pred_len"]),
                "seed": _coalesce(config.get("seed"), parsed["seed"]),
                "mse": point.get("mse"),
                "mae": point.get("mae"),
                "rmse": point.get("rmse"),
                "coverage": interval.get("coverage"),
                "coverage_primary": interval.get("coverage_primary"),
                "mpiw": interval.get("mean_width"),
                "winkler": interval.get("winkler"),
                "msis": interval.get("msis"),
                "calibration_error": interval.get("calibration_error"),
                "alpha": interval.get("alpha"),
                "coverage_scope": interval.get("coverage_scope"),
                "interval_origin": interval.get("interval_origin"),
                "sanity_passed": metrics.get("sanity_passed"),
                "protocol_present": (metrics_path.parent / "protocol.json").exists(),
            }
        )
    return pd.DataFrame(rows)


def collect_ablation_outputs(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in results_root.glob("**/*ablation*.json"):
        payload = _safe_read_json(path)
        if not isinstance(payload, dict):
            continue
        # run_ablation outputs a dict keyed by ablation name
        if not payload:
            continue
        if all(isinstance(v, dict) and "point" in v for v in payload.values()):
            for ablation, res in payload.items():
                point = res.get("point", {})
                interval = res.get("interval", {})
                rows.append(
                    {
                        "artifact": str(path),
                        "ablation": ablation,
                        "mse": point.get("mse"),
                        "mae": point.get("mae"),
                        "coverage_primary": interval.get("coverage_primary"),
                        "mpiw": interval.get("mean_width"),
                        "winkler": interval.get("winkler"),
                        "calibration_error": interval.get("calibration_error"),
                    }
                )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    available = [m for m in metrics if m in df.columns]
    if df.empty or not available:
        return pd.DataFrame()
    group_cols = ["family", "model", "dataset", "pred_len"]
    present = [c for c in group_cols if c in df.columns]
    out = df.groupby(present, dropna=False)[available].agg(["mean", "std"]).reset_index()
    out.columns = [
        "_".join(str(x) for x in c if x).rstrip("_") if isinstance(c, tuple) else c for c in out.columns
    ]
    return out.sort_values(present).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication CSV tables.")
    parser.add_argument("--results_root", type=Path, default=Path("./results"))
    parser.add_argument("--output_dir", type=Path, default=Path("./results/publication_tables"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_df = collect_run_metrics(args.results_root)
    ablation_df = collect_ablation_outputs(args.results_root)

    run_df.to_csv(args.output_dir / "all_runs_flat.csv", index=False)
    ablation_df.to_csv(args.output_dir / "ablations_flat.csv", index=False)

    eligible = run_df[
        (run_df["sanity_passed"] == True)
        & (run_df["protocol_present"] == True)
    ] if not run_df.empty else run_df
    point_tbl = summarize(eligible, ["mse", "mae", "rmse"])
    interval_eligible = eligible[
        eligible["coverage_primary"].notna()
        & ~eligible["interval_origin"].isin(["raw_uq"])
    ] if not eligible.empty else eligible
    interval_tbl = summarize(interval_eligible, ["coverage_primary", "mpiw", "winkler", "msis", "calibration_error"])

    point_tbl.to_csv(args.output_dir / "table_point_metrics.csv", index=False)
    interval_tbl.to_csv(args.output_dir / "table_interval_metrics.csv", index=False)

    if not ablation_df.empty:
        ablation_summary = (
            ablation_df.groupby("ablation", dropna=False)[["mse", "mae", "coverage_primary", "mpiw", "calibration_error"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        ablation_summary.columns = [
            "_".join(str(x) for x in c if x).rstrip("_") if isinstance(c, tuple) else c
            for c in ablation_summary.columns
        ]
        ablation_summary.to_csv(args.output_dir / "table_ablation_metrics.csv", index=False)

    summary = {
        "runs_detected": int(len(run_df)),
        "primary_eligible_runs": int(len(eligible)),
        "ablation_files_detected": int(len(ablation_df["artifact"].unique())) if not ablation_df.empty else 0,
        "output_dir": str(args.output_dir),
    }
    with (args.output_dir / "table_generation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

