#!/usr/bin/env python
"""Paired Winkler inference: primary conditioning mechanism vs flat CP.

Reads the per-origin scores that every CISSN run already saves, so it needs no
retraining. For each dataset x horizon cell it averages the per-origin Winkler
difference (primary minus flat CP, so negative favours the primary mechanism)
over seeds, then resamples consecutive origins in blocks to get a confidence
interval. Across cells it reports a Wilcoxon signed-rank test.

Every analysis choice is a required argument with no default. Write them into
the prespecification file first, then pass them here unchanged. The cells share
datasets, and horizons within one dataset share data, so the number of
independent units is closer to the number of datasets than to the number of
cells; the output reports both counts.

Evidence role: confirmatory only when run on `--evidence_role confirmation`
artifacts after the prespecification is written. On any other artifacts the
output is exploratory.

Usage:
    uv run python scripts/paired_interval_inference.py \\
        --results_root results/publication --datasets ETTh2 weather exchange_rate \\
        --block_multiplier 1 --n_boot 5000 --seed 0 \\
        --output results/publication/paired_interval_inference.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from cissn.evaluation.inference import moving_block_bootstrap_ci

WINKLER_COLUMN = 2
# Which calibrator is compared with flat CP: the run's primary one, or state-scaled CP.
METHOD_FILES = {
    "primary": "per_origin_interval_metrics.npy",
    "state_scaled": "per_origin_interval_state_scaled_metrics.npy",
}


def _load_cell_runs(results_root: Path, datasets: set[str], method: str = "primary") -> dict:
    cells: dict = defaultdict(list)
    for metrics_path in sorted(results_root.glob("CISSN_*/metrics.json")):
        run_dir = metrics_path.parent
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if config["data"] not in datasets:
            continue
        # The prespecified arm is legacy CISSN under RevIN. Hybrid runs share the
        # `CISSN_` prefix and the confirmation role, so without this filter they
        # would be averaged into the same cell.
        if config.get("architecture") != "legacy" or not config.get("revin"):
            continue
        if config.get("evidence_role") != "confirmation":
            raise ValueError(f"{run_dir.name}: evidence_role is not 'confirmation'.")
        if not metrics.get("structural_passed", False):
            print(f"skip {run_dir.name}: structural_passed is false")
            continue
        cells[(config["data"], int(config["pred_len"]))].append({
            "seed": int(config["seed"]),
            "stride": int(config.get("test_origin_stride", 1)),
            "conditioning": config["conformal_conditioning"],
            "scale_geometry": config.get("scale_geometry", "scalar"),
            "primary": np.load(run_dir / METHOD_FILES[method]),
            "flat": np.load(run_dir / "per_origin_interval_flat_cp_metrics.npy"),
            "origins": np.load(run_dir / "test_origin_indices.npy"),
        })
    return cells


def _cell_delta(runs: list[dict]) -> np.ndarray:
    # Seeds share the test rows. If the origins differ, the rows are not paired
    # and averaging them would be meaningless, so stop instead of aligning.
    if len({run["conditioning"] for run in runs}) != 1:
        raise ValueError("A cell mixes conditioning modes.")
    if len({run["scale_geometry"] for run in runs}) != 1:
        raise ValueError("A cell mixes scale geometries.")
    seeds = [run["seed"] for run in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"A cell holds two runs for one seed: {sorted(seeds)}.")
    first = runs[0]["origins"]
    if any(not np.array_equal(first, run["origins"]) for run in runs[1:]):
        raise ValueError("Seeds of one cell have different test origins.")
    if len({run["stride"] for run in runs}) != 1:
        raise ValueError("Seeds of one cell use different test-origin strides.")
    flat = np.mean([run["flat"][:, WINKLER_COLUMN] for run in runs], axis=0)
    primary = np.mean([run["primary"][:, WINKLER_COLUMN] for run in runs], axis=0)
    return primary - flat, float(flat.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--block_multiplier", type=float, required=True,
                        help="Block length = ceil(multiplier * pred_len). Use at least 1.")
    parser.add_argument("--n_boot", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--method", choices=sorted(METHOD_FILES), default="primary",
                        help="calibrator compared with flat CP: the run's primary one, or state-scaled CP")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.block_multiplier < 1.0:
        raise ValueError("block_multiplier below 1 gives intervals that are too narrow.")

    cells = _load_cell_runs(args.results_root, set(args.datasets), args.method)
    if not cells:
        raise RuntimeError("No matching confirmation runs found.")

    rows = []
    for (data, horizon), runs in sorted(cells.items()):
        # Origins are `stride` steps apart, so a block must span horizon / stride of them.
        stride = runs[0]["stride"]
        block_len = int(np.ceil(args.block_multiplier * horizon / stride))
        delta, flat_mean = _cell_delta(runs)
        result = moving_block_bootstrap_ci(delta, block_len, args.n_boot, args.seed, args.confidence)
        rows.append({
            "data": data,
            "horizon": horizon,
            "stride": stride,
            "seeds": sorted(run["seed"] for run in runs),
            "conditioning": runs[0]["conditioning"],
            "method": args.method,
            "scale_geometry": runs[0]["scale_geometry"],
            "flat_winkler_mean": flat_mean,
            "relative_mean": result["mean"] / flat_mean,
            "relative_ci_low": result["ci_low"] / flat_mean,
            "relative_ci_high": result["ci_high"] / flat_mean,
            **result,
        })

    means = np.array([row["mean"] for row in rows])
    summary: dict = {
        "n_cells": len(rows),
        "n_datasets": len({row["data"] for row in rows}),
        "mean_of_cell_means": float(means.mean()),
        "cells_with_primary_better": int((means < 0).sum()),
    }
    if len(rows) >= 2 and np.any(means != 0):
        test = wilcoxon(means, alternative="two-sided")
        summary["wilcoxon_two_sided_p"] = float(test.pvalue)
    summary["dataset_mean_delta"] = {
        d: float(np.mean([r["mean"] for r in rows if r["data"] == d]))
        for d in sorted({r["data"] for r in rows})
    }

    analysis = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"analysis": analysis, "cells": rows, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
