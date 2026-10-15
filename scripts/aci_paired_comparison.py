#!/usr/bin/env python
"""Adaptive conformal inference (ACI) against flat CP on the same CISSN forecasts.

RUNBOOK Step 7, P2. For each finished CISSN run (legacy architecture, RevIN, confirmation
role) this script:

1. reloads the checkpoint and reruns calibration through `_calibrate_conformal`, which
   recovers the second-half calibration residuals that flat CP used;
2. checks the reload: ACI with gamma 0 must reproduce the run's saved flat-CP per-origin
   scores (`per_origin_interval_flat_cp_metrics.npy`), or the script stops;
3. runs ACI with the given gamma on the saved test residuals, in origin order, with an
   H-step feedback delay (`cissn/conformal/aci.py`);
4. per dataset x horizon cell, averages the per-origin Winkler score over seeds and
   bootstraps the ACI-minus-flat difference in blocks of consecutive origins, exactly as
   `paired_interval_inference.py` does.

ACI uses realised test errors as they arrive; split conformal prediction does not. Report
it as a calibrator with more information, not as a like-for-like comparator. Evidence
role: descriptive. Every analysis setting is a required argument with no default.

Usage:
    uv run python scripts/aci_paired_comparison.py --results_root results/publication \\
        --checkpoints_root checkpoints/publication --datasets ETTh2 weather exchange_rate \\
        --gamma 0.005 --block_multiplier 1 --n_boot 5000 --seed 0 \\
        --output results/publication/aci_vs_flat_exposed.json
"""
import argparse
import json
import sys
import tempfile
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cissn.conformal.aci import adaptive_conformal_quantiles  # noqa: E402
from cissn.evaluation.inference import moving_block_bootstrap_ci  # noqa: E402
from cissn.evaluation.metrics import per_origin_interval_scores  # noqa: E402
from experiments.run_benchmark import Experiment  # noqa: E402
from paired_interval_inference import WINKLER_COLUMN, _cell_delta  # noqa: E402

# Per-origin Winkler scores are means over float32 forecasts; the reload check allows that noise.
REPRODUCTION_TOLERANCE = 1e-4


def _calibration_residuals(run: Path, checkpoints_root: Path, config: dict) -> np.ndarray:
    config = {**config, "require_gpu": False, "no_progress": True}
    exp = Experiment(Namespace(**config))
    exp._load_checkpoint(checkpoints_root / run.name)
    exp._build_conditioning_predictors()
    _, cal_loader = exp._get_data(flag="cal")
    with tempfile.TemporaryDirectory() as tmp:
        exp._calibrate_conformal(cal_loader, artifact_dir=tmp)
        return np.load(Path(tmp) / "calibration_residuals.npy")


def score_run(run: Path, checkpoints_root: Path, config: dict, gamma: float) -> dict:
    alpha, horizon = float(config["conformal_alpha"]), int(config["pred_len"])
    pred, true = np.load(run / "pred.npy"), np.load(run / "true.npy")
    origins = np.load(run / "test_origin_indices.npy")
    test_scores = np.abs(pred - true)
    cal_scores = _calibration_residuals(run, checkpoints_root, config)

    flat_saved = np.load(run / "per_origin_interval_flat_cp_metrics.npy")
    flat_q, _ = adaptive_conformal_quantiles(cal_scores, test_scores, origins, horizon, alpha, 0.0)
    flat = per_origin_interval_scores(pred - flat_q, pred + flat_q, true, alpha)
    drift = float(np.max(np.abs(flat[:, WINKLER_COLUMN] - flat_saved[:, WINKLER_COLUMN])
                         / np.abs(flat_saved[:, WINKLER_COLUMN])))
    if drift > REPRODUCTION_TOLERANCE:
        raise ValueError(f"{run.name}: reload does not reproduce the saved flat CP (max relative drift {drift:.2e}).")

    aci_q, n_capped = adaptive_conformal_quantiles(cal_scores, test_scores, origins, horizon, alpha, gamma)
    return {
        "run_dir": run.name,
        "seed": int(config["seed"]),
        "stride": int(config.get("test_origin_stride", 1)),
        "conditioning": config["conformal_conditioning"],
        "scale_geometry": config.get("scale_geometry", "scalar"),
        "n_calibration": int(cal_scores.shape[0]),
        "reproduction_max_relative_drift": drift,
        "n_capped": n_capped,
        "origins": origins,
        "primary": per_origin_interval_scores(pred - aci_q, pred + aci_q, true, alpha),
        "flat": flat_saved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--checkpoints_root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--gamma", type=float, required=True, help="ACI step size")
    parser.add_argument("--block_multiplier", type=float, required=True)
    parser.add_argument("--n_boot", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.block_multiplier < 1.0:
        raise ValueError("block_multiplier below 1 gives intervals that are too narrow.")

    cells: dict = defaultdict(list)
    for metrics_path in sorted(args.results_root.glob("CISSN_*/metrics.json")):
        run = metrics_path.parent
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if config["data"] not in args.datasets or config.get("architecture") != "legacy" or not config.get("revin"):
            continue
        if config.get("evidence_role") != "confirmation":
            raise ValueError(f"{run.name}: evidence_role is not 'confirmation'.")
        if not json.loads(metrics_path.read_text(encoding="utf-8")).get("structural_passed", False):
            print(f"skip {run.name}: structural_passed is false")
            continue
        print(f"ACI on {run.name}", flush=True)
        cells[(config["data"], int(config["pred_len"]))].append(
            score_run(run, args.checkpoints_root, config, args.gamma))
    if not cells:
        raise RuntimeError("No matching confirmation runs found.")

    rows = []
    for (data, horizon), runs in sorted(cells.items()):
        stride = runs[0]["stride"]
        delta, flat_mean = _cell_delta(runs)
        result = moving_block_bootstrap_ci(
            delta, int(np.ceil(args.block_multiplier * horizon / stride)), args.n_boot, args.seed, args.confidence)
        aci = np.mean([run["primary"] for run in runs], axis=0)
        flat = np.mean([run["flat"] for run in runs], axis=0)
        rows.append({
            "data": data, "horizon": horizon, "stride": stride,
            "seeds": sorted(run["seed"] for run in runs),
            "flat_winkler_mean": flat_mean,
            "relative_mean": result["mean"] / flat_mean,
            "relative_ci_low": result["ci_low"] / flat_mean,
            "relative_ci_high": result["ci_high"] / flat_mean,
            "aci_coverage": float(aci[:, 0].mean()), "flat_coverage": float(flat[:, 0].mean()),
            "aci_width": float(aci[:, 1].mean()), "flat_width": float(flat[:, 1].mean()),
            "runs": [{k: run[k] for k in ("run_dir", "seed", "n_calibration", "n_capped",
                                          "reproduction_max_relative_drift")} for run in runs],
            **result,
        })

    analysis = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"analysis": analysis, "cells": rows}, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['data']} h{row['horizon']}: relative {row['relative_mean']:+.4f} "
              f"[{row['relative_ci_low']:+.4f}, {row['relative_ci_high']:+.4f}]")


if __name__ == "__main__":
    main()
