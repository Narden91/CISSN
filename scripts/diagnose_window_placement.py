#!/usr/bin/env python
"""Where the conditioning window sits decides the Winkler gain, not only its size.

`diagnose_conditioning_headroom.py` fits the conditioning window and the
calibration window inside the test period, right before the evaluated windows.
The protocol fits them on the older calibration split. The two differ in
recency, in window size, and in overlap: stride-1 windows closer than
`pred_len` share targets, so a calibration window that touches the evaluated
windows leaks information into them.

This script scores one fixed set of evaluation windows (the last
`n_test - region_start` test windows) under three placements of the fit and
calibration windows, with the same window sizes the protocol uses:

- `protocol`          saved conditioning and calibration halves (older, not in test).
- `adjacent_no_gap`   fit and calibration windows end where the evaluated windows start.
- `adjacent_gap`      the same, with a `pred_len` gap before the evaluated windows and between fit and calibration.

Each mechanism is compared with flat CP calibrated on the same calibration
window, so the placement cancels out of the delta except through the
conditioning. Reading: `adjacent_no_gap` against `adjacent_gap` isolates
overlap leakage; `protocol` against `adjacent_gap` mixes recency with the shift
between the calibration split and the test period. Recency alone cannot be
separated: with this few test windows a gapped adjacent placement is also the
earliest possible one.

Evidence role: exploratory reuse of ETTh1 test artifacts. Not selection-eligible.

Usage:
    uv run python scripts/diagnose_window_placement.py \\
        --seeds 42 123 456 --output results/validation/window_placement.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cissn.conformal import StateConditionalConformal, StateScaledConformal
from cissn.evaluation.inference import moving_block_bootstrap_ci
from cissn.evaluation.metrics import per_origin_interval_scores
from diagnose_conditioning_headroom import _flat_bounds

SCHEMES = ("protocol", "adjacent_no_gap", "adjacent_gap")
SETTING = "CISSN_ETTh1_M_sl96_pl336_sd5_dm64_fullrevin_lc1_lt0p5_a0p1_per_feature_seed{seed}"


def window_slices(scheme: str, n_fit: int, n_cal: int, gap: int, region_start: int) -> tuple[slice, slice]:
    """Fit and calibration slices (over test windows) for one placement."""
    if scheme == "adjacent_no_gap":
        fit, cal = slice(region_start - n_cal - n_fit, region_start - n_cal), slice(region_start - n_cal, region_start)
    elif scheme == "adjacent_gap":
        cal_end = region_start - gap
        cal = slice(cal_end - n_cal, cal_end)
        fit = slice(cal.start - gap - n_fit, cal.start - gap)
    else:
        raise ValueError(f"Unknown scheme {scheme!r}.")
    if fit.start < 0 or cal.stop > region_start:
        raise ValueError(f"{scheme}: windows do not fit before region_start={region_start}.")
    return fit, cal


def _bounds(mechanism, fit_s, fit_r, cal_s, cal_r, eval_s, eval_p, alpha, seed, n_clusters):
    if mechanism == "flat":
        return _flat_bounds(cal_r, eval_p, alpha)
    if mechanism == "cluster":
        model = StateConditionalConformal(alpha=alpha, n_clusters=n_clusters, random_state=seed)
        model.fit_partition(fit_s)
    else:
        model = StateScaledConformal(alpha=alpha, scale_geometry="per_cell")
        model.fit_scale(fit_s, fit_r)
    model.calibrate(cal_s, cal_r)
    lower, upper = model.predict(torch.from_numpy(eval_s).float(), torch.from_numpy(eval_p).float())
    return lower.numpy(), upper.numpy()


def run_seed(seed: int, alpha: float, n_clusters: int, n_boot: int) -> dict:
    results_dir = Path("results/validation") / SETTING.format(seed=seed)
    ckpt_dir = Path("checkpoints/validation") / SETTING.format(seed=seed)
    states, residuals = np.load(results_dir / "states.npy"), np.load(results_dir / "residuals.npy")
    preds, trues = np.load(results_dir / "pred.npy"), np.load(results_dir / "true.npy")
    protocol = {
        "fit": (np.load(ckpt_dir / "conditioning_states.npy"), np.load(ckpt_dir / "conditioning_residuals.npy")),
        "cal": (np.load(ckpt_dir / "calibration_states.npy"), np.load(ckpt_dir / "calibration_residuals.npy")),
    }
    n_fit, n_cal, horizon = protocol["fit"][0].shape[0], protocol["cal"][0].shape[0], preds.shape[1]
    # Smallest start at which every placement keeps both gaps; fixes one shared evaluation set.
    region_start = 2 * horizon + n_fit + n_cal
    if states.shape[0] - region_start < horizon:
        raise ValueError("Too few test windows left for a block-bootstrap evaluation.")
    eval_s, eval_p, eval_t = states[region_start:], preds[region_start:], trues[region_start:]

    out = {"seed": seed, "n_fit": n_fit, "n_cal": n_cal, "gap": horizon,
           "region_start": region_start, "n_eval": int(eval_s.shape[0]), "schemes": {}}
    for scheme in SCHEMES:
        if scheme == "protocol":
            (fit_s, fit_r), (cal_s, cal_r) = protocol["fit"], protocol["cal"]
        else:
            fit, cal = window_slices(scheme, n_fit, n_cal, horizon, region_start)
            fit_s, fit_r, cal_s, cal_r = states[fit], residuals[fit], states[cal], residuals[cal]
        winkler = {}
        for mechanism in ("flat", "cluster", "per_cell"):
            lo, hi = _bounds(mechanism, fit_s, fit_r, cal_s, cal_r, eval_s, eval_p, alpha, seed, n_clusters)
            winkler[mechanism] = per_origin_interval_scores(lo, hi, eval_t, alpha)[:, 2]
        out["schemes"][scheme] = {
            "flat_winkler": float(winkler["flat"].mean()),
            **{
                f"{m}_minus_flat": moving_block_bootstrap_ci(winkler[m] - winkler["flat"], horizon, n_boot, seed)
                for m in ("cluster", "per_cell")
            },
        }
    # Protocol windows sit before the test period, so they can score any test region
    # without a gap. If the gain moves with the region, region choice explains
    # differences between analyses that evaluate different windows.
    (fit_s, fit_r), (cal_s, cal_r) = protocol["fit"], protocol["cal"]
    edges = np.linspace(0, states.shape[0], 5, dtype=int)
    out["protocol_region_scan"] = []
    for start, stop in zip(edges[:-1], edges[1:]):
        winkler = {}
        for mechanism in ("flat", "cluster", "per_cell"):
            lo, hi = _bounds(mechanism, fit_s, fit_r, cal_s, cal_r, states[start:stop], preds[start:stop],
                             alpha, seed, n_clusters)
            winkler[mechanism] = per_origin_interval_scores(lo, hi, trues[start:stop], alpha)[:, 2]
        out["protocol_region_scan"].append({
            "start": int(start), "stop": int(stop), "flat_winkler": float(winkler["flat"].mean()),
            "cluster_minus_flat": float((winkler["cluster"] - winkler["flat"]).mean()),
            "per_cell_minus_flat": float((winkler["per_cell"] - winkler["flat"]).mean()),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = [run_seed(seed, args.alpha, args.n_clusters, args.n_boot) for seed in args.seeds]
    summary = {}
    for scheme in SCHEMES:
        for mechanism in ("cluster", "per_cell"):
            key = f"{mechanism}_minus_flat"
            deltas = [run["schemes"][scheme][key]["mean"] for run in runs]
            summary[f"{scheme}/{mechanism}"] = {
                "mean_delta": float(np.mean(deltas)),
                "per_seed": deltas,
                "seeds_better": int(sum(d < 0 for d in deltas)),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"evidence_role": "exploratory_test_reuse", "selection_eligible": False,
                    "runs": runs, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print("\nprotocol windows, test quarters (mean over seeds): start-stop flat cluster-flat per_cell-flat")
    for i in range(4):
        rows = [run["protocol_region_scan"][i] for run in runs]
        print(f"  {rows[0]['start']:>5}-{rows[0]['stop']:<5} {np.mean([r['flat_winkler'] for r in rows]):8.3f}"
              f" {np.mean([r['cluster_minus_flat'] for r in rows]):+9.4f}"
              f" {np.mean([r['per_cell_minus_flat'] for r in rows]):+9.4f}")
    print(f"\n{'scheme/mechanism':<28}{'mean dW vs flat':>16}{'per seed':>40}")
    for name, row in summary.items():
        print(f"{name:<28}{row['mean_delta']:>+16.4f}   " + "  ".join(f"{d:+.4f}" for d in row["per_seed"]))


if __name__ == "__main__":
    main()
