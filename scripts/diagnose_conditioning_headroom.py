#!/usr/bin/env python
"""
Compare state conditioning mechanisms on a shared calibration window, and bound
what a per-sample difficulty estimator can buy.

This script reports, from a saved run's artifacts:

1. **Paired comparison.** Flat CP, cluster SCCP, and both state-scaled
   geometries scored on the SAME chronological calibration window and the SAME
   test forecasts. A comparison where one method calibrates on more data than
   another is not evidence about conditioning -- that error produced a result
   this repository had to retract (see docs/methodology.md).
2. **Label-informed scalar reference.** State-scaled CP where sigma is the
   *realized* per-sample residual scale. This uses evaluation labels, so it is
   not a usable predictor, an oracle bound, or selection evidence.
3. **Variance decomposition.** How much residual variance sits on the
   per-sample axis (all a scalar sigma can reach) against the per-cell axis
   (already captured by `per_feature` quantiles).

The scalar reference is not a bound on state conditioning as such. The conditioning
signal is a state x cell interaction: the state changes *which cells are hard*,
not how hard a window is overall. So the `percell` column and cluster SCCP
routinely beat the scalar label-informed column -- they exploit an axis the scalar
geometry cannot represent. Treat it as a diagnostic only.

Evidence role: this script reloads saved evaluation artifacts and repeatedly
partitions them. Its output is always exploratory. It must never choose defaults,
modify a study manifest, or enter a confirmatory table.

Usage:
    uv run python scripts/diagnose_conditioning_headroom.py \\
        --run_dir results/validation/CISSN_ETTh1_... \\
        --output results/validation/conditioning_headroom.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cissn.conformal import StateConditionalConformal, StateScaledConformal
from cissn.conformal.state_conditional import split_conformal_q_level
from cissn.evaluation.metrics import compute_picp, winkler_score

# Chronological cut points. Each defines test = [cut:]; the region before it is
# split in half into a scale/partition-fitting window and a calibration window,
# mirroring the real protocol's requirement that sigma(s) and the K-Means
# partition are fit on data disjoint from the calibration split.
DEFAULT_CUTS = (0.3, 0.4, 0.5, 0.6)


def _flat_bounds(cal_residuals: np.ndarray, test_preds: np.ndarray, alpha: float):
    q = np.quantile(
        cal_residuals, split_conformal_q_level(cal_residuals.shape[0], alpha), axis=0, method="higher"
    )
    return test_preds - q, test_preds + q


def _label_informed_scalar_bounds(cal_residuals, test_residuals, test_preds, alpha):
    """Diagnostic bounds using realized evaluation residual scales.

    This is intentionally label-informed and has no upper-bound interpretation:
    it optimizes neither Winkler score nor every scalar difficulty function.
    """
    cal_sigma = cal_residuals.reshape(cal_residuals.shape[0], -1).mean(1).reshape(-1, 1, 1)
    test_sigma = test_residuals.reshape(test_residuals.shape[0], -1).mean(1).reshape(-1, 1, 1)
    normalized = cal_residuals / cal_sigma
    q = np.quantile(
        normalized, split_conformal_q_level(normalized.shape[0], alpha), axis=0, method="higher"
    )
    width = q * test_sigma
    return test_preds - width, test_preds + width


def _score(lower, upper, trues, alpha):
    return {
        "winkler": float(winkler_score(lower, upper, trues, alpha=alpha)),
        "coverage": float(compute_picp(lower, upper, trues)),
        "mean_width": float(np.mean(upper - lower)),
    }


def _variance_decomposition(test_residuals: np.ndarray) -> dict:
    """Split residual variance into the axis sigma(s) can reach and the rest."""
    total = float(test_residuals.var())
    per_sample = float(test_residuals.reshape(test_residuals.shape[0], -1).mean(1).var())
    per_cell = float(test_residuals.mean(0).var())
    return {
        "total_variance": total,
        "per_sample_variance": per_sample,
        "per_sample_fraction": per_sample / total if total > 0 else None,
        "per_cell_variance": per_cell,
        "per_cell_fraction": per_cell / total if total > 0 else None,
    }


def diagnose(run_dir: Path, alpha: float, n_clusters: int, seed: int, cuts=DEFAULT_CUTS) -> dict:
    states = np.load(run_dir / "states.npy")
    residuals = np.load(run_dir / "residuals.npy")
    preds = np.load(run_dir / "pred.npy")
    trues = np.load(run_dir / "true.npy")
    n = states.shape[0]

    rows = []
    for cut in cuts:
        fit_end = int(n * cut * 0.5)
        cal_end = int(n * cut)
        if fit_end < 2 or cal_end - fit_end < 2 or n - cal_end < 2:
            raise ValueError(f"cut={cut} leaves an empty fit, calibration, or test window for n={n}.")

        fit_states, fit_residuals = states[:fit_end], residuals[:fit_end]
        cal_states, cal_residuals = states[fit_end:cal_end], residuals[fit_end:cal_end]
        test_states = states[cal_end:]
        test_preds, test_trues = preds[cal_end:], trues[cal_end:]
        test_residuals = residuals[cal_end:]

        entry = {"cut": cut, "n_fit": fit_end, "n_cal": cal_end - fit_end, "n_test": n - cal_end}

        lower, upper = _flat_bounds(cal_residuals, test_preds, alpha)
        entry["flat_cp"] = _score(lower, upper, test_trues, alpha)

        cluster = StateConditionalConformal(alpha=alpha, n_clusters=n_clusters, random_state=seed)
        cluster.fit_partition(fit_states)
        cluster.calibrate(cal_states, cal_residuals)
        lo, hi = cluster.predict(
            torch.from_numpy(test_states).float(), torch.from_numpy(test_preds).float()
        )
        entry["cluster_cp"] = _score(lo.numpy(), hi.numpy(), test_trues, alpha)

        scaled = StateScaledConformal(alpha=alpha)
        scaled.fit_scale(fit_states, fit_residuals)
        scaled.calibrate(cal_states, cal_residuals)
        lo, hi = scaled.predict(
            torch.from_numpy(test_states).float(), torch.from_numpy(test_preds).float()
        )
        entry["state_scaled_cp"] = _score(lo.numpy(), hi.numpy(), test_trues, alpha)

        # Per-cell sigma: the geometry that can express the state x cell
        # interaction a scalar sigma is blind to.
        per_cell = StateScaledConformal(alpha=alpha, scale_geometry="per_cell")
        per_cell.fit_scale(fit_states, fit_residuals)
        per_cell.calibrate(cal_states, cal_residuals)
        lo, hi = per_cell.predict(
            torch.from_numpy(test_states).float(), torch.from_numpy(test_preds).float()
        )
        entry["state_scaled_per_cell_cp"] = _score(lo.numpy(), hi.numpy(), test_trues, alpha)

        sigma_test = scaled.difficulty_score(test_states)
        true_scale = test_residuals.reshape(test_residuals.shape[0], -1).mean(1)
        entry["sigma_diagnostics"] = {
            "sigma_cv": float(sigma_test.std() / sigma_test.mean()),
            "true_scale_cv": float(true_scale.std() / true_scale.mean()),
            "corr_sigma_true_scale": float(np.corrcoef(sigma_test, true_scale)[0, 1]),
        }

        lower, upper = _label_informed_scalar_bounds(cal_residuals, test_residuals, test_preds, alpha)
        entry["label_informed_per_sample_reference"] = _score(lower, upper, test_trues, alpha)

        flat_w = entry["flat_cp"]["winkler"]
        entry["winkler_delta_vs_flat"] = {
            "cluster_cp": entry["cluster_cp"]["winkler"] - flat_w,
            "state_scaled_cp": entry["state_scaled_cp"]["winkler"] - flat_w,
            "state_scaled_per_cell_cp": entry["state_scaled_per_cell_cp"]["winkler"] - flat_w,
            "label_informed_per_sample_reference": (
                entry["label_informed_per_sample_reference"]["winkler"] - flat_w
            ),
        }
        rows.append(entry)

    cal_end = int(n * cuts[-1])
    summary = {
        "run_dir": str(run_dir),
        "evidence_role": "exploratory_test_reuse",
        "selection_eligible": False,
        "alpha": alpha,
        "n_clusters": n_clusters,
        "cuts": list(cuts),
        "per_cut": rows,
        "variance_decomposition": _variance_decomposition(residuals[cal_end:]),
    }
    for method in (
        "cluster_cp",
        "state_scaled_cp",
        "state_scaled_per_cell_cp",
        "label_informed_per_sample_reference",
    ):
        deltas = [r["winkler_delta_vs_flat"][method] for r in rows]
        summary.setdefault("summary_vs_flat", {})[method] = {
            "mean_winkler_delta": float(np.mean(deltas)),
            "cuts_won": int(sum(d < 0 for d in deltas)),
            "cuts_total": len(deltas),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Run directory holding states.npy, residuals.npy, pred.npy, true.npy.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = diagnose(Path(args.run_dir), args.alpha, args.n_clusters, args.seed)

    print(f"{'cut':<6}{'flat':>9}{'cluster':>10}{'scaled':>10}{'percell':>10}{'label-ref':>10}")
    for row in result["per_cut"]:
        print(
            f"{row['cut']:<6}{row['flat_cp']['winkler']:9.3f}"
            f"{row['cluster_cp']['winkler']:10.3f}"
            f"{row['state_scaled_cp']['winkler']:10.3f}"
            f"{row['state_scaled_per_cell_cp']['winkler']:10.3f}"
            f"{row['label_informed_per_sample_reference']['winkler']:10.3f}"
        )
    print("\nMean Winkler delta vs flat CP (negative = better than flat):")
    for method, stats in result["summary_vs_flat"].items():
        print(f"  {method:<20} {stats['mean_winkler_delta']:+.4f} "
              f"(better on {stats['cuts_won']}/{stats['cuts_total']} cuts)")
    decomp = result["variance_decomposition"]
    print(f"\nResidual variance reachable by a per-sample sigma(s): "
          f"{100 * decomp['per_sample_fraction']:.2f}%")
    print(f"Residual variance on the per-cell axis (already handled by per_feature): "
          f"{100 * decomp['per_cell_fraction']:.2f}%")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
