#!/usr/bin/env python
"""Hybrid against its own frozen DLinear base on the evaluated test origins.

RUNBOOK Step 5 pairs each hybrid with the frozen base inside the same run. This
script keeps that definition on test: it reloads a finished hybrid run's
checkpoint, runs `forward_components` on the run's saved test origins, and
compares the `base` output with the `total` output, overall and per feature.
Both come from one set of weights and one data pass, so the pairing is exact and
adds no training-run variance.

A reproduction check guards the reload: `total` must match the run's saved
`pred.npy`. With three seeds the output reports counts of seeds better, not a
significance test.

Evidence role: confirmatory only on artifacts from `--evidence_role confirmation`
runs, as prespecified in `docs/prespecification.md`, section 6.

Usage:
    uv run python scripts/paired_point_comparison.py --results_root results/publication \\
        --checkpoints_root checkpoints/publication --data ECL --pred_len 336 \\
        --seeds 42 123 456 --output results/publication/hybrid_vs_base_ECL.json
"""
import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_benchmark import HybridExperiment  # noqa: E402

HYBRID_PATTERN = "CISSN_{data}_M_sl96_pl{pred_len}_sd5_dm64_hybrid_anchored_lc0_lt0_*_seed{seed}__*"
REPRODUCTION_TOLERANCE = 1e-3
BATCH = 64


def _one_dir(root: Path, pattern: str) -> Path:
    matches = sorted(path for path in root.glob(pattern) if (path / "metrics.json").is_file())
    if len(matches) != 1:
        raise ValueError(f"Expected one finished run for {pattern!r}, found {len(matches)}.")
    return matches[0]


def compare_seed(results_root: Path, checkpoints_root: Path, data: str, pred_len: int, seed: int) -> dict:
    run = _one_dir(results_root, HYBRID_PATTERN.format(data=data, pred_len=pred_len, seed=seed))
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    config["require_gpu"] = False
    exp = HybridExperiment(Namespace(**config))
    exp._load_checkpoint(checkpoints_root / run.name)
    exp._set_train_mode(False)

    origins = np.load(run / "test_origin_indices.npy")
    saved_pred, saved_true = np.load(run / "pred.npy"), np.load(run / "true.npy")
    test_data, _ = exp._get_data(flag="test")
    f_dim = -1 if config["features"] == "MS" else 0

    base_rows, total_rows, true_rows = [], [], []
    with torch.no_grad():
        for start in range(0, origins.size, BATCH):
            items = [test_data[i] for i in origins[start:start + BATCH]]
            batch_x = torch.from_numpy(np.stack([item[0] for item in items])).float().to(exp.device)
            parts = exp.hybrid.forward_components(batch_x)
            base_rows.append(parts["base"][:, -pred_len:, f_dim:].cpu().numpy())
            total_rows.append(parts["total"][:, -pred_len:, f_dim:].cpu().numpy())
            true_rows.append(np.stack([item[1] for item in items])[:, -pred_len:, f_dim:])
    base, total, true = (np.concatenate(rows) for rows in (base_rows, total_rows, true_rows))

    # A reload that does not reproduce the saved forecast would compare the wrong weights.
    drift = float(np.abs(total - saved_pred).max())
    if drift > REPRODUCTION_TOLERANCE or not np.allclose(true, saved_true):
        raise ValueError(f"seed {seed}: reload does not reproduce the saved run (max drift {drift:.2e}).")

    def feature_mse(pred):
        return ((pred.astype(np.float64) - true.astype(np.float64)) ** 2).mean(axis=(0, 1))

    h_mse, b_mse = feature_mse(total), feature_mse(base)
    return {
        "seed": seed,
        "run_dir": run.name,
        "n_origins": int(origins.size),
        "reproduction_max_abs_drift": drift,
        "base_mse": float(b_mse.mean()),
        "hybrid_mse": float(h_mse.mean()),
        "mse_ratio": float(h_mse.mean() / b_mse.mean()),
        "per_feature_change": (h_mse / b_mse - 1.0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--checkpoints_root", type=Path, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = [
        compare_seed(args.results_root, args.checkpoints_root, args.data, args.pred_len, seed)
        for seed in args.seeds
    ]
    ratios = np.array([run["mse_ratio"] for run in runs])
    summary = {
        "seeds_hybrid_better": int((ratios < 1).sum()),
        "seeds_total": len(runs),
        "mean_mse_ratio": float(ratios.mean()),
        "worst_feature_change_any_seed": float(max(max(run["per_feature_change"]) for run in runs)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": runs, "summary": summary}, indent=2), encoding="utf-8")
    for run in runs:
        print(f"seed {run['seed']}: base {run['base_mse']:.4f}  hybrid {run['hybrid_mse']:.4f}  "
              f"ratio {run['mse_ratio']:.4f}  drift {run['reproduction_max_abs_drift']:.1e}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
