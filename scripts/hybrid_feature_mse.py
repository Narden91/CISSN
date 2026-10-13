#!/usr/bin/env python
"""Per-feature validation MSE of a hybrid run against its own frozen DLinear base.

Closes RUNBOOK Step 5 criterion 4: no feature's mean MSE may degrade by more
than 5%. `run_selection.py` saves only the pooled validation MSE, so this loads
each saved checkpoint and scores the validation split again. It does not train,
and it never builds the test loader (`HybridSelectionExperiment` raises on it).

The base and the hybrid are scored on the same batches inside one model, so the
pairing is exact. Per feature, the mean over seeds is compared with the base
mean over the same seeds, as the criterion states "mean MSE".

Usage:
    uv run python scripts/hybrid_feature_mse.py \\
        --checkpoint_dirs checkpoints/selection/CISSN_ETTh1_..._hybrid_anchored_..._seed42 ... \\
        --output results/selection/hybrid_feature_mse.json
"""
import argparse
import json
import re
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_selection import HybridSelectionExperiment  # noqa: E402

MAX_FEATURE_DEGRADATION = 0.05


def per_feature_mse(checkpoint_dir: Path) -> dict:
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    if config["architecture"] != "hybrid":
        raise ValueError(f"{checkpoint_dir.name}: not a hybrid run.")
    # CPU is enough for one validation pass and keeps this usable without a GPU.
    config["require_gpu"] = False
    exp = HybridSelectionExperiment(Namespace(**config))
    exp._load_checkpoint(checkpoint_dir)
    _, loader = exp._get_data(flag="val")

    f_dim = -1 if config["features"] == "MS" else 0
    pred_len = config["pred_len"]
    sq_base = sq_hybrid = None
    count = 0
    exp._set_train_mode(False)
    with torch.no_grad():
        for batch_x, batch_y, _bxm, _bym in loader:
            batch_x = batch_x.float().to(exp.device)
            target = batch_y.float().to(exp.device)[:, -pred_len:, f_dim:]
            parts = exp.hybrid.forward_components(batch_x)
            base = parts["base"][:, -pred_len:, f_dim:]
            total = parts["total"][:, -pred_len:, f_dim:]
            b = ((base - target) ** 2).double().sum(dim=(0, 1)).cpu().numpy()
            h = ((total - target) ** 2).double().sum(dim=(0, 1)).cpu().numpy()
            sq_base = b if sq_base is None else sq_base + b
            sq_hybrid = h if sq_hybrid is None else sq_hybrid + h
            count += target.shape[0] * target.shape[1]
    return {
        "setting": checkpoint_dir.name,
        "seed": int(config["seed"]),
        "n_values_per_feature": int(count),
        "base_mse": (sq_base / count).tolist(),
        "hybrid_mse": (sq_hybrid / count).tolist(),
    }


def summarise_arm(runs: list[dict]) -> dict:
    base = np.mean([run["base_mse"] for run in runs], axis=0)
    hybrid = np.mean([run["hybrid_mse"] for run in runs], axis=0)
    relative = hybrid / base - 1.0
    return {
        "seeds": sorted(run["seed"] for run in runs),
        "base_mse_mean": base.tolist(),
        "hybrid_mse_mean": hybrid.tolist(),
        "relative_change_per_feature": relative.tolist(),
        "worst_feature_change": float(relative.max()),
        "criterion_4_passes": bool(relative.max() <= MAX_FEATURE_DEGRADATION),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = [per_feature_mse(path) for path in args.checkpoint_dirs]
    arms: dict[str, list[dict]] = {}
    for run in runs:
        arms.setdefault(re.sub(r"_seed\d+$", "", run["setting"]), []).append(run)
    summary = {arm: summarise_arm(members) for arm, members in arms.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": runs, "arms": summary}, indent=2), encoding="utf-8")
    for arm, result in summary.items():
        print(f"{arm}: worst feature {result['worst_feature_change'] * 100:+.2f}% "
              f"criterion 4 {'passes' if result['criterion_4_passes'] else 'FAILS'}")


if __name__ == "__main__":
    main()
