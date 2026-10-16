"""Exchangeable two-regime check; these are simulated, not research dataset results."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_paper_suite import fit_methods, score_bounds, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("results/paper_strengthening/simulated"))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cal_phase, test_phase = rng.integers(0, 2, 4000), rng.integers(0, 2, 20000)
    cal = np.abs(rng.normal(size=4000) * np.where(cal_phase == 0, 1, 4))[:, None, None]
    true = (rng.normal(size=20000) * np.where(test_phase == 0, 1, 4))[:, None, None]
    states = cal_phase[:, None].astype("float32")
    rows = []
    for alpha in (0.05, 0.1, 0.2):
        fitted = fit_methods(cal, states, cal_phase, alpha, 2, args.seed, 1, 2)
        labels = fitted["cluster"].assign_clusters(test_phase[:, None].astype("float32"))
        for method in ("flat_paired", "flat_full", "phase", "state"):
            if method == "phase":
                widths = fitted["phase"].half_widths(test_phase)
            elif method == "state":
                widths = np.stack([fitted["cluster"].quantiles[int(k)] for k in labels])
            else:
                widths = np.broadcast_to(fitted[method].quantile_, true.shape)
            scores, _ = score_bounds(np.zeros_like(true), true, widths, alpha)
            for regime in (0, 1):
                mean = scores[test_phase == regime].mean(axis=0)
                rows.append(dict(alpha=alpha, method=method, regime=regime, coverage=mean[0], width=mean[1], winkler=mean[2]))
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "results.csv", index=False)
    write_json(args.output / "config.json", {"seed": args.seed, "n_cal": 4000, "n_test": 20000,
        "sigmas": [1, 4], "regime_probability": 0.5, "role": "exchangeable_simulation", "alphas": [0.05, 0.1, 0.2]})
    for name, value in (("cal_phase", cal_phase), ("cal_residuals", cal), ("test_phase", test_phase), ("test_targets", true)):
        np.save(args.output / f"{name}.npy", value)
    print(pd.DataFrame(rows).query("alpha == 0.1").to_string(index=False))


if __name__ == "__main__":
    main()
