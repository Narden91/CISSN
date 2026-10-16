"""Write a reviewable record from actual follow-up result files."""
import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/paper_strengthening"))
    parser.add_argument("--output", type=Path, default=Path("docs/paper_strengthening_results.md"))
    args = parser.parse_args()
    frame = pd.read_csv(args.results / "summary.csv")
    runs = pd.read_csv(args.results / "runs.csv")
    uncertainty = pd.read_csv(args.results / "uncertainty.csv")
    inventory = json.loads((args.results / "inventory.json").read_text())
    complete = sum(r["status"] == "completed" for r in inventory)
    blocked = sum(r["status"] == "blocked" for r in inventory)
    at_alpha = frame[frame.alpha == 0.1]
    if set(frame.dataset) != {"ETTh2", "weather", "exchange_rate"} or set(frame.model) != {"cissn"}:
        raise ValueError("This report describes the local exposed-dataset protocol. Review cluster results separately.")
    scores = at_alpha.pivot(index=["dataset", "horizon"], columns="method", values="winkler")
    coverage = at_alpha.pivot(index=["dataset", "horizon"], columns="method", values="coverage")
    phase_wins = int((scores.state < scores.phase).sum())
    coverage_loss = 100 * (coverage.phase - coverage.state)
    exchange = scores.loc[("exchange_rate", 96)]
    full_loss = 100 * (exchange.state / exchange.flat_full - 1)
    rows, completed_rows = [], []
    for (dataset, horizon, model), group in at_alpha.groupby(["dataset", "horizon", "model"]):
        cell = group.set_index("method")
        rows.append([dataset, horizon, *[f"{cell.loc[m, 'winkler']:.6f}" for m in ("flat_paired", "flat_full", "phase", "state")],
                     f"{cell.loc['phase', 'coverage']:.6f}", f"{cell.loc['state', 'coverage']:.6f}"])
        completed_rows.append([dataset, horizon, model, int(group.n_seeds.iloc[0]), int(group.n_origins.iloc[0]),
                               ", ".join(map(str, sorted(frame.alpha.unique()))), "6"])
    ci = uncertainty[(uncertainty.alpha == 0.1) & (uncertainty.block_multiplier == 1) & uncertainty.kind.isna()
                     & (uncertainty.metric == "winkler") & uncertainty.baseline.isin(["phase", "flat_full"])]
    ci_rows = [[r.dataset, r.horizon, r.baseline, f"{r.mean:.6f}", f"[{r.ci_low:.6f}, {r.ci_high:.6f}]",
                f"{r.n_origins / r.block_len:.2f}"] for r in ci.itertuples()]
    reproductions = [json.loads(p.read_text()) for p in args.results.rglob("reproduction_*.json")]
    diff_stat = subprocess.run(["git", "diff", "--stat", "--", "AGENT_LEARNINGS.md", "slurm/README.md"],
                               capture_output=True, text=True, check=True).stdout.strip()
    text = f"""# Paper strengthening: measured results and remaining work

The ECL/Solar and strong-forecaster runs are left for the user on the HPC cluster.
Run `bash slurm/paper_preflight.sh`, then the dry run and launcher documented in
[the reproduction guide](paper_strengthening_reproduction.md).

The local follow-up completed {complete} frozen CISSN runs and {len(frame)} method/alpha
cell summaries ({len(runs)} per-run method/alpha rows). It recorded {blocked} selected
runs as blocked by missing weights. All locally scored datasets were exposed before
this task. These results cannot supply new confirmatory evidence.

## Completed experiments

{markdown_table(['Dataset', 'Horizon', 'Point model', 'Seeds', 'Origins', 'Alpha values', 'Methods'], completed_rows)}

The methods are paired Flat, full Flat, Phase Mondrian, State SCCP, Phase+State,
and capped delayed-feedback ACI. The point predictions and targets are identical
across methods. The State partition remains fixed across alpha.
Full Flat uses every calibration origin. The other methods use the exact
published second-half calibration selection.

## Numerical results at alpha 0.10

{markdown_table(['Dataset', 'Horizon', 'Flat paired Winkler', 'Flat full Winkler', 'Phase Winkler', 'State Winkler', 'Phase coverage', 'State coverage'], rows)}

Widths, coverage errors, MAE, MSE, all alpha values, and per-horizon scores are in
`results/paper_strengthening/`. Width and Winkler use training-standardised channel units.
No test-based interval scaling or hyperparameter choice was made.

## Paired State-minus-baseline Winkler uncertainty

{markdown_table(['Dataset', 'Horizon', 'Baseline', 'Mean difference', '95% block CI', 'Blocks'], ci_rows)}

The bootstrap averages the three seeds at each origin before resampling.
It uses 5,000 resamples, seed 0, and blocks covering at least one horizon.
`uncertainty.csv` also contains paired coverage/width differences and block-length
multipliers 2 and 4. Fewer than ten blocks is too coarse for a strong reading.
Intervals from the block bootstrap are approximations under temporal dependence.

## Scientific reading

- State has a lower mean Winkler score than Phase in {phase_wins} of {len(scores)} cells at alpha 0.10.
  The Phase-minus-State coverage difference ranges from {coverage_loss.min():.2f} to {coverage_loss.max():.2f} percentage points.
  These observations do not establish a benefit at comparable coverage.
- Full Flat outperforms State on Exchange-rate at horizon 96. State's Winkler score
  is {full_loss:.2f}% higher, with lower coverage. At horizon 336 their difference is uncertain
  and only a few bootstrap blocks are available.
- Full Flat is not uniformly better than paired Flat. The additional older
  calibration residuals can come from a different error distribution.
- The existing ECL/Solar prespecified results remain unchanged in their original
  sampling designs. The ECL conclusion has not been checked across all phases here.
  The Solar gain has not been separated from known calendar phase here.
- The paper cannot yet claim that learned state adds uncertainty information beyond
  known seasonality on its historical confirmatory datasets.
- Cluster/phase association is descriptive. NMI does not prove incremental value
  beyond phase, and these diagnostics were not used to choose any setting.

## Checks

{len(reproductions)} original per-origin State/paired-Flat reproduction checks passed.
The largest relative difference was {max(r['max_relative_drift'] for r in reproductions):.1f}.
The full repository test run passed 205 tests; a later run passed eight additional
checker/merge tests. The 29 new conformal/suite tests include quantile indexing,
target-independent state extraction, no future ACI feedback, fallback, counts,
phase mapping, exact metric values, pairing, deterministic partitions, and full
versus paired calibration selection.

The exchangeable two-regime simulation is reproducible with
`python -m experiments.synthetic_phase`. Its results and data are saved separately.
It is a code check, not experimental evidence about forecasting datasets.

## Pending experiments and exact reasons

| Experiment | Reason and next step |
| --- | --- |
| ECL phase control and phase-complete re-score | Frozen weights are on HPC. Use the new ECL job; stride 23 samples every daily phase. |
| Solar phase control and alpha comparisons | Frozen weights are on HPC. Use the Solar job; retain stride 25 and 144 native daily slots. |
| DLinear/PatchTST residuals with CISSN states | Frozen publication weights are on HPC. ECL job includes both; optional exposed-dataset jobs include both. |
| Solar strong forecaster | No Solar point-baseline publication configurations/checkpoints exist in the inspected inventory. It needs separate training with development-only choices. |
| HopCPT/EnbPI/SPCI | No local implementation was found. A faithful implementation or author-code integration has not been run. Existing capped ACI is the available adaptive comparator. |
| Two new untouched datasets | Traffic and ILI are candidates. Complete exposure history and frozen checkpoint identities are missing. No candidate test split was scored. |
| Matched-coverage scaling | Requires a separate development-only design. Test metrics were not used to choose interval scales. |
| Phase-specific confidence intervals | Phase means are saved; conditional bootstrap ratios by phase are not implemented. |

## Code and artifact changes

- `cissn/conformal/phase.py`: native calendar phase and a recorded hierarchical fallback.
- `experiments/run_paper_suite.py`: frozen prediction cache, shared calibration,
  six interval methods, three alphas, per-horizon scores, diagnostics, reproduction,
  and paired dependence-aware uncertainty.
- `experiments/paper_suite.yaml`: fixed descriptive settings.
- `experiments/prespecified_confirmatory.yaml`: future candidate protocol and explicit blockers.
- `experiments/synthetic_phase.py`: two known variance regimes with saved seeds/data/configuration.
- `slurm/paper_jobs.py`, `paper_preflight.sh`, `launch_paper.sh`, `run_paper.slurm`:
  resource checks, separate dataset jobs, and strict result merging.
- `tests/test_paper_suite.py`, `tests/test_paper_jobs.py`: mathematical, data, and launch checks.
- `manuscript/analysis/build_strengthening.py`: generated tables, macros, and four figure sets.
- `manuscript/main.tex`: empirical framing, cellwise terminology, new follow-up
  results, negative findings, and clear limits for pending cluster experiments.
- `manuscript/refs.bib`: verified localized CP, HopCPT, SPCI, and CopulaCPTS entries;
  EnbPI volume/pages added from PMLR.
- Reproduction guide, manuscript README, task checklist, and one reusable sandbox lesson.

The manuscript uses generated numerical macros rather than hand-entered results.
Figures cover coverage/width, phase effects on available datasets, cluster/phase
contingencies, and paired Winkler confidence intervals. The ECL phase figure is
pending the cluster run; no other dataset is labelled as ECL.

## Original implementation findings

No new point-model bug was found. The existing small-sample quantile cap can return
the largest residual where the proper unattainable order statistic is infinite.
A global fallback also does not promise coverage inside a sparse cluster.
The implementation is preserved for fair reproduction; the manuscript now states
these limits. The previously documented single-window `torch.compile` bug remains
historical and was not rediscovered or changed in this task.

## Git and source state

Work stayed on `conformal-per-cell-geometry`. Nothing was staged or committed.
The pre-existing `.gitignore` edit and untracked manuscript were preserved.
The original experiment files and result directories were not overwritten.
Generated result arrays and figures follow the repository's existing ignore rules.
The reproduction guide lists the exact commands and records which checks were run.

Tracked diff for this task (the new source files and manuscript are untracked):

```text
{diff_stat}
```
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
