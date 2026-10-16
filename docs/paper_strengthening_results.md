# Paper strengthening: measured results and remaining work

The ECL/Solar and strong-forecaster runs are left for the user on the HPC cluster.
Run `bash slurm/paper_preflight.sh`, then the dry run and launcher documented in
[the reproduction guide](paper_strengthening_reproduction.md).

The local follow-up completed 18 frozen CISSN runs and 108 method/alpha
cell summaries (324 per-run method/alpha rows). It recorded 60 selected
runs as blocked by missing weights. All locally scored datasets were exposed before
this task. These results cannot supply new confirmatory evidence.

## Completed experiments

| Dataset | Horizon | Point model | Seeds | Origins | Alpha values | Methods |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | cissn | 3 | 2785 | 0.05, 0.1, 0.2 | 6 |
| ETTh2 | 336 | cissn | 3 | 2545 | 0.05, 0.1, 0.2 | 6 |
| exchange_rate | 96 | cissn | 3 | 1424 | 0.05, 0.1, 0.2 | 6 |
| exchange_rate | 336 | cissn | 3 | 1184 | 0.05, 0.1, 0.2 | 6 |
| weather | 96 | cissn | 3 | 10445 | 0.05, 0.1, 0.2 | 6 |
| weather | 336 | cissn | 3 | 10205 | 0.05, 0.1, 0.2 | 6 |

The methods are paired Flat, full Flat, Phase Mondrian, State SCCP, Phase+State,
and capped delayed-feedback ACI. The point predictions and targets are identical
across methods. The State partition remains fixed across alpha.
Full Flat uses every calibration origin. The other methods use the exact
published second-half calibration selection.

## Numerical results at alpha 0.10

| Dataset | Horizon | Flat paired Winkler | Flat full Winkler | Phase Winkler | State Winkler | Phase coverage | State coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | 4.985235 | 6.163206 | 5.328793 | 4.917879 | 0.927939 | 0.899268 |
| ETTh2 | 336 | 7.654900 | 9.238083 | 8.016188 | 7.485446 | 0.905384 | 0.892505 |
| exchange_rate | 96 | 2.475736 | 2.118160 | 2.504770 | 2.512307 | 0.922347 | 0.895484 |
| exchange_rate | 336 | 4.051847 | 3.904990 | 4.124290 | 3.924870 | 0.981921 | 0.939476 |
| weather | 96 | 1.987611 | 2.098203 | 2.215280 | 1.977590 | 0.953967 | 0.936478 |
| weather | 336 | 2.559448 | 2.711138 | 2.779154 | 2.555609 | 0.949473 | 0.932998 |

Widths, coverage errors, MAE, MSE, all alpha values, and per-horizon scores are in
`results/paper_strengthening/`. Width and Winkler use training-standardised channel units.
No test-based interval scaling or hyperparameter choice was made.

## Paired State-minus-baseline Winkler uncertainty

| Dataset | Horizon | Baseline | Mean difference | 95% block CI | Blocks |
| --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | flat_full | -1.245327 | [-1.597255, -0.786571] | 29.01 |
| ETTh2 | 96 | phase | -0.410915 | [-0.540154, -0.254233] | 29.01 |
| ETTh2 | 336 | flat_full | -1.752636 | [-2.404527, -1.298204] | 7.57 |
| ETTh2 | 336 | phase | -0.530741 | [-0.659502, -0.425621] | 7.57 |
| exchange_rate | 96 | flat_full | 0.394147 | [0.271245, 0.527140] | 14.83 |
| exchange_rate | 96 | phase | 0.007536 | [-0.151197, 0.135656] | 14.83 |
| exchange_rate | 336 | flat_full | 0.019880 | [-0.126176, 0.090852] | 3.52 |
| exchange_rate | 336 | phase | -0.199420 | [-0.349619, -0.161183] | 3.52 |
| weather | 96 | flat_full | -0.120613 | [-0.135800, -0.106710] | 108.80 |
| weather | 96 | phase | -0.237690 | [-0.267182, -0.203688] | 108.80 |
| weather | 336 | flat_full | -0.155529 | [-0.177218, -0.132882] | 30.37 |
| weather | 336 | phase | -0.223545 | [-0.274202, -0.171128] | 30.37 |

The bootstrap averages the three seeds at each origin before resampling.
It uses 5,000 resamples, seed 0, and blocks covering at least one horizon.
`uncertainty.csv` also contains paired coverage/width differences and block-length
multipliers 2 and 4. Fewer than ten blocks is too coarse for a strong reading.
Intervals from the block bootstrap are approximations under temporal dependence.

## Scientific reading

- State has a lower mean Winkler score than Phase in 5 of 6 cells at alpha 0.10.
  The Phase-minus-State coverage difference ranges from 1.29 to 4.24 percentage points.
  These observations do not establish a benefit at comparable coverage.
- Full Flat outperforms State on Exchange-rate at horizon 96. State's Winkler score
  is 18.61% higher, with lower coverage. At horizon 336 their difference is uncertain
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

36 original per-origin State/paired-Flat reproduction checks passed.
The largest relative difference was 0.0.
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
AGENT_LEARNINGS.md | 7 +++++++
 slurm/README.md    | 5 +++++
 2 files changed, 12 insertions(+)
```
