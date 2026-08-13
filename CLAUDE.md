# CISSN repository guide

## Scope

CISSN forecasts multivariate time series with a structured five-dimensional latent state and state-conditional conformal intervals. The executable publication protocol is `RUNBOOK.md`; do not create alternative experiment plans.

## What is novel here vs. reused

State this explicitly in any paper draft, PR description, or summary of this work — do not let it stay implicit:

- **Novel, and regime-dependent**: state-conditional interval calibration. The conditioning signal is a **state x cell interaction** — the state changes *which horizon-feature cells are hard*, not how hard a window is overall. Per-cluster quantile *shape* deviates by ~0.20-0.26 while per-cluster *level* varies by only ~0.03, and the per-sample axis holds under 1.2% of residual variance against ~19% on the per-cell axis. Consequences: a **scalar** `sigma(s)` cannot express the signal (measured +0.011 Winkler vs flat CP, 5/12 seed-cut wins — no improvement, and a scalar *oracle* using test labels gains under 1%), whereas `StateScaledConformal(scale_geometry='per_cell')`, one log-linear regression per cell, reaches -0.237 (12/12) and cluster SCCP reaches -0.124 (12/12). All of this holds **only in the RevIN regime**; on a pre-RevIN (amplitude-collapsed) run every mechanism loses to flat CP. Never state a conditioning result without the regime.
- **Currently strongest under the real protocol**: `StateConditionalConformal` (cluster). On ETTh1-h336/seed 42/RevIN, paired on identical residuals: flat 3.7869, state-scaled scalar 3.7877, state-scaled per-cell 3.6916, cluster 3.5962. Per-cell state-scaled beats flat CP and the scalar geometry; it does not beat cluster SCCP. Report both, and do not call either mechanism the paper's established primary contribution on ETTh1 alone.
- **Known limitation, measured**: the protocol fits `sigma(s)` on *train* residuals the forecaster was trained to shrink. Per-cell coefficient spread is 0.076 fit in-sample vs 0.137 fit out-of-sample — the in-sample fit sees ~half the true state-to-cell coupling, which is why the protocol result trails the diagnostic. Fixing it means giving sigma its own held-out fitting window; that is a split-contract change, not a tuning change.
- `--scale_geometry {scalar,per_cell}` selects the state-scaled predictor's sigma shape. Default is `scalar` (unchanged behaviour and artifacts); `per_cell` is the geometry with measured headroom. Ignored under `--multivariate_strategy max`.
- Any comparison between conditioning mechanisms must hold the calibration window fixed. A method calibrated on more data than its comparator is not evidence about conditioning; this exact error produced the retracted result above and is locked by `TestConditioningComparisonFairness` in `tests/test_utils.py`.
- **Novel, secondary**: the `DisentangledStateEncoder`'s structured block-diagonal transition (level/trend/seasonal-rotation/residual) with a disentanglement penalty, and the `HybridCISSN` architecture that adds this state as an additive correction on top of a frozen DLinear base rather than routing the whole forecast through it.
- **Reused, not novel**: DLinear (Zeng et al., AAAI 2023) as base forecaster/baseline; split conformal prediction's core exact-order-statistic quantile mechanism (standard, e.g. Lei et al.); normalized/locally-weighted conformal scores (Papadopoulos et al.) as the general mechanism state-scaled conformal instantiates; MC-Dropout, Deep Ensemble, DeepState, PatchTST as baseline comparators; MSIS, Winkler score, PICP as standard interval metrics.
- Do not claim unconditional time-series coverage — split conformal's exchangeability assumption is violated by overlapping windows, and this is explicitly not fixed by ACF-based inflation (see `docs/methodology.md`). Do not claim `state coordinates are physically identified` (e.g. "this is literally trend") unless synthetic recovery and stability checks pass; use "level-like", "trend-like" language otherwise.
- When reporting results, distinguish the interval-calibration results (flat CP, cluster SCCP, state-scaled CP) from the point-forecast architecture (hybrid/legacy encoder) — a win on one does not imply a win on the other, and the two must be evaluated and stated separately.

## Commands

```powershell
uv run python tests/run_tests.py
uv run python scripts/verify_datasets.py
uv run python experiments/run_benchmark.py --help
uv run python experiments/run_baseline.py --help
```

Use `--require_gpu --require_clean_git` for final publication runs. Review each
saved artifact-review file (`sanity.json` on disk — see wording note below)
before including its result in publication tables: exclude a run only when
`structural_passed` is false, never for poor forecast quality.

## Core contracts

- Data splits are chronological: train, calibration, validation, test. Calibration is a tail of the canonical train window; validation selects checkpoints; test is evaluation only.
- `StateConditionalConformal.fit_partition(train_states)` must precede `calibrate(cal_states, residuals)`; `StateScaledConformal.fit_scale(train_states, train_residuals)` must precede its own `calibrate(cal_states, residuals)`. Do not fit state clustering or the state-scaled sigma(s) regression on calibration or test data.
- `--conformal_conditioning {cluster,scale}` selects the primary interval mechanism; default is `cluster` (unchanged run-directory names). `scale` is **not** slated to become the default — the evidence that motivated that plan was withdrawn (see the novelty section above and `docs/methodology.md`); promoting it now requires new evidence from RUNBOOK Step 3b, not the prior diagnostic. Every run calibrates and reports **both**, plus flat CP, regardless of which is primary: `metrics.json` always contains `interval` (primary), `interval_flat_cp`, `interval_cluster_cp`, and `interval_state_scaled` under fixed, mode-tagged keys. Never compare one conditioning mechanism against a flat CP or a competing mechanism fitted in a separate training run.
- Default score geometry is `per_feature`; its primary coverage is marginal. `max` is simultaneous and must use `coverage_primary`, not marginal coverage, for calibration claims.
- Conditional-coverage claims (as opposed to marginal PICP) require `conditional_coverage.worst_slab_coverage` / `max_coverage_deviation` in `metrics.json`, computed on prespecified, method-agnostic bins fit on train data (`fit_coverage_bin_edges`/`conditional_coverage_by_bin` in `cissn/evaluation/metrics.py`) — never on each method's own partition, which would make the comparison unfair.
- Temporal dependence is diagnostic evidence only. Never add ACF-based interval inflation or claim unconditional time-series coverage.
- DLinear uses replicate endpoint padding, matching its reference moving-average decomposition.
- MC-Dropout and Deep Ensemble are conformalized on the calibration split by default. Raw UQ intervals must be labelled `raw_uq`.
- Each final result must contain `metrics.json`, `sanity.json`, `history.json`, `runtime.json`, `config.json`, `environment.json`, and `protocol.json`.
- `sanity.json` (the artifact-review file) separates structural validity from forecast quality. Only structural failure (empty, non-finite, shape-inconsistent, inverted bounds) excludes a run from publication and fails `--strict_artifacts`; quality flags are advisory. Quality references are computed from the training split at the run's own forecast horizon — never from test statistics.
- `--architecture hybrid` freezes a trained DLinear base and adds a zero-initialised five-state correction, so correction-stage epoch 0 equals the base exactly and a failed correction stage falls back to it. Legacy is the default and its run directory names are unchanged.
- Every benchmark run calibrates flat CP, cluster SCCP, and state-scaled CP on the same residuals from the same model and scores all three on the same forecasts. This is the paired evidence for state conditioning; never compare any conditioning mechanism against a comparator fitted in a separate training run.
- CISSN's five-state constraint caps forecast rank at 5 while DLinear's own forecasts need rank ~96 on ETTh1-h336. The resulting point-accuracy gap is architectural, not a tuning failure — regularisation and removing the refinement MLP both make it worse. Present the five-state design as an interpretability/calibration choice with a measured accuracy cost, never as a competitive point forecaster.
- On ETTh1, validation improvements need not transfer to test (splits are non-stationary and differ from each other). Do not promote an architecture on a single-seed validation delta.
- `coverage_joint` is near zero under `per_feature` by construction. Simultaneous claims require `--multivariate_strategy max`, which currently reaches only ~0.49 joint against nominal 0.90; report that as a negative result rather than a guarantee.
- Every run records `var(pred)/var(true)` per epoch and on test. A low ratio means the forecast collapsed toward the mean (amplitude), which is a different failure from an information bottleneck (rank) and needs a different fix — do not conflate them. See `docs/methodology.md`.
- `--revin` applies reversible instance normalisation to the whole model; it is opt-in and off by default. RevIN statistics come only from the input window, never the target. Under `--features MS` the forecast must be denormalised with `select_channels()` so the target's own statistics are used, not feature 0's.

## Wording

Do not use "gate" or "sanity" in prose written in this repo (commit messages, PR descriptions, docstrings, comments, docs) — the words persist as filename/API surface (`sanity.json`, `check_forecast_sanity`) for compatibility, but new prose should say what the mechanism does instead:
- Instead of "gate"/"gates publication": say "excludes a run from publication" or "acceptance criteria" or name the specific check (e.g. "structural validity check").
- Instead of "sanity check"/"sanity"/"gate" as a verb for the pass/fail review: say "artifact review", "structural validity check", or "forecast quality review" depending on which of the two it is (see `cissn/evaluation/sanity.py` docstring for the split).
- Do not rename `sanity.json`, `check_forecast_sanity`, `--strict_sanity` (deprecated alias), or other existing identifiers to satisfy this — that would break saved artifacts and callers. This rule governs new prose, not existing API/file names.

## Layout

- `cissn/models/`: encoder and forecast head.
- `cissn/conformal/`: state-conditional split conformal predictor.
- `cissn/baselines/`: implemented comparators and Flat CP.
- `cissn/data/`: canonical registry, integrity checks, datasets, loaders.
- `experiments/`: benchmark, baseline, ablation, and multi-seed runners.
- `scripts/`: verification and publication artifact generators.
- `tests/`: unit and runner contracts.
- `docs/`: current architecture, dataset, methodology, and execution documentation.

## Editing rules

Keep changes small, typed where practical, and covered by tests. Preserve chronological separation and output artifact compatibility. Never alter a reported result in place; rerun it under a new protocol hash.
