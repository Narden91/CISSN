# CISSN repository guide

## Scope

CISSN forecasts multivariate time series with a structured five-dimensional latent state and state-conditional conformal intervals. The executable publication protocol is `RUNBOOK.md`; do not create alternative experiment plans.

## What is novel here vs. reused

State this explicitly in any paper draft, PR description, or summary of this work — do not let it stay implicit:

- **Novel**: state-conditional interval calibration via a continuous, learned difficulty estimator (`StateScaledConformal` in `cissn/conformal/state_conditional.py`) — a log-linear regression of residual scale on the five-dimensional latent state gives `sigma(s)`, and conformal quantiles are calibrated on residuals normalized by that scale (Papadopoulos-style normalized conformal, with the learned structured state as the difficulty estimator). Fit on train states/residuals before calibration and never refit on calibration/test data. This is the paper's primary contribution.
- **Ablation, demonstrating the pivot**: `StateConditionalConformal`, which discretizes the same state into K-Means clusters and calibrates one quantile per cluster. Measured on this repository's development runs, cluster membership explains only a fraction of the state's relationship to residual scale relative to the continuous state (R^2 ~0.17 vs ~0.73 on a representative ETTh1-h336 diagnostic), and the discrete predictor's win over flat CP is a smaller, less consistent effect than the continuous predictor's. Report both: the ablation is part of the evidence for the pivot, not dead code.
- **Novel, secondary**: the `DisentangledStateEncoder`'s structured block-diagonal transition (level/trend/seasonal-rotation/residual) with a disentanglement penalty, and the `HybridCISSN` architecture that adds this state as an additive correction on top of a frozen DLinear base rather than routing the whole forecast through it.
- **Reused, not novel**: DLinear (Zeng et al., AAAI 2023) as base forecaster/baseline; split conformal prediction's core exact-order-statistic quantile mechanism (standard, e.g. Lei et al.); normalized/locally-weighted conformal scores (Papadopoulos et al.) as the general mechanism state-scaled conformal instantiates; MC-Dropout, Deep Ensemble, DeepState, PatchTST as baseline comparators; MSIS, Winkler score, PICP as standard interval metrics.
- Do not claim unconditional time-series coverage — split conformal's exchangeability assumption is violated by overlapping windows, and this is explicitly not fixed by ACF-based inflation (see `docs/methodology.md`). Do not claim `state coordinates are physically identified` (e.g. "this is literally trend") unless synthetic recovery and stability checks pass; use "level-like", "trend-like" language otherwise.
- When reporting results, distinguish the interval-calibration contribution (state-scaled conformal, with cluster-based SCCP as the ablation) from the point-forecast architecture (hybrid/legacy encoder) — a win on one does not imply a win on the other, and the two must be evaluated and stated separately.

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
- `--conformal_conditioning {cluster,scale}` selects the primary interval mechanism; default is `cluster` (unchanged run-directory names) until the RUNBOOK's dev check confirms `scale` end-to-end, after which `scale` is intended to become the default (see RUNBOOK.md). Every run calibrates and reports **both**, plus flat CP, regardless of which is primary: `metrics.json` always contains `interval` (primary), `interval_flat_cp`, `interval_cluster_cp`, and `interval_state_scaled` under fixed, mode-tagged keys. Never compare one conditioning mechanism against a flat CP or a competing mechanism fitted in a separate training run.
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
