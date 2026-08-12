# CISSN repository guide

## Scope

CISSN forecasts multivariate time series with a structured five-dimensional latent state and state-conditional conformal intervals. The executable publication protocol is `RUNBOOK.md`; do not create alternative experiment plans.

## Commands

```powershell
uv run python tests/run_tests.py
uv run python scripts/verify_datasets.py
uv run python experiments/run_benchmark.py --help
uv run python experiments/run_baseline.py --help
```

Use `--require_gpu --require_clean_git` for final publication runs. Review each
saved `sanity.json` before including its result in publication tables.

## Core contracts

- Data splits are chronological: train, calibration, validation, test. Calibration is a tail of the canonical train window; validation selects checkpoints; test is evaluation only.
- `StateConditionalConformal.fit_partition(train_states)` must precede `calibrate(cal_states, residuals)`. Do not fit state clustering on calibration or test data.
- Default score geometry is `per_feature`; its primary coverage is marginal. `max` is simultaneous and must use `coverage_primary`, not marginal coverage, for calibration claims.
- Temporal dependence is diagnostic evidence only. Never add ACF-based interval inflation or claim unconditional time-series coverage.
- DLinear uses replicate endpoint padding, matching its reference moving-average decomposition.
- MC-Dropout and Deep Ensemble are conformalized on the calibration split by default. Raw UQ intervals must be labelled `raw_uq`.
- Each final result must contain `metrics.json`, `sanity.json`, `history.json`, `runtime.json`, `config.json`, `environment.json`, and `protocol.json`.

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
