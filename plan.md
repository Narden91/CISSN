# CISSN Experiment And Publication Readiness Plan

## Summary

Prepare CISSN for defensible experiments and publication by fixing protocol validity first, then hardening data, conformal calibration, experiment orchestration, artifacts, tests, and publication claims.

The canonical execution plan for the first Q1 journal submission now lives in `document.md`. This file is kept as the repo-readiness summary rather than the runnable experiment checklist.

Current baseline confirmed before implementation:
- Existing 13 unit tests passed.
- All benchmark files were present.
- Solar failed to load because `solar_AL.txt` had no `date` column.
- The worktree was already dirty before this pass.

## Implemented In This Pass

- Added a canonical dataset registry in `cissn/data/registry.py`.
- Added a distinct chronological `cal` split for conformal calibration.
- Added a Solar-specific loader that creates a deterministic 10-minute datetime index and stable channel names.
- Made `MS` target ordering explicit so target slicing does not depend on CSV column order.
- Hardened SCCP:
  - respects `n_clusters=1`;
  - never fits more clusters than samples;
  - resets fitted state on every `fit()`;
  - handles constant residual ACF without `nan`;
  - exposes cluster assignment and JSON-serializable cluster diagnostics;
  - marks `max` as simultaneous coverage and other strategies as marginal;
  - warns that `mahalanobis` is experimental.
- Fixed the benchmark protocol:
  - no test-set evaluation during training;
  - calibration uses the separate `cal` split;
  - best early-stopped checkpoint is reloaded before calibration;
  - conformal alpha and cluster count are CLI/configurable;
  - metric alpha matches conformal alpha;
  - run names include dataset, feature mode, horizon, seed, alpha, strategy, and key model settings.
- Added experiment artifacts:
  - `metrics.json`, `config.json`, `environment.json`, `cluster_stats.json`, `coverage_by_cluster.json`, `runtime.json`;
  - predictions, targets, intervals, states, residuals, and cluster labels.
- Added YAML config support with examples under `experiments/configs/`.
- Fixed multi-seed orchestration to avoid run overwrites and emit JSON plus raw CSV aggregation.
- Fixed ablation toggles so 5D `no_structured_A` and `no_correction_mlp` actually change the encoder.
- Added shared baseline training/evaluation helpers for parity across baselines.
- Added `experiments/run_baseline.py` for `dlinear`, `patchtst`, `deepstate`, `mc_dropout`, and `deep_ensemble`, with smoke-validated artifact parity.
- Revised publication-facing claims so coverage assumptions, ACF limitations, and unimplemented soft fallback are no longer overstated.

## Remaining Publication Work

- Execute the locked Q1 journal experiment grids in `document.md`.
- Run full smoke, pilot, and benchmark grids after tests pass.
- Add table-generation scripts for main results, ablations, and calibration results.
- Generate analysis figures: state PCA/t-SNE, reliability curves, interval width vs. state, contribution heatmaps, and runtime scaling.
- Validate the revised theory empirically: cluster-size sensitivity, ACF sensitivity, distribution shift, noise robustness, and missing-data robustness.
- Add a reproducibility appendix with exact commands, configs, seed list, hardware, raw result paths, and environment snapshots.

## Acceptance Criteria

- No training loop reads the test split before final evaluation.
- Every conformal interval is calibrated on a split disjoint from validation and test.
- Every reported run has deterministic paths and saved configs/artifacts.
- Solar either loads successfully through the new loader or fails with a clear data-specific error.
- Ablations modify the intended architectural component.
- Publication claims state assumptions explicitly and do not imply unconditional guarantees.
