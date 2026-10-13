# Prespecification

Written 2026-10-13, on branch `conformal-per-cell-geometry`. `RUNBOOK.md`, "Prespecified
statistics", requires this file before any confirmatory test split is opened. It is a
record of choices, not a plan: `RUNBOOK.md` stays the protocol, and this file adds the
statistical analysis and one extra sealed dataset to it.

## 1. What was already seen, stated before any choice

- The RevIN CISSN grid on ETTh2, weather, and exchange-rate ran on 2026-08-31 and
  2026-09-01 at commit `fb5231d`, before this file existed. No aggregate metric from those
  runs was read while writing this file; only run names, file lists, `metrics.json` key
  names, and the configuration files. One exception: to check array shapes, the first two
  per-origin rows (primary and flat-CP scores) of one run (ETTh2, horizon 96, seed 42) were
  printed, and the primary Winkler score was lower in both. Two of 2,785 origins say
  nothing about an endpoint on ECL, but the exposure is recorded.
- Before that, a pre-RevIN grid evaluated the same three test splits, and its results
  were read: flat CP beat the primary mechanism on all four datasets in that regime
  (`results/superseded/pre_revin_scalar_grid/`).
- Two one-epoch ECL smoke runs (legacy and hybrid) went through `run_selection.py`, which
  never builds the test loader, to check memory and time. They scored the validation split
  only, wrote to scratch directories, and were deleted. No ECL test window has been
  evaluated.
- ETTh1 was the development dataset. Its test split is exposed. Only validation numbers
  from ETTh1 selected anything.

| Dataset | Test split status | Role |
| --- | --- | --- |
| ETTh1 | exposed, development | diagnostic only; excluded from every endpoint |
| ETTh2, weather, exchange-rate | exposed to the pre-RevIN grid; RevIN grid unread | secondary, descriptive |
| ECL (electricity) | never evaluated | **confirmatory** |
| traffic, ILI, solar | never evaluated | not used; see section 9 |

ETTm1 and ETTm2 are not used. They cover the same dates and the same transformers as
ETTh1 and ETTh2 at 15-minute resolution, so they are not independent test data.

## 2. Decisions already taken on validation data

| Step | Rule (`RUNBOOK.md`) | Outcome | Computed |
| --- | --- | --- | --- |
| 3b.2 conditioning | promote `scale` only if it beats flat CP and cluster SCCP on at least 2 of 3 seeds | beats both on 1 of 3. Primary stays `cluster`, `--scale_geometry scalar` | 2026-10-13 from `results/selection/*scalecond*/selection.json`, files dated 2026-08-28 to 2026-08-31 |
| 5 hybrid | at least 1% better than paired DLinear, 2 of 3 seeds, none worse than 2%, no feature worse than 5% | `hybrid_anchored`: -3.69% mean, 3 of 3, worst seed -2.81%, worst feature +0.73% | 2026-10-13; `results/selection/hybrid_feature_mse.json` |
| 4 RevIN | no regression in validation MSE or coverage on ETTh2 and weather | pending. Decides only whether baselines rerun under RevIN. Cannot change any endpoint below | |

## 3. Confirmatory design

- Dataset: ECL, 321 channels, `seq_len` 96, chronological split from the registry
  (14,299 train, 2,295 validation, 4,927 test windows at horizon 336).
- Horizons: 96 and 336. Seeds: 42, 123, 456.
- Model: CISSN, `--revin`, `--conformal_conditioning cluster`, `--scale_geometry scalar`,
  `per_feature` intervals, `alpha` 0.1, `n_clusters` 5, 20 epochs, patience 5, cosine
  schedule, batch size 128. These are the settings of the exposed-dataset grid. No tuning
  on ECL.
- `--test_origin_stride 24` (one origin per day, equal to the seasonal period). Stride 1
  would write about 10 GB per run at horizon 336 and gives windows that share most of
  their targets. The stride thins evaluation only; training and calibration are unchanged.
- Sealed run flags: `--evidence_role confirmation --immutable_artifacts --strict_determinism
  --require_gpu --require_clean_git`.
- A run whose `structural_passed` is false is excluded and rerun once at the same seed. No
  run is excluded for poor forecast quality.

## 4. Primary endpoint

One endpoint: ECL, horizon 336, per-origin Winkler score of cluster SCCP minus flat CP,
with flat CP calibrated on the second calibration half (`interval_flat_cp`).

- Per origin, average the Winkler score over the three seeds, then take the difference.
  Seeds share test origins, so the rows pair exactly.
- Divide by the mean flat-CP Winkler score to get a relative effect.
- 95% percentile CI by moving-block bootstrap: block length `ceil(336 / 24)` = 14 origins
  (one horizon), 5,000 resamples, seed 0. Command in section 8.
- Reading, fixed now. Let the CI of the relative effect be [L, U]. Exactly one outcome
  applies, checked in this order:
  1. **Better, beyond the margin**: U < -1%.
  2. **Better, within the margin**: U < 0 and not outcome 1.
  3. **Worse**: L > 0.
  4. **No material difference**: not 1, 2, or 3, and the whole CI lies inside -1% to +1%.
  5. **Inconclusive**: everything else.
  Outcomes 1 and 2 both support the words "better than flat CP"; only outcome 1 supports
  the words "meaningfully better". The 1% margin is a practical threshold chosen from the
  size of the gains seen so far (1% to 6%); it is not a power calculation.
- This is the only result an abstract may call confirmatory. Horizon 336 is the primary
  horizon because every selection used it. That makes it a familiar horizon, not a neutral
  one; on ECL it is untouched.

## 5. Secondary endpoints (descriptive; no claim of confirmation)

1. ECL horizon 96: the same statistic as section 4.
2. Exposed grid, 13 cells (ETTh2 5 horizons, weather 4, exchange-rate 4), stride 1, block
   length one horizon: per-cell relative effect with CI, and a two-sided Wilcoxon
   signed-rank test over the 13 cell means. Horizons within a dataset share data, so the
   number of independent units is nearer 3 than 13. Report per-dataset means beside the
   test and do not describe it as a 13-sample result.
3. Marginal coverage per cell with a block-bootstrap CI, and `conditional_coverage`
   (`worst_slab_coverage`, `max_coverage_deviation`) from `metrics.json`. State a
   marginal-coverage claim only when the CI lies inside 0.87 to 0.93. ETTh1 validation
   coverage was 0.74 to 0.82 for every method, so coverage claims stay relative.
4. Cluster SCCP against per-cell state-scaled CP cannot be tested from the grid:
   the grid's secondary mechanism is `scalar`, which the diagnostics show cannot express
   the signal, and it saves no per-origin scores. This stays a diagnostic question
   (`scripts/diagnose_window_placement.py`).

Per-cell "N of 3 seeds" counts are descriptive. With 3 seeds they have a false-positive
rate near 50% per cell under no effect.

## 6. Point forecasts (reported separately from intervals)

- CISSN against DLinear, PatchTST, DeepState, MC-Dropout, Deep Ensemble: compare only
  within one regime. If Step 4 selects RevIN, the four non-DeepState baselines rerun with
  `--revin` and the non-RevIN copies move to `results/superseded/baselines_no_revin/`.
- Hybrid: confirmatory check of `hybrid_anchored` on ECL horizon 336, 3 seeds. Each run
  is compared with its own frozen DLinear base, as in Step 5: `scripts/paired_point_comparison.py`
  reloads the checkpoint and scores `base` and `total` on the same evaluated test origins.
  It stops if `total` does not reproduce the saved `pred.npy`. Report the per-seed test-MSE
  ratio and per-feature changes. With 3 seeds, state the count of seeds better, not a
  p-value. A prior hybrid run improved validation by 4.1% and lost on test
  (`docs/methodology.md`), so a validation gain alone supports no point-accuracy claim. No
  separately trained DLinear run is part of this check.
- Five-state CISSN is an interpretability and calibration design with a measured accuracy
  cost. No competitive point-forecast claim is prespecified.

## 7. Claims allowed and not allowed

Allowed, in this scope: relative Winkler differences between mechanisms on the same
residuals, with CIs that account for window overlap. Not allowed: unconditional
time-series coverage; a marginal 0.90 claim without the section 5 rule; physical meaning
of state coordinates ("level-like" and "trend-like" only); any statement about the
exposed datasets as untouched; any conditioning result without the RevIN regime stated.

## 8. Analysis settings

Fixed values for `scripts/paired_interval_inference.py`: `--block_multiplier 1`,
`--n_boot 5000`, `--seed 0`, 95% confidence. Run it once on `--datasets ECL` (primary and
secondary 1; read the primary endpoint from the horizon-336 row only) and once on
`--datasets ETTh2 weather exchange_rate` (secondary 2). The hybrid check of section 6 is
`scripts/paired_point_comparison.py`. The exact commands are in `experiments.md`, block R7.

## 9. Not prespecified, and why

- traffic (862 channels), solar, and ILI (966 weekly rows, about 100 calibration windows)
  are untouched but add cost without changing the confirmatory design. One untouched
  dataset with 321 channels and thousands of origins carries the primary endpoint.
- Hyperparameter changes. None is allowed after this date for the confirmatory runs.

## 10. Which runs this file covers, and deviations

This file covers an ECL run only if its `environment.json` records a `git_commit` that
contains this file, with `git_dirty` false. Any other ECL run is outside the prespecification.

Deviations:

Any change to this file after the first ECL test split is opened is recorded below with
its date and reason. The primary endpoint, horizon, bootstrap parameters, and reading
rule do not change.

(no deviations recorded)
