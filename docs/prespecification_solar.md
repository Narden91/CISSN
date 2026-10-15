# Prespecification: solar, second confirmatory dataset

**Status: draft, written 2026-10-15.** It takes effect when it is committed. Commit it before any
P3 run: `slurm/preflight.sh` refuses to launch a P3 block while it is missing from `HEAD`, and a P3
run counts as confirmatory only if its `environment.json` records a commit that contains this file,
with `git_dirty` false. `RUNBOOK.md` Step 7 stays the protocol; `docs/prespecification.md` stays
unchanged and still governs ECL.

This file replaces the traffic draft of commit `ef84e5e`. No traffic or solar run existed at the
change (section 8).

## 1. What was already seen, stated before any choice

- **solar.** No model has been scored on a solar test window. To write this file, these were read:
  the file shape (137 channels, 52,560 rows), the split lengths in section 2, and, on the first 70%
  of the rows only (the training span), the share of zero entries (53%) and the number of
  zero-variance channels (0). One local run trained one epoch on solar through
  `run_selection.py` with `--scale_geometry per_cell`, which scores the validation split only and
  never builds the test loader. It checked that the per-cell path works on zero-inflated data. It
  ended with exit code 0 and all values finite; no validation number was read or used. Slurm block
  `p3s` repeats the check on the cluster.
- **ECL result, known.** Cluster SCCP against flat CP at horizon 336: -1.07% (95% CI -1.57% to
  -0.51%), read as "better, within the margin". At horizon 96: -2.05% (95% CI -2.47% to -1.66%).
  This result motivates the replication and selects no setting below.
- **Per-cell state-scaled CP, known.** A development diagnostic on the ETTh1 test split, which
  reuses test data, found -6.4% against flat CP. The validation selection (Step 3b.2) did not
  separate it from the other calibrators (it beat both on 1 of 3 seeds). Neither number is
  evidence; together they are the reason for the second endpoint.
- **Phase dependence, known.** On ETTh2 at horizon 336 (an exposed dataset, stride 1, three
  seeds), the relative Winkler change of cluster SCCP against flat CP depends on the origin's phase
  (origin index mod 24): from -8.2% to +7.6% (all phases: -2.2%; phase 0 alone: +7.6%). At horizon
  96 the range is -5.6% to +4.0%. This is the reason for the test-origin stride in section 2.
- **Exposed datasets.** ETTh1, ETTh2, weather, exchange-rate, and ECL test splits are open. None
  of them enters an endpoint here.

## 2. Confirmatory design

- Dataset: solar, 137 channels, 10-minute data, `seq_len` 96, chronological split from the
  registry, seasonal period 144 steps (one day). At horizon 336: 29,003 train, 4,921 validation,
  7,023 calibration (about 3,500 per half), 10,177 test windows. At horizon 96: 29,243, 5,161,
  7,263 (about 3,600 per half), 10,417.
- Horizons: 336 and 96. Seeds: 42, 123, 456.
- Model and calibration: the ECL settings (`docs/prespecification.md`, section 3) with one change,
  `--scale_geometry per_cell`. That flag sets only the state-scaled calibrator's sigma shape. It
  does not change training, cluster SCCP, or flat CP. No tuning on solar.
- **Test origins: `--test_origin_stride 25`.** 25 shares no factor with the daily period of 144, so
  consecutive origins cycle through every time of day, and the test set covers the phases that the
  calibration windows cover. A stride equal to the period would fix one phase, and section 1 shows
  that one phase can move the result by several points. Origins: 408 at horizon 336, 417 at horizon
  96. Training and calibration use every window.
- Bootstrap block length: `ceil(H / 25)` origins, which is 14 at horizon 336 (350 steps, at least
  one horizon). The test split holds 29.1 blocks at horizon 336 and 104 at horizon 96.
- Sealed run flags: `--evidence_role confirmation --immutable_artifacts --strict_determinism
  --require_gpu --require_clean_git`.
- A run whose `structural_passed` is false is excluded and rerun once at the same seed. No run is
  excluded for poor forecast quality.

## 3. Endpoints

Both endpoints use the statistic of `docs/prespecification.md`, section 4: per origin, average the
Winkler score over the three seeds, take the difference against flat CP (calibrated on the second
calibration half), divide by the mean flat-CP score, and get a percentile CI by moving-block
bootstrap with 5,000 resamples and seed 0.

- **E1, primary.** solar, horizon 336, cluster SCCP minus flat CP.
- **E2, key secondary.** solar, horizon 336, per-cell state-scaled CP minus flat CP.

**Multiplicity (Holm, two endpoints).** The claim "better than flat CP" is tested for E1 and E2
together:

1. Compute the relative effect of both endpoints with a 97.5% CI.
2. If the upper bound of one endpoint lies below 0, that endpoint supports "better". The other
   endpoint is then read with its 95% CI.
3. If neither upper bound lies below 0 at 97.5%, both endpoints are read with their 97.5% CI.

**Reading.** Each endpoint gets exactly one outcome from the five-outcome rule of
`docs/prespecification.md`, section 4 (margin 1%), applied to the CI that the Holm steps assign to
it. The words "replicated on a second untouched dataset" are allowed only if E1 reads "better,
beyond the margin" or "better, within the margin".

## 4. Secondary analyses (descriptive; no claim of confirmation)

1. Horizon 96: the E1 and E2 statistics with 95% CIs. No reading rule.
2. Marginal coverage and `conditional_coverage` from `metrics.json`, as in
   `docs/prespecification.md`, section 5.3.
3. Effect by time of day: the E1 statistic in six bins of origin index mod 144 (24 steps each).
   This shows whether the effect depends on phase on solar.
4. ACI against flat CP (`scripts/aci_paired_comparison.py`, `--gamma 0.005`, block multiplier 1,
   5,000 resamples, seed 0). ACI uses realised test errors; report it as a calibrator with more
   information.

## 5. Analysis settings

`scripts/paired_interval_inference.py --datasets solar --block_multiplier 1 --n_boot 5000 --seed 0`,
once per `--method {primary, state_scaled}` and per `--confidence {0.975, 0.95}`. Read E1 from
`--method primary` and E2 from `--method state_scaled`, horizon-336 rows only. The exact commands
are in `experiments.md`, section 6.

## 6. Claims allowed and not allowed

The rules of `docs/prespecification.md`, section 7, apply. In addition: do not pool solar with ECL
into one estimate. Report the two E1 results side by side, each with its own CI and block count.
The two datasets also differ in test-origin design (ECL: one phase; solar: all phases), so state
that next to any comparison. E2 has no ECL counterpart, because ECL ran with scalar geometry.

## 7. Not prespecified

- Cluster SCCP against per-cell state-scaled CP directly. The run saves both per-origin files, so
  it can be reported as exploratory only.
- Baselines and the hybrid model on solar.
- Any hyperparameter change on solar.
- A re-score of the ECL checkpoints at a stride that covers every phase. If it is done, it is
  post-hoc and reported beside the prespecified ECL result.

## 8. Deviations

Any change to this file after the first solar test window is evaluated is recorded below with its
date and reason. E1, E2, the horizon, the bootstrap settings, the Holm steps, and the reading rule
do not change.

1. 2026-10-15, dataset and stride changed from the traffic draft (`ef84e5e`) before any traffic or
   solar run. Reasons: traffic at horizon 336 gives only 9.5 bootstrap blocks, which the manuscript
   calls too coarse to read (solar gives 29.1), and a stride of 24 on hourly data fixes one phase
   (section 1). The endpoints, the Holm steps, and the reading rule are unchanged.
