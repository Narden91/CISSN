# Prespecification: traffic, second confirmatory dataset

**Status: draft, written 2026-10-15.** It takes effect when it is committed. Commit it
before any P3 run: `slurm/preflight.sh` refuses to launch while it is missing from `HEAD`,
and a P3 run counts as confirmatory only if its `environment.json` records a commit that
contains this file, with `git_dirty` false. `RUNBOOK.md` Step 7 stays the protocol;
`docs/prespecification.md` stays unchanged and still governs ECL.

## 1. What was already seen, stated before any choice

- **traffic.** No model has run on it, and no test window has been evaluated. To write this
  file, the file shape (862 channels, 17,544 rows) and the split lengths in section 2 were
  read; no value was printed. Before P3, Slurm block `p3s` trains one epoch on traffic
  through `run_selection.py`, which never builds the test loader, to check memory. It scores
  the validation split only, and no number from it selects anything.
- **ECL result, known.** Cluster SCCP against flat CP at horizon 336: -1.07% (95% CI
  -1.57% to -0.51%), read as "better, within the margin". At horizon 96: -2.05% (95% CI
  -2.47% to -1.66%). This result motivates the replication; it selects no setting below.
- **Per-cell state-scaled CP, known.** A development diagnostic on the ETTh1 test split,
  which reuses test data, found -6.4% against flat CP. The validation selection (Step 3b.2)
  did not separate it from the other calibrators (it beat both on 1 of 3 seeds). Neither
  number is evidence; together they are the reason for the secondary endpoint.
- **Exposed datasets.** ETTh1, ETTh2, weather, exchange-rate, and ECL test splits are
  open. None of them enters an endpoint here.

## 2. Confirmatory design

- Dataset: traffic, 862 channels, `seq_len` 96, chronological split from the registry. At
  horizon 336: 9,393 train, 1,419 validation, 2,121 calibration (1,060 per half), 3,175
  test windows. At horizon 96: 9,633, 1,659, 2,361 (1,180 per half), 3,415.
- Horizons: 336 and 96. Seeds: 42, 123, 456.
- Model and calibration: the ECL settings (`docs/prespecification.md`, section 3) with one
  change, `--scale_geometry per_cell`. That flag sets only the state-scaled calibrator's
  sigma shape. It does not change training, cluster SCCP, or flat CP. No tuning on traffic.
- `--test_origin_stride 24`: 133 origins at horizon 336, 143 at horizon 96.
- Sealed run flags: `--evidence_role confirmation --immutable_artifacts --strict_determinism
  --require_gpu --require_clean_git`.
- A run whose `structural_passed` is false is excluded and rerun once at the same seed. No
  run is excluded for poor forecast quality.

## 3. Endpoints

Both endpoints use the statistic of `docs/prespecification.md`, section 4: per origin,
average the Winkler score over the three seeds, take the difference against flat CP
(calibrated on the second calibration half), divide by the mean flat-CP score, and get a
percentile CI by moving-block bootstrap with block length `ceil(H / 24)` origins, 5,000
resamples, seed 0.

- **E1, primary.** traffic, horizon 336, cluster SCCP minus flat CP.
- **E2, key secondary.** traffic, horizon 336, per-cell state-scaled CP minus flat CP.

**Multiplicity (Holm, two endpoints).** The claim "better than flat CP" is tested for E1 and
E2 together:

1. Compute the relative effect of both endpoints with a 97.5% CI.
2. If the upper bound of one endpoint lies below 0, that endpoint supports "better". The
   other endpoint is then read with its 95% CI.
3. If neither upper bound lies below 0 at 97.5%, both endpoints are read with their 97.5%
   CI.

**Reading.** Each endpoint gets exactly one outcome from the five-outcome rule of
`docs/prespecification.md`, section 4 (margin 1%), applied to the CI that the Holm steps
assign to it. The words "replicated on a second untouched dataset" are allowed only if E1
reads "better, beyond the margin" or "better, within the margin".

**Known limit of the design.** At horizon 336 the test split holds about 9.5 blocks of one
horizon (3,175 windows / 336). The stride does not change that count. ECL had 14.7. The CI
will be coarse, and the manuscript calls fewer than ten blocks too coarse for the
descriptive cells. The reading rule above still applies to E1 and E2. Report the block count
next to the result.

## 4. Secondary analyses (descriptive; no claim of confirmation)

1. Horizon 96: E1 and E2 statistics with 95% CIs. No reading rule.
2. Marginal coverage and `conditional_coverage` from `metrics.json`, as in
   `docs/prespecification.md`, section 5.3.
3. ACI against flat CP (`scripts/aci_paired_comparison.py`, `--gamma 0.005`, block
   multiplier 1, 5,000 resamples, seed 0). ACI uses realised test errors; report it as a
   calibrator with more information.

## 5. Analysis settings

`scripts/paired_interval_inference.py --datasets traffic --block_multiplier 1 --n_boot 5000
--seed 0`, once per `--method {primary, state_scaled}` and per `--confidence {0.975, 0.95}`.
Read E1 from `--method primary` and E2 from `--method state_scaled`, horizon-336 rows only.
The exact commands are in `experiments.md`, step P3.

## 6. Claims allowed and not allowed

The rules of `docs/prespecification.md`, section 7, apply. In addition: do not pool traffic
with ECL into one estimate. Report the two E1 results side by side, each with its own CI and
block count; E2 has no ECL counterpart, because ECL ran with scalar geometry.

## 7. Not prespecified

- Cluster SCCP against per-cell state-scaled CP directly. The run saves both per-origin
  files, so it can be reported as exploratory only.
- Baselines and the hybrid model on traffic.
- Any hyperparameter change on traffic.

## 8. Deviations

Any change to this file after the first traffic test window is evaluated is recorded below
with its date and reason. E1, E2, the horizon, the bootstrap settings, the Holm steps, and the
reading rule do not change.

None yet.
