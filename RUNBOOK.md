# CISSN publication runbook

This is the sole launch protocol. Results from earlier protocols are not publication evidence
and must not be pooled with these artifacts. Open run commands are in `experiments.md`. The
finished blocks R1 to R7 are in `git show a1add62:experiments.md`. The confirmatory analysis
and the reading rules are in `docs/prespecification.md` (ECL) and
`docs/prespecification_traffic.md` (traffic, draft).

> Evidence note: ETTh1 test artifacts informed development and stay diagnostics only. The
> ETTh2, weather, and exchange-rate test splits were read in a pre-RevIN grid, so their
> results are secondary and descriptive. ECL is the confirmatory dataset. A second
> confirmatory dataset is pending a decision (Step 7, P3). All selection uses validation data
> only.

## Execution status

`[x]` an artifact on disk satisfies the step under the current code. `[~]` partly done.
`[-]` dropped, with the reason in its row. `[ ]` the step must be launched. Runs that cannot
satisfy their step moved to `results/superseded/` with a reason in that folder's `README.md`;
they are diagnostics and never enter a publication table. Updated 2026-10-15.

| Step | State | Evidence on disk |
| --- | --- | --- |
| 1 environment, data, tests | `[x]` | 2026-08-27: CUDA available (torch `2.11.0+cu128`), `verify_datasets.py` OK on all 10 datasets, 166 tests passed |
| 2 DLinear reference (ETTh1-h336, seed 42) | `[x]` | `results/validation/BASELINE_dlinear_ETTh1_M_sl96_pl336_seed42`: clean git, `structural_passed`, test MSE `0.619` |
| 3 CISSN end-to-end (ETTh1-h336) | `[x]` | seeds 42, 123, 456 under `--revin` in `results/validation/`; the pre-RevIN run is in `results/superseded/CISSN_ETTh1_h336_seed42_preRevIN/` |
| 3b.0 headroom diagnostic, ETTh1-h336 | `[x]` | `results/validation/conditioning_headroom_revin_seed*.json` |
| 3b.0 headroom diagnostic, other datasets | `[-]` | dropped: it would reuse an exposed test split, and Step 3b.2 already decides the question |
| 3b.2 validation-only selection | `[x]` | `scale` beats both calibrators on 1 of 3 seeds, so `cluster` stays primary (table in Step 3b.2) |
| 4 RevIN selection (ETTh2, weather) | `[x]` | RevIN selected: `results/selection/r1_decision.json` |
| 5 hybrid variant selection | `[x]` | `hybrid_anchored` meets all four criteria: `results/selection/hybrid_feature_mse.json` |
| Prespecified statistics | `[x]` | `docs/prespecification.md`, written 2026-10-13 |
| Main grid, CISSN | `[x]` | 54 exposed-dataset runs; 9 ECL runs at `6adc24b` (legacy h96 and h336, hybrid h336) |
| Main grid, baselines under RevIN | `[~]` | 204 of 216 R5 runs; 12 MC-Dropout and Deep Ensemble runs at h192 await the encoder-fix rerun (`experiments.md` E1). DeepState: 54 runs without RevIN |
| Ablations | `[x]` | `results/publication/ablations_ETTh1_h336_s*.json`, regime without RevIN |
| Publication review | `[~]` | draft `manuscript/main.tex`; numbers from `manuscript/analysis/build_numbers.py` |
| 7 P1 ECL baselines, P2 ACI | `[ ]` | start after `experiments.md` E1; see Step 7 |
| 7 P3 second confirmatory dataset | `[ ]` | blocked: dataset undecided; see Step 7 |

Epoch budget: `--train_epochs 20 --patience 5` was checked against a 60-epoch, patience-10 arm
on ETTh1-h336, seed 42, RevIN. That arm stopped at epoch 27 with its best validation at epoch
17. Twenty epochs is not a truncation. Keep the budget so the grid stays comparable with Step 2.

## Locked design

- Datasets: exposed grid `ETTh1` (development, excluded from every endpoint), `ETTh2`,
  `weather`, `exchange_rate`. Confirmatory: `ECL`. Step 7 adds a second one (pending, P3).
- Seeds: `42,123,456`.
- Horizons: `24,96,192,336,720` for ETT; `96,192,336,720` for weather and exchange-rate; `96,336`
  for ECL and the P3 dataset.
- Shared training: `--train_epochs 20 --patience 5 --lradj cosine --batch_size 128`.
- Shared calibration: chronological train-tail calibration split, `--cal_fraction 0.2`,
  `--conformal_alpha 0.1`.
- Primary interval geometry: `--multivariate_strategy per_feature`; report `coverage_primary`
  (marginal). `max` is a separate simultaneous-coverage analysis.
- Publication safeguards: `--require_gpu --require_clean_git`.

The cluster partition (`StateConditionalConformal.fit_partition`) and the state-scaled sigma
regression (`StateScaledConformal.fit_scale`) both learn from the first half of the calibration
split. Quantile calibration uses the second half. This gives both calibrators the same fitting
set; `docs/methodology.md` describes the earlier asymmetric version. Serial dependence is
recorded in artifacts and is never converted into a coverage guarantee.

All runners show live batch progress. Use `--no_progress` only for CI or captured logs.

## Step 1: environment, data, tests `[x]`

```powershell
uv sync
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
uv run python scripts/verify_datasets.py
uv run python tests/run_tests.py
```

Stop if a command fails. `verify_datasets.py` must pass for every dataset in the locked design.

## Step 2: DLinear reference reproduction `[x]`

```powershell
uv run python experiments/run_baseline.py --model dlinear --data ETTh1 --pred_len 336 --seed 42 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --multivariate_strategy per_feature --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation
```

Apply the artifact review rule (Publication review). Record the expected full-train reference
and the observed fair-split result separately; this protocol reserves train data for
calibration.

## Step 3: CISSN end-to-end `[x]`

```powershell
uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 336 --seed 42 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation
```

Before you read coverage, check `cluster_stats.json`, `scale_stats.json`, and
`dependence_diagnostics.json` for fallback clusters, fitted scale coefficients, and serial
dependence.

## Step 3b: state conditioning `[x]`

> Selection override: this step is development-only. Its diagnostic reads test artifacts, so no
> outcome here may choose a calibrator. The choice belongs to `experiments/run_selection.py`,
> which trains and calibrates identically but scores every calibrator on the validation split
> only; it never builds the test loader. See Step 3b.2.

`--conformal_conditioning cluster` (K-Means partition) is the default; `scale` (continuous
`sigma(state)`) is the alternative. Every run reports both and flat CP, so the flag only sets
which one drives the headline `interval` block.

A scalar sigma has almost no headroom: the per-sample axis holds about 1% of residual variance,
and a label-informed per-sample reference gains under 1%. The signal sits in a state x cell
interaction that only a per-cell sigma or the cluster calibrator can express. Development
measurements, ETTh1-h336, 3 RevIN seeds x 4 nested cuts (not independent; effective n is near 3
seeds, near 1 dataset): mean Winkler difference against flat CP is cluster `-0.124`, scalar
`+0.011`, per-cell `-0.237` (`docs/methodology.md`).

### Step 3b.0: headroom diagnostic on saved artifacts (no GPU) `[x]`

It needs a run folder with `states.npy`, `residuals.npy`, `pred.npy`, and `true.npy`, which any
completed `run_benchmark.py` run writes. Use a `--revin` run.

```powershell
uv run python scripts/diagnose_conditioning_headroom.py --run_dir ./results/validation/CISSN_<dataset>_..._fullrevin_..._seed42 --output ./results/validation/conditioning_headroom_revin.json
```

Read `variance_decomposition.per_sample_fraction` and the `summary_vs_flat` win counts. Older
artifacts from the asymmetric fitting scheme are in `results/superseded/headroom/` and
`results/superseded/percell/`; they are diagnostics, never publication evidence.

### Step 3b.2: validation-only selection runs `[x]`

```powershell
uv run python experiments/run_selection.py --data ETTh1 --pred_len 336 --seed 42 --revin --conformal_conditioning scale --scale_geometry per_cell --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
```

Repeat for seeds `123,456`. Read `interval`, `interval_flat_cp`, and `interval_cluster_cp` from
each `selection.json`. **Decision rule**: promote `scale` only if `interval_state_scaled` beats
both `interval_flat_cp` and `interval_cluster_cp` in Winkler score on at least 2 of 3 seeds. Do
not read any test artifact for this decision.

Outcome (validation Winkler score, ETTh1-h336, RevIN, per-cell geometry): `scale` beat both on
1 of 3 seeds, so `cluster` stays primary with `--scale_geometry scalar`.

| Seed | flat CP | cluster SCCP | state-scaled | beats both |
| --- | --- | --- | --- | --- |
| 42 | 6.7219 | 6.7309 | 6.7727 | no |
| 123 | 6.7183 | 6.7104 | 6.6638 | yes |
| 456 | 6.7342 | 6.6947 | 6.7095 | no |

## Step 4: instance normalisation `[x]`

> Selection override: ETTh1 test values are diagnostic context only. RevIN is chosen from
> `experiments/run_selection.py` (validation only).

Legacy CISSN collapses toward the training mean on ETTh1-h336: the forecast keeps about 7% of
the target variance, which lowers MSE without tracking the signal. `--revin` removes the
level-tracking burden. On the ETTh1 test split (diagnostic only), test MSE falls `1.280 -> 0.771`
and coverage moves `0.788 -> 0.908` (`docs/methodology.md`).

Run ETTh2 and weather at h336, seeds 42, 123, 456, with and without `--revin`. The exact cells
are block `r1` in `slurm/cells.py`; one cell is:

```powershell
uv run python experiments/run_selection.py --data ETTh2 --pred_len 336 --seed 42 --revin --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
```

**Decision rule**: RevIN becomes the regime only if, on both datasets, its mean `validation_mse`
is not higher and its mean validation coverage is not lower than without it. Check
`vali_variance_ratio` in `history.json`: a ratio that falls while validation loss improves is
amplitude collapse, not a hard dataset.

Outcome (`results/selection/r1_decision.json`): RevIN selected. Validation MSE `1.153` against
`5.685` on ETTh2 and `0.637` against `0.684` on weather. Coverage `0.921` against `0.859` and
`0.907` against `0.878`. CISSN and the four non-DeepState baselines run with `--revin`.

## Step 5: hybrid variant selection (validation only) `[x]`

The legacy architecture routes the forecast through a five-dimensional state. Without RevIN this
caps forecast rank at 5. Under RevIN the cap is near `5 + 2C`, and the measured effective rank is
7 to 8 (`docs/methodology.md`, "Rank and the DLinear gap under RevIN"). The hybrid keeps DLinear
as the base and adds the state as a correction:

```text
total = DLinear(history) + LinearCorrectionHead(StateEncoder(history))
```

The head is zero-initialised and the base is frozen, so correction epoch 0 equals the base and a
failed correction stage falls back to it. `runtime.json` records `base_val_loss` and
`correction_improved_on_base`.

Run three variants on ETTh1-h336, seeds 42, 123, 456, through the selection controller:

```powershell
uv run python experiments/run_selection.py --architecture hybrid --data ETTh1 --pred_len 336 --seed 42 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
uv run python experiments/run_selection.py --architecture hybrid --state_dynamics anchored --data ETTh1 --pred_len 336 --seed 42 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
uv run python experiments/run_selection.py --architecture hybrid --state_dynamics anchored --state_revin --data ETTh1 --pred_len 336 --seed 42 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
```

Select on **validation** MSE only (`validation_mse` in `selection.json`). Promote a variant only
if it:

- improves on paired DLinear by at least `1%` on average;
- improves on at least two of three seeds;
- is no worse than `2%` on any seed;
- degrades no feature's mean MSE by more than `5%`.

Ties within `0.5%` resolve to anchored dynamics without RevIN. `--state_revin` rescales the
correction and changes the epoch-0 gradient scale, so treat it as a separate arm. If no variant
meets the criteria, report the hybrid as a negative result and do not retune on test data.

Outcome: `hybrid_anchored` changed validation MSE by `-4.15%`, `-4.11%`, `-2.81%` (mean
`-3.69%`, 3 of 3 seeds), worst feature `+0.73%` against the `5%` limit. `hybrid_legacy` reached
`-1.29%` (worst feature `+4.08%`) and `hybrid_anchored_revin` `-0.17%`. A prior hybrid run gained
4.1% on validation and lost on test (`docs/methodology.md`), so this supports no point-accuracy
claim.

## Prespecified statistics `[x]`

**Rule.** Write and commit a prespecification before any confirmatory test split is opened:
`docs/prespecification.md` for ECL, and `docs/prespecification_traffic.md` (draft) for the
second dataset. A confirmatory run counts only if its `environment.json` records a commit that
contains the file, with `git_dirty` false.

`docs/prespecification.md` holds the fixed values. The principles:

- One primary endpoint, the only result an abstract may call confirmatory: ECL, horizon 336,
  cluster SCCP against flat CP, Winkler score. Change of 2026-10-13: an earlier version of this
  section proposed an endpoint aggregated over the whole grid, tested with a paired test over
  cells. `docs/prespecification.md` replaced it before any ECL test split was opened.
- The primary horizon is a selected horizon, not a neutral one: every development and selection
  measurement used h336. State that, and note that on ECL the horizon was untouched.
- Per-cell "N of 3 seeds" win counts are descriptive. With 3 seeds they have a false-positive
  rate near 50% per cell under no effect. The same holds for "N of M nested cuts" figures from
  `diagnose_conditioning_headroom.py`; the cuts are nested, not independent.
- Every other cell or comparison is secondary and descriptive.

## Main grid `[~]`

Every command is a sealed confirmation run. It needs `--evidence_role confirmation` together
with `--immutable_artifacts --strict_determinism --require_clean_git` (`enforce_evidence_contract`
requires all three). Without them a run is `development` and cannot enter a publication table.

The exposed-dataset CISSN grid ran with `--revin --conformal_conditioning cluster
--scale_geometry scalar`. Every run also reports flat CP and state-scaled CP, so the flag sets
only which one is primary. Baselines run through `run_baseline.py` with the same options. RevIN
applies to `dlinear`, `patchtst`, `mc_dropout`, and `deep_ensemble`, not to DeepState
(`docs/methodology.md`). UQ baselines are conformalized by default; raw output needs
`--uq_interval_mode raw` and is secondary. The slurm blocks in `slurm/cells.py` build every
command.

Ablations run after the main comparison. A run fixes one `--scale_geometry`, so two cells are
not reachable from `run_ablation.py`: both geometries in one run, and a conditioning mode crossed
with an architecture ablation. Neither is added; the validation selection and the window-placement
diagnostic cover the geometry question.

## Step 7: additions for submission `[ ]`

Three additions answer the gaps listed in `venue.md`. None changes the ECL endpoint, its
analysis, or its reading rule. Each result is reported as added after the ECL confirmatory
analysis.

**Order.** The Step 7 code and the traffic draft are already committed (`ef84e5e`). Do not pull
them on the cluster until `experiments.md` E1 finishes, so the E1 runs share a commit with the
other 204 R5 runs apart from the encoder fix. Then pull and start Step 7. P1 and P2 do not
depend on P3 and can launch first.

**Transfer.** Pull only the small files the analysis reads (`pack_small` in `experiments.md`,
section 2.2). Never copy whole run folders.

**P1: point baselines on ECL.** DLinear, PatchTST-lite, MC-Dropout, and Deep Ensemble under
RevIN, horizons 336 and 96, seeds 42, 123, 456 (24 runs, block `p1`). `--test_origin_stride 24`
scores the same daily origins as the ECL CISSN runs; training and calibration use every window.

- Evidence role: descriptive point accuracy within the RevIN regime (`docs/prespecification.md`,
  section 6). Report the ratio to CISSN per cell and the count of seeds; no test.
- Baseline intervals are flat CP on the full calibration split (`conformalized_full_cal`). Never
  compare them with the paired calibrators of a CISSN run.

**P2: adaptive conformal inference (ACI).** ACI (Gibbs and Candès, 2021) on the same residuals of
the same CISSN models, against flat CP (`scripts/aci_paired_comparison.py`; update rule in
`cissn/conformal/aci.py`).

- Step size `gamma = 0.005`, the value Gibbs and Candès used, fixed now. No other value runs.
- One level per horizon-feature cell. Origin i updates the levels only for origins at least H
  steps later, because its target is fully observed only then.
- A level that asks beyond the largest calibration score uses that score; the output counts
  these pairs (`n_capped`).
- Reload check: ACI with `gamma = 0` must reproduce the saved flat-CP per-origin scores
  (relative tolerance 1e-4), or the script stops.
- Datasets: ECL (block `p2`), the exposed grid (workstation), and traffic (block `p3b`).
- Evidence role: descriptive. ACI uses realised test errors as they arrive; report it as a
  calibrator with more information than split CP. Use the bootstrap settings of
  `docs/prespecification.md`, section 8.

**P3: second confirmatory test (dataset pending).** Six CISSN runs (horizons 336 and 96, seeds 42,
123, 456, block `p3`) with the ECL settings plus `--scale_geometry per_cell`. The draft
`docs/prespecification_traffic.md` fixes the design, two endpoints (cluster SCCP and per-cell
state-scaled CP, each against flat CP at horizon 336), the Holm steps, and the reading rule.

The dataset is undecided. Traffic at horizon 336 gives only 9.5 bootstrap blocks, and the
manuscript calls fewer than ten too coarse to read. Options: solar at horizon 336 (30.4 blocks,
stride to settle) or traffic with horizon 96 as primary (35.8 blocks; ECL h96 is already known).
The draft is already committed, and no traffic run exists, so it can change freely until the
first traffic test window is opened. Preflight now passes for P3, so only this rule stops a
launch: do not launch `p3s`, `p3`, or `p3b` until the dataset is decided.

**After Step 7.** `build_numbers.py` places the P1 runs in the point table. The ACI and traffic
results need new rows in that script before the manuscript cites them; add them from the output
files only, after the runs exist.

## Publication review `[~]`

**Artifact review rule.** Before aggregating, every result needs the full artifact contract and a
protocol manifest that shows the same split, calibration, and shared training settings for the
comparison cell. Exclude only incomplete results, runs with `structural_passed: false`, runs on
different splits, and raw-UQ results. `structural_passed` is false when an artifact is unreadable:
empty, non-finite, shape-inconsistent, or with inverted interval bounds. Quality flags in
`sanity.json` are advisory. Never drop a run for poor forecast quality; report it with its flags.

**Reporting.** Report these per cell: mean and standard deviation over seeds, paired method
comparisons with the seed win count beside the mean, `coverage_primary`, width, Winkler score,
MSIS, coverage scope, interval origin, `worst_prespecified_bin_coverage` and
`max_coverage_deviation`, cluster and scale fallbacks, and dependence diagnostics. A favourable
mean can hide a sign that flips between seeds (`docs/methodology.md`). Do not describe the
time-series result as unconditional distribution-free coverage.

**Naming rule.** Call the state coordinates "level-like", "trend-like", "seasonal-like", and
"residual-like". Do not claim that they identify physical components unless recovery tests on
simulated data and stability checks pass.

**Source of numbers.** `manuscript/analysis/build_numbers.py` writes every manuscript number from
`results/`. The older table and figure scripts (`generate_publication_tables.py` and others) and
their R7 commands are in `git show a1add62:experiments.md`.
