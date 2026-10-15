# CISSN publication runbook

Sole launch protocol. Earlier-protocol results not publication evidence. Never pool with these artifacts. Open run commands in `experiments.md`. Finished blocks R1 to R7 in `git show a1add62:experiments.md`. Confirmatory analysis and reading rules in `docs/prespecification.md` (ECL) and
`docs/prespecification_solar.md` (solar, draft).

> Evidence note: ETTh1 test artifacts informed development, stay diagnostics only. ETTh2, weather, exchange-rate test splits read in pre-RevIN grid, so results secondary, descriptive. ECL = confirmatory dataset. Second confirmatory dataset: solar (Step 7, P3). All selection use validation data only.

## Execution status

`[x]` artifact on disk satisfies step under current code. `[~]` partly done.
`[-]` dropped, reason in its row. `[ ]` step must launch. Runs that cannot satisfy step moved to `results/superseded/`, reason in that folder's `README.md`; diagnostics, never enter publication table. Updated 2026-10-15.

| Step | State | Evidence on disk |
| --- | --- | --- |
| 1 environment, data, tests | `[x]` | 2026-08-27: CUDA available (torch `2.11.0+cu128`), `verify_datasets.py` OK on all 10 datasets, 166 tests passed |
| 2 DLinear reference (ETTh1-h336, seed 42) | `[x]` | `results/validation/BASELINE_dlinear_ETTh1_M_sl96_pl336_seed42`: clean git, `structural_passed`, test MSE `0.619` |
| 3 CISSN end-to-end (ETTh1-h336) | `[x]` | seeds 42, 123, 456 under `--revin` in `results/validation/`; pre-RevIN run in `results/superseded/CISSN_ETTh1_h336_seed42_preRevIN/` |
| 3b.0 headroom diagnostic, ETTh1-h336 | `[x]` | `results/validation/conditioning_headroom_revin_seed*.json` |
| 3b.0 headroom diagnostic, other datasets | `[-]` | dropped: reuse exposed test split, Step 3b.2 already decides question |
| 3b.2 validation-only selection | `[x]` | `scale` beats both calibrators on 1 of 3 seeds, so `cluster` stays primary (table in Step 3b.2) |
| 4 RevIN selection (ETTh2, weather) | `[x]` | RevIN selected: `results/selection/r1_decision.json` |
| 5 hybrid variant selection | `[x]` | `hybrid_anchored` meets all four criteria: `results/selection/hybrid_feature_mse.json` |
| Prespecified statistics | `[x]` | `docs/prespecification.md`, written 2026-10-13 |
| Main grid, CISSN | `[x]` | 54 exposed-dataset runs; 9 ECL runs at `6adc24b` (legacy h96 and h336, hybrid h336) |
| Main grid, baselines under RevIN | `[~]` | 204 of 216 R5 runs; 12 MC-Dropout and Deep Ensemble runs at h192 await encoder-fix rerun (block `r5`, `experiments.md`). DeepState: 54 runs without RevIN |
| Ablations | `[x]` | `results/publication/ablations_ETTh1_h336_s*.json`, regime without RevIN |
| Publication review | `[~]` | draft `manuscript/main.tex`; numbers from `manuscript/analysis/build_numbers.py` |
| 7 P1 ECL baselines, P2 ACI | `[ ]` | launch with `bash slurm/launch.sh` (blocks `p1`, `p2`); see Step 7 |
| 7 P3 solar confirmatory test | `[ ]` | draft `docs/prespecification_solar.md`; launch with `bash slurm/launch.sh`; see Step 7 |

Epoch budget: `--train_epochs 20 --patience 5` checked against 60-epoch, patience-10 arm on ETTh1-h336, seed 42, RevIN. Arm stopped epoch 27, best validation epoch 17. Twenty epochs not truncation. Keep budget so grid stays comparable with Step 2.

## Locked design

- Datasets: exposed grid `ETTh1` (development, excluded from every endpoint), `ETTh2`,
  `weather`, `exchange_rate`. Confirmatory: `ECL`. Step 7 adds `solar` (P3).
- Seeds: `42,123,456`.
- Horizons: `24,96,192,336,720` for ETT; `96,192,336,720` for weather and exchange-rate; `96,336`
  for ECL and `solar`.
- Shared training: `--train_epochs 20 --patience 5 --lradj cosine --batch_size 128`.
- Shared calibration: chronological train-tail calibration split, `--cal_fraction 0.2`,
  `--conformal_alpha 0.1`.
- Primary interval geometry: `--multivariate_strategy per_feature`; report `coverage_primary`
  (marginal). `max` = separate simultaneous-coverage analysis.
- Publication safeguards: `--require_gpu --require_clean_git`.

Cluster partition (`StateConditionalConformal.fit_partition`) and state-scaled sigma regression (`StateScaledConformal.fit_scale`) both learn from first half of calibration split. Quantile calibration use second half. Both calibrators get same fitting set; `docs/methodology.md` describes earlier asymmetric version. Serial dependence recorded in artifacts, never converted into coverage guarantee.

All runners show live batch progress. Use `--no_progress` only for CI or captured logs.

## Step 1: environment, data, tests `[x]`

```powershell
uv sync
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
uv run python scripts/verify_datasets.py
uv run python tests/run_tests.py
```

Stop if command fails. `verify_datasets.py` must pass for every dataset in locked design.

## Step 2: DLinear reference reproduction `[x]`

```powershell
uv run python experiments/run_baseline.py --model dlinear --data ETTh1 --pred_len 336 --seed 42 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --multivariate_strategy per_feature --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation
```

Apply artifact review rule (Publication review). Record expected full-train reference and observed fair-split result separately; protocol reserves train data for calibration.

## Step 3: CISSN end-to-end `[x]`

```powershell
uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 336 --seed 42 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation
```

Before reading coverage, check `cluster_stats.json`, `scale_stats.json`, `dependence_diagnostics.json` for fallback clusters, fitted scale coefficients, serial dependence.

## Step 3b: state conditioning `[x]`

> Selection override: step development-only. Diagnostic reads test artifacts, so no outcome here may choose calibrator. Choice belongs to `experiments/run_selection.py`, which trains and calibrates identically but scores every calibrator on validation split only; never builds test loader. See Step 3b.2.

`--conformal_conditioning cluster` (K-Means partition) default; `scale` (continuous
`sigma(state)`) alternative. Every run reports both and flat CP, so flag only sets which one drives headline `interval` block.

Scalar sigma has almost no headroom: per-sample axis holds about 1% of residual variance, label-informed per-sample reference gains under 1%. Signal sits in state x cell interaction that only per-cell sigma or cluster calibrator can express. Development measurements, ETTh1-h336, 3 RevIN seeds x 4 nested cuts (not independent; effective n near 3 seeds, near 1 dataset): mean Winkler difference against flat CP is cluster `-0.124`, scalar
`+0.011`, per-cell `-0.237` (`docs/methodology.md`).

### Step 3b.0: headroom diagnostic on saved artifacts (no GPU) `[x]`

Needs run folder with `states.npy`, `residuals.npy`, `pred.npy`, `true.npy`, which any completed `run_benchmark.py` run writes. Use `--revin` run.

```powershell
uv run python scripts/diagnose_conditioning_headroom.py --run_dir ./results/validation/CISSN_<dataset>_..._fullrevin_..._seed42 --output ./results/validation/conditioning_headroom_revin.json
```

Read `variance_decomposition.per_sample_fraction` and `summary_vs_flat` win counts. Older artifacts from asymmetric fitting scheme in `results/superseded/headroom/` and
`results/superseded/percell/`; diagnostics, never publication evidence.

### Step 3b.2: validation-only selection runs `[x]`

```powershell
uv run python experiments/run_selection.py --data ETTh1 --pred_len 336 --seed 42 --revin --conformal_conditioning scale --scale_geometry per_cell --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
```

Repeat for seeds `123,456`. Read `interval`, `interval_flat_cp`, `interval_cluster_cp` from each `selection.json`. **Decision rule**: promote `scale` only if `interval_state_scaled` beats both `interval_flat_cp` and `interval_cluster_cp` in Winkler score on at least 2 of 3 seeds. Read no test artifact for this decision.

Outcome (validation Winkler score, ETTh1-h336, RevIN, per-cell geometry): `scale` beat both on 1 of 3 seeds, so `cluster` stays primary with `--scale_geometry scalar`.

| Seed | flat CP | cluster SCCP | state-scaled | beats both |
| --- | --- | --- | --- | --- |
| 42 | 6.7219 | 6.7309 | 6.7727 | no |
| 123 | 6.7183 | 6.7104 | 6.6638 | yes |
| 456 | 6.7342 | 6.6947 | 6.7095 | no |

## Step 4: instance normalisation `[x]`

> Selection override: ETTh1 test values diagnostic context only. RevIN chosen from
> `experiments/run_selection.py` (validation only).

Legacy CISSN collapses toward training mean on ETTh1-h336: forecast keeps about 7% of target variance, lowers MSE without tracking signal. `--revin` removes level-tracking burden. On ETTh1 test split (diagnostic only), test MSE falls `1.280 -> 0.771`
and coverage moves `0.788 -> 0.908` (`docs/methodology.md`).

Run ETTh2 and weather at h336, seeds 42, 123, 456, with and without `--revin`. Exact cells are block `r1` in `slurm/cells.py`; one cell:

```powershell
uv run python experiments/run_selection.py --data ETTh2 --pred_len 336 --seed 42 --revin --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
```

**Decision rule**: RevIN becomes regime only if, on both datasets, mean `validation_mse` not higher and mean validation coverage not lower than without it. Check `vali_variance_ratio` in `history.json`: ratio falls while validation loss improves = amplitude collapse, not hard dataset.

Outcome (`results/selection/r1_decision.json`): RevIN selected. Validation MSE `1.153` against
`5.685` on ETTh2 and `0.637` against `0.684` on weather. Coverage `0.921` against `0.859` and
`0.907` against `0.878`. CISSN and four non-DeepState baselines run with `--revin`.

## Step 5: hybrid variant selection (validation only) `[x]`

Legacy architecture routes forecast through five-dimensional state. Without RevIN this caps forecast rank at 5. Under RevIN cap near `5 + 2C`, measured effective rank 7 to 8 (`docs/methodology.md`, "Rank and the DLinear gap under RevIN"). Hybrid keeps DLinear as base, adds state as correction:

```text
total = DLinear(history) + LinearCorrectionHead(StateEncoder(history))
```

Head zero-initialised, base frozen, so correction epoch 0 equals base and failed correction stage falls back to it. `runtime.json` records `base_val_loss` and
`correction_improved_on_base`.

Run three variants on ETTh1-h336, seeds 42, 123, 456, through selection controller:

```powershell
uv run python experiments/run_selection.py --architecture hybrid --data ETTh1 --pred_len 336 --seed 42 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
uv run python experiments/run_selection.py --architecture hybrid --state_dynamics anchored --data ETTh1 --pred_len 336 --seed 42 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
uv run python experiments/run_selection.py --architecture hybrid --state_dynamics anchored --state_revin --data ETTh1 --pred_len 336 --seed 42 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
```

Select on **validation** MSE only (`validation_mse` in `selection.json`). Promote variant only if it:

- improves on paired DLinear by at least `1%` on average;
- improves on at least two of three seeds;
- is no worse than `2%` on any seed;
- degrades no feature's mean MSE by more than `5%`.

Ties within `0.5%` resolve to anchored dynamics without RevIN. `--state_revin` rescales correction and changes epoch-0 gradient scale, so treat as separate arm. If no variant meets criteria, report hybrid as negative result, do not retune on test data.

Outcome: `hybrid_anchored` changed validation MSE by `-4.15%`, `-4.11%`, `-2.81%` (mean
`-3.69%`, 3 of 3 seeds), worst feature `+0.73%` against `5%` limit. `hybrid_legacy` reached
`-1.29%` (worst feature `+4.08%`) and `hybrid_anchored_revin` `-0.17%`. Prior hybrid run gained 4.1% on validation, lost on test (`docs/methodology.md`), so this supports no point-accuracy claim.

## Prespecified statistics `[x]`

**Rule.** Write and commit prespecification before any confirmatory test split opens:
`docs/prespecification.md` for ECL, `docs/prespecification_solar.md` (draft) for second dataset. Confirmatory run counts only if its `environment.json` records commit containing the file, with `git_dirty` false.

`docs/prespecification.md` holds fixed values. Principles:

- One primary endpoint, only result abstract may call confirmatory: ECL, horizon 336,
  cluster SCCP against flat CP, Winkler score. Change of 2026-10-13: earlier version of this section proposed endpoint aggregated over whole grid, tested with paired test over cells. `docs/prespecification.md` replaced it before any ECL test split opened.
- Primary horizon is selected horizon, not neutral: every development and selection measurement used h336. State that, note that on ECL horizon untouched.
- Per-cell "N of 3 seeds" win counts descriptive. With 3 seeds, false-positive rate near 50% per cell under no effect. Same for "N of M nested cuts" figures from
  `diagnose_conditioning_headroom.py`; cuts nested, not independent.
- Every other cell or comparison secondary, descriptive.

## Main grid `[~]`

Every command is sealed confirmation run. Needs `--evidence_role confirmation` together
with `--immutable_artifacts --strict_determinism --require_clean_git` (`enforce_evidence_contract`
requires all three). Without them run is `development`, cannot enter publication table.

Exposed-dataset CISSN grid ran with `--revin --conformal_conditioning cluster
--scale_geometry scalar`. Every run also reports flat CP and state-scaled CP, so flag only sets which is primary. Baselines run through `run_baseline.py` with same options. RevIN applies to `dlinear`, `patchtst`, `mc_dropout`, `deep_ensemble`, not DeepState
(`docs/methodology.md`). UQ baselines conformalized by default; raw output needs
`--uq_interval_mode raw`, secondary. Slurm blocks in `slurm/cells.py` build every command.

Ablations run after main comparison. Run fixes one `--scale_geometry`, so two cells not reachable from `run_ablation.py`: both geometries in one run, and conditioning mode crossed with architecture ablation. Neither added; validation selection and window-placement diagnostic cover geometry question.

## Step 7: additions for submission `[ ]`

Three additions answer gaps listed in `venue.md`. None change ECL endpoint, its analysis, or its reading rule. Each result reported as added after ECL confirmatory analysis.

**Order.** Commit and push Step 7 code and `docs/prespecification_solar.md` (preflight refuses a P3 launch without that file in `HEAD`). Copy `data/solar_AL.txt` to the cluster. Pull on cluster, run `bash slurm/launch.sh`: submits baseline reruns (`r5`), P1 (`p1`), P2 (`p2`), P3 (`p3s`, `p3`, `p3b`) together. `r5` reruns then run at later commit than other 204 R5 runs. Code differs by encoder fix and by Step 7 additions that leave stride-1 computation unchanged (stride option with default 1, extra output files); manuscript states both commits.

**Transfer.** Pull only small files analysis reads (`pack_small` in `experiments.md`,
section 4). Never copy whole run folders.

**P1: point baselines on ECL.** DLinear, PatchTST-lite, MC-Dropout, Deep Ensemble under RevIN, horizons 336 and 96, seeds 42, 123, 456 (24 runs, block `p1`). `--test_origin_stride 24`
scores same daily origins as ECL CISSN runs; training and calibration use every window.

- Evidence role: descriptive point accuracy within RevIN regime (`docs/prespecification.md`,
  section 6). Report ratio to CISSN per cell and seed count; no test.
- Baseline intervals are flat CP on full calibration split (`conformalized_full_cal`). Never compare with paired calibrators of CISSN run.

**P2: adaptive conformal inference (ACI).** ACI (Gibbs and Candès, 2021) on same residuals of same CISSN models, against flat CP (`scripts/aci_paired_comparison.py`; update rule in
`cissn/conformal/aci.py`).

- Step size `gamma = 0.005`, value Gibbs and Candès used, fixed now. No other value runs.
- One level per horizon-feature cell. Origin i updates levels only for origins at least H steps later, because its target fully observed only then.
- Level that asks beyond largest calibration score uses that score; output counts these pairs (`n_capped`).
- Reload check: ACI with `gamma = 0` must reproduce saved flat-CP per-origin scores
  (relative tolerance 1e-4), or script stops.
- Datasets: ECL (block `p2`), exposed grid (workstation), solar (block `p3b`).
- Evidence role: descriptive. ACI uses realised test errors as they arrive; report as calibrator with more information than split CP. Use bootstrap settings of
  `docs/prespecification.md`, section 8.

**P3: second confirmatory test on solar.** Six CISSN runs (horizons 336 and 96, seeds 42, 123, 456,
block `p3`) with ECL settings plus `--scale_geometry per_cell` and `--test_origin_stride 25`. Draft
`docs/prespecification_solar.md` fixes design, two endpoints (cluster SCCP and per-cell
state-scaled CP, each against flat CP at horizon 336), Holm steps, reading rule. Block `p3s` is a
one-epoch, validation-only check; `p3` starts after it succeeds.

Why solar: traffic at horizon 336 gives only 9.5 bootstrap blocks, and the manuscript calls fewer than
ten too coarse to read. Solar gives 29. It also differs more from ECL than traffic does.

Why stride 25: solar's daily period is 144 steps, and 25 shares no factor with it, so test origins
cycle through every time of day. On ETTh2 at h336 the cluster-SCCP effect depends on the origin's
phase, from `-8.2%` to `+7.6%` (all phases `-2.2%`; one fixed phase at stride 24 gives `+7.6%`).
Calibration windows cover every phase, so a test set at one phase can move the result by several
points. The ECL runs use stride 24 on hourly data, which fixes one phase. A post-hoc re-score of the
ECL checkpoints at stride 25 would test that; it is not decided or built.

**After Step 7.** `build_numbers.py` places P1 runs in point table. ACI and solar results need new rows in that script before manuscript cites them; add from output files only, after runs exist.

## Publication review `[~]`

**Artifact review rule.** Before aggregating, every result needs full artifact contract and protocol manifest showing same split, calibration, shared training settings for comparison cell. Exclude only incomplete results, runs with `structural_passed: false`, runs on different splits, raw-UQ results. `structural_passed` false when artifact unreadable:
empty, non-finite, shape-inconsistent, or inverted interval bounds. Quality flags in
`sanity.json` advisory. Never drop run for poor forecast quality; report it with its flags.

**Reporting.** Report per cell: mean and standard deviation over seeds, paired method comparisons with seed win count beside mean, `coverage_primary`, width, Winkler score,
MSIS, coverage scope, interval origin, `worst_prespecified_bin_coverage` and
`max_coverage_deviation`, cluster and scale fallbacks, dependence diagnostics. Favourable mean can hide sign that flips between seeds (`docs/methodology.md`). Do not describe time-series result as unconditional distribution-free coverage.

**Naming rule.** Call state coordinates "level-like", "trend-like", "seasonal-like", "residual-like". Do not claim they identify physical components unless recovery tests on simulated data and stability checks pass.

**Source of numbers.** `manuscript/analysis/build_numbers.py` writes every manuscript number from
`results/`. Older table and figure scripts (`generate_publication_tables.py` and others) and their R7 commands in `git show a1add62:experiments.md`.