# CISSN — experiments: status and what is left

Updated 2026-10-13 from a check of `results/` and `checkpoints/` on branch
`conformal-per-cell-geometry` at `efadb63`. Companion to `commands.md` (every command,
including the finished ones) and `RUNBOOK.md` (the protocol and the authority on *why*).
If this file and `RUNBOOK.md` disagree, `RUNBOOK.md` wins.

Everything below is PowerShell, literal, from the repository root. Every run block is safe
to Ctrl+C and re-paste: each one checks for the artifact its runner writes last and skips
the cells already on disk.

## Status

| Step | State | On disk |
| --- | --- | --- |
| Pre-RevIN grid moved out of the root | `[x]` | `results/superseded/pre_revin_scalar_grid/` (41 runs + 6 aggregate files). `results/superseded/README.md` has no row for it yet |
| 3. CISSN end-to-end, ETTh1-h336, 3 seeds | `[x]` | `results/validation/CISSN_ETTh1_..._fullrevin_..._seed{42,123,456}` |
| 3b.0. Headroom diagnostic, 3 seeds | `[x]` | `results/validation/conditioning_headroom_revin_seed*.json` |
| 3b.2. Conditioning selection, 3 seeds | `[x]` | outcome computed below: keep `cluster` |
| 4. RevIN selection, ETTh2 + weather | `[~]` 2 of 12 | RevIN seed 42 of both datasets only |
| 5. Hybrid variants, 3 arms x 3 seeds | `[~]` 9 of 9 runs, decision incomplete | criterion 4 (per-feature MSE) cannot be computed from saved artifacts |
| Prespecification | `[ ]` | no file. **The main grid already ran without it** |
| Main grid — CISSN | `[~]` 52 of 54 | missing: weather h192 seeds 42 and 456 |
| Main grid — baselines | `[~]` 256 of 270, plus 29 duplicates | `deep_ensemble` 40 of 54; `dlinear` has 83 dirs for 54 cells |
| Ablations | `[ ]` | — |
| Publication review | `[ ]` | — |

## Decisions computed from validation

Both outcomes use `selection.json` (validation split, test loader never built) and the
checkpoint-side `runtime.json`. No test artifact was read to compute them.

### Step 3b.2 — keep `cluster` as the primary mechanism

Rule: promote `scale` only if `interval_state_scaled` beats **both** flat CP and cluster
SCCP on validation Winkler on at least 2 of 3 seeds. Geometry `per_cell`, ETTh1-h336, RevIN.

| Seed | flat CP | cluster SCCP | state-scaled per-cell | beats both? |
| --- | --- | --- | --- | --- |
| 42 | 6.7219 | 6.7309 | 6.7727 | no |
| 123 | 6.7183 | 6.7104 | 6.6638 | yes |
| 456 | 6.7342 | 6.6947 | 6.7095 | no (beats flat, loses to cluster) |

Result: 1 of 3. `cluster` stays primary, `--scale_geometry` stays `scalar`. This matches
what the main grid ran. All three mechanisms lie within 1% of each other on every seed.
The seed 123 and 456 `selection.json` files are dated 2026-08-31 15:42 and 15:46, before
the first main-grid run (16:11 the same day), so the data for this decision existed before
the grid started. Copy this table into `commands.md` as the rule requires.

### Step 5 — `hybrid_anchored` passes criteria 1-3; criterion 4 is pending

Paired DLinear = the frozen base inside each hybrid run (`base_val_loss` in
`checkpoints/selection/<setting>/runtime.json`). The base value is identical across the
three arms at each seed, so the pairing is exact. Delta = `validation_mse / base - 1`.

| Arm | seed 42 | seed 123 | seed 456 | mean | better on | worst seed |
| --- | --- | --- | --- | --- | --- | --- |
| `hybrid_legacy` | -0.95% | -1.21% | -1.70% | -1.29% | 3/3 | -0.95% |
| `hybrid_anchored` | -4.15% | -4.11% | -2.81% | -3.69% | 3/3 | -2.81% |
| `hybrid_anchored_revin` | 0.00% (fell back) | 0.00% (fell back) | -0.51% | -0.17% | 1/3 | 0.00% |

- Criteria 1-3 (mean at least 1% better, better on at least 2 of 3 seeds, no seed worse
  than 2%): `hybrid_anchored` and `hybrid_legacy` pass. `hybrid_anchored` leads by 2.4
  points, so the 0.5% tie rule does not apply.
- Criterion 4 (no feature's mean MSE worse by more than 5%): **not computable**.
  `run_selection.py` saves no per-feature validation MSE for the hybrid or its base. The
  checkpoints (`checkpoint_base.pth`, `checkpoint.pth`, `checkpoint_head.pth`) are on disk,
  so the check needs inference on the validation split only, not retraining.
- Record the outcome as "anchored: criteria 1-3 pass, criterion 4 pending", not as
  "promoted".
- Context for the reader, not an override of the rule: `docs/methodology.md:303` records
  an earlier hybrid run whose validation loss improved by 4.1% while its ETTh1 test MSE was
  0.799 against the frozen base's 0.619. ETTh1 test is diagnostic only.
- `RUNBOOK.md` gives no main-grid command for a promoted hybrid. If criterion 4 passes, a
  decision is needed on which grid cells the hybrid runs on.

## Facts that change how results can be stated

1. **The main grid ran before the prespecification existed.** The RevIN CISSN grid ran on
   2026-08-31 and 2026-09-01 at `fb5231d`. No prespecification file exists. The ETTh2,
   weather, and exchange-rate test splits now carry two sets of evaluations: the
   pre-RevIN grid (whose results were read: flat CP beat the primary mechanism on all four
   datasets) and the RevIN grid. Write the prespecification before anyone reads a number
   from `results/publication/CISSN_*`, and state in it that it was written after those runs
   finished.
2. **Validation coverage on ETTh1 is far below nominal for every mechanism, flat CP
   included.** At nominal 0.90, ETTh1-h336 validation coverage is 0.74-0.76 for legacy
   CISSN and 0.75-0.82 for the hybrids. ETTh2 and weather (seed 42) reach 0.91-0.94. The
   calibration split is the tail of the train window, so this is consistent with a shift
   between calibration and later data, not with a defect of one mechanism. On ETTh1, only
   relative Winkler statements between mechanisms hold; a marginal coverage claim does not.
3. **The diagnostic gain does not appear under the protocol.** Under the fixed fitting
   scheme, `diagnose_conditioning_headroom.py` (ETTh1 test, diagnostic only) gives
   per-cell state-scaled -0.21 to -0.27 Winkler vs flat CP and cluster -0.10 to -0.16, on
   all 4 cuts for each seed. The protocol path on validation gives differences under 1%.
   The "not yet re-measured" note in `CLAUDE.md` about the cluster vs per-cell ordering is
   out of date for the diagnostic.
4. **Point-forecast tables mix regimes.** CISSN ran with `--revin`; every baseline ran with
   `revin=False`. Interval comparisons are not affected, because flat CP, cluster SCCP, and
   state-scaled CP are paired inside each CISSN run. A CISSN-vs-DLinear point-accuracy row
   is confounded until Step 4 finishes and the baselines run in the same regime.
5. **The main-grid skip checks in the previous version of this file never matched.**
   `--immutable_artifacts` appends `__<design hash>` to every run directory, and the design
   hash includes the git commit. A path check without a wildcard always misses, and a
   rerun at a new commit trains a second copy of every cell. The selection and validation
   checks did match, because those runs are not immutable. Every immutable block below
   uses a wildcard.
6. **Mixed commits inside one dataset.** The two missing weather h192 cells will run at
   the commit that carries this file; the other 52 CISSN cells ran at `fb5231d`. Between
   `fb5231d` and `efadb63` the only code change is in `cissn/evaluation/sanity.py`
   (per-member ensemble history for the deep-ensemble forecast quality review). Record the
   commit next to the weather results.

## Experiments left to run

Order: L0 and L1 first (no GPU). L2 and L3 are independent of each other. L4 and L5 can
start after the pre-launch steps below. L6 depends on L3. L7 depends on L2. L8 and L9 come
last.

**Before any GPU launch.** Every run block passes `--require_clean_git`, which refuses
any output from `git status --short`. This file and `results/superseded/README.md` are
tracked, so commit them (and any other pending change) first. Then confirm that no code
changed since the grid commits:

```powershell
git status --short
git diff --stat efadb63 HEAD -- cissn experiments scripts
```

The first command must print nothing. The second must print nothing. Land a code change
only after L3, L4, and L5 finish, or those cells run on different code from the rest of
the grid.

```text
L0 housekeeping ─┐
L1 prespec ──────┼─> L4 CISSN weather h192 ─┐
                 ├─> L5 deep_ensemble 14 ───┤
L3 RevIN select ─┴─> L6 baselines + RevIN ──┼─> L8 ablations -> L9 publication review
L2 hybrid crit.4 ──> L7 hybrid grid (if promoted) ┘
```

### L0. Housekeeping `[ ]` no GPU

1. **Empty temporary directories.** `results/` holds 93 `.publication.tmp-*` directories.
   `finalize_result_directory` moves the run out of the temporary root but leaves the root
   behind, so every immutable run adds one. None holds a file (one holds an empty
   `BASELINE_deep_ensemble_ETTh1_..._pl24_seed42_...__a14269d4459f` directory from an
   interrupted attempt; the cell is complete under a different hash). Read the count, then
   delete:

   ```powershell
   $tmp = Get-ChildItem ./results -Directory -Force -Filter '.publication.tmp-*' | Where-Object { -not (Get-ChildItem $_.FullName -Recurse -File -Force) }
   $tmp.Count
   $tmp | Remove-Item -Recurse
   ```

2. **Checkpoint directories with no finished result.** These come from crashed runs. They
   do not block a rerun (the hash differs at a new commit), but move them so nothing pairs
   them with a result by name:

   ```powershell
   New-Item -ItemType Directory -Force ./checkpoints/superseded
   Move-Item ./checkpoints/publication/CISSN_weather_M_sl96_pl192_sd5_dm64_fullrevin_lc1_lt0p5_a0p1_per_feature_seed42__61706acd3b1b ./checkpoints/superseded/
   Move-Item ./checkpoints/publication/CISSN_weather_M_sl96_pl192_sd5_dm64_fullrevin_lc1_lt0p5_a0p1_per_feature_seed456__62b390c8b1c5 ./checkpoints/superseded/
   Move-Item ./checkpoints/publication/BASELINE_dlinear_ETTh2_M_sl96_pl720_seed456__e895403d7fe9 ./checkpoints/superseded/
   ```

   `deep_ensemble` member checkpoints (`..._memberN_seed...`) have no result directory by
   design. Several cells carry two member sets under different hashes; only the hash that
   matches the finalized result directory is current.

3. **`dlinear` duplicates.** `scripts/generate_publication_tables.py` reads every
   `metrics.json` under `results/publication` recursively, so both copies would enter the
   table. Rule, fixed before any metric is read: keep the `b8edcc9` set, because it is
   complete (54 of 54) and matches the commit of the other single-model baselines. Move
   the 29 `cd8cb6a` directories:

   ```powershell
   $dest = './results/superseded/dlinear_cd8cb6a_duplicate'
   New-Item -ItemType Directory -Force $dest
   Get-ChildItem ./results/publication -Directory -Filter 'BASELINE_dlinear_*' | Where-Object {
     (Get-Content (Join-Path $_.FullName 'environment.json') -Raw | ConvertFrom-Json).git_commit -like 'cd8cb6a*'
   } | Move-Item -Destination $dest
   ```

4. **Stale aggregate files.** `results/publication/cissn_*.json`/`.csv` came from
   `run_multiseed.py`. `cissn_weather.json` holds only h336 and h720. The table generator
   does not read these files. Do not regenerate them by rerunning `run_multiseed.py`: at a
   new commit it retrains every cell under a new hash, then fails on two matching
   directories.

5. Add rows for `pre_revin_scalar_grid/` and `dlinear_cd8cb6a_duplicate/` to
   `results/superseded/README.md`.

### L1. Prespecification `[ ]` blocking, no command

Write the primary endpoint and the primary horizon into a prespecification file next to
the publication results before anyone reads a number from `results/publication/CISSN_*`.
See `RUNBOOK.md`, "Prespecified statistics". The file must also state:

- that it was written on its actual date, after the RevIN main grid finished;
- that the ETTh2, weather, and exchange-rate test splits carry evaluations from the
  pre-RevIN grid, whose results were read, and from the RevIN grid;
- whether ETTh1 test (diagnostic only) is in the confirmatory endpoint. Without it, 13
  dataset x horizon cells remain (ETTh2 5, weather 4, exchange-rate 4);
- the Step 3b.2 and Step 5 outcomes above, with their dates.

### L2. Step 5 criterion 4: per-feature validation MSE `[ ]` no GPU needed

No command exists yet. It needs a validation-only script that loads each hybrid
checkpoint and its frozen base and writes per-feature validation MSE for both. Until this
runs, Step 5 has no outcome.

### L3. Step 4: RevIN selection, ETTh2 + weather `[~]` 2 of 12

Outstanding: both non-RevIN arms, and seeds 123/456 of both RevIN arms. Selection runs
do not use `--immutable_artifacts`, so the path check needs no wildcard.

```powershell
$seeds = 42,123,456
$datasets = 'ETTh2','weather'
foreach ($data in $datasets) {
  foreach ($revin in $true,$false) {
    foreach ($seed in $seeds) {
      $variant = if ($revin) { "_fullrevin" } else { "" }
      $setting = "CISSN_${data}_M_sl96_pl336_sd5_dm64${variant}_lc1_lt0p5_a0p1_per_feature_seed$seed"
      if (Test-Path "./results/selection/$setting/selection.json") {
        Write-Host "skip $data revin=$revin seed $seed (already done)"
        continue
      }
      $revinFlag = if ($revin) { @('--revin') } else { @() }
      uv run python experiments/run_selection.py --data $data --pred_len 336 --seed $seed @revinFlag --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
    }
  }
}
```

**Decision**: flip `--revin` to default only if it regresses neither `validation_mse` nor
validation coverage on either dataset. CISSN already ran with `--revin` explicitly, so
this outcome now decides only whether the baselines rerun under RevIN (L6).

### L4. Main grid — CISSN, weather h192 seeds 42 and 456 `[ ]`

Run the two cells directly with `run_benchmark.py`, with the same flags the grid used.
Do not use `run_multiseed.py` (see L0 step 4).

```powershell
foreach ($seed in 42,456) {
  $setting = "CISSN_weather_M_sl96_pl192_sd5_dm64_fullrevin_lc1_lt0p5_a0p1_per_feature_seed$seed"
  if (Test-Path "./results/publication/${setting}__*/metrics.json") {
    Write-Host "skip weather h192 seed $seed (already done)"
    continue
  }
  uv run python experiments/run_benchmark.py --data weather --pred_len 192 --seed $seed --revin --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --conformal_conditioning cluster --scale_geometry scalar --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --checkpoints ./checkpoints/publication --results_dir ./results/publication
}
```

Before launch, compare the printed configuration with `config.json` of
`CISSN_weather_M_sl96_pl336_..._seed42__cc2b351d0bda`: only `pred_len` and `seed` may
differ.

### L5. Main grid — `deep_ensemble`, 14 cells `[~]` 40 of 54

Missing: weather h720 seeds 123 and 456, and all 12 exchange-rate cells. The 40 finished
cells ran at `efadb63`. These 14 run at the commit that carries this file, which differs
only in documentation if the pre-launch `git diff` above prints nothing.

```powershell
$seeds = 42,123,456
$grid = @(
  @{ data = 'ETTh1';         horizons = 24,96,192,336,720 },
  @{ data = 'ETTh2';         horizons = 24,96,192,336,720 },
  @{ data = 'weather';       horizons = 96,192,336,720 },
  @{ data = 'exchange_rate'; horizons = 96,192,336,720 }
)
foreach ($cell in $grid) {
  foreach ($h in $cell.horizons) {
    foreach ($seed in $seeds) {
      if (Test-Path "./results/publication/BASELINE_deep_ensemble_$($cell.data)_M_sl96_pl${h}_seed${seed}_*/metrics.json") {
        Write-Host "skip deep_ensemble $($cell.data) h$h seed $seed (already done)"
        continue
      }
      uv run python experiments/run_baseline.py --model deep_ensemble --data $cell.data --pred_len $h --seed $seed --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --multivariate_strategy per_feature --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --checkpoints ./checkpoints/publication --results_dir ./results/publication
      if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: deep_ensemble $($cell.data) h$h seed $seed (exit $LASTEXITCODE)" }
    }
  }
}
```

With the wildcard check, a nonzero exit now means a real failure, not "already done".

### L6. Baselines under RevIN `[ ]` only if L3 selects RevIN

Add `--revin` to `dlinear`, `patchtst`, `mc_dropout`, and `deep_ensemble` (never
DeepState) and rerun those models over the full grid. `run_baseline.py` appends the token
`fullrevin` after the seed token, so the RevIN runs never collide with the existing ones
and the skip check below matches only RevIN runs.

```powershell
$models = 'dlinear','patchtst','mc_dropout','deep_ensemble'
$seeds = 42,123,456
$grid = @(
  @{ data = 'ETTh1';         horizons = 24,96,192,336,720 },
  @{ data = 'ETTh2';         horizons = 24,96,192,336,720 },
  @{ data = 'weather';       horizons = 96,192,336,720 },
  @{ data = 'exchange_rate'; horizons = 96,192,336,720 }
)
foreach ($model in $models) {
  foreach ($cell in $grid) {
    foreach ($h in $cell.horizons) {
      foreach ($seed in $seeds) {
        if (Test-Path "./results/publication/BASELINE_${model}_$($cell.data)_M_sl96_pl${h}_seed${seed}_*fullrevin__*/metrics.json") {
          Write-Host "skip $model $($cell.data) h$h seed $seed (already done)"
          continue
        }
        uv run python experiments/run_baseline.py --model $model --revin --data $cell.data --pred_len $h --seed $seed --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --multivariate_strategy per_feature --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --checkpoints ./checkpoints/publication --results_dir ./results/publication
        if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $model $($cell.data) h$h seed $seed (exit $LASTEXITCODE)" }
      }
    }
  }
}
```

When the loop finishes, move the non-RevIN directories of these four models to
`results/superseded/baselines_no_revin/`, so one table never mixes regimes. `dlinear` names
carry no token between the seed and the hash, so `seed42_*fullrevin__*` matches only the
RevIN copy for every model.

### L7. Hybrid main grid `[ ]` only if L2 passes

`RUNBOOK.md` has no command for this. Decide the cells first.

### L8. Ablations `[ ]`

Run only after the main comparison completes.

```powershell
uv run python experiments/run_ablation.py --data ETTh1 --pred_len 336 --seed 42 --ablations full,no_structured_A,no_disentanglement_loss,flat_cp,no_correction_mlp,state_dim_4 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --multivariate_strategy per_feature --require_gpu --require_clean_git --output ./results/publication/ablations_ETTh1_h336_s42.json
```

Two cells the main grid cannot reach, to add here only if the ablation table needs them:
both scale geometries within a single run (a run fixes one `--scale_geometry`), and any
conditioning mode crossed with an architecture ablation such as `no_structured_A` or
`state_dim_4`.

### L9. Publication review `[ ]`

```powershell
uv run python scripts/generate_publication_tables.py --results_root ./results/publication --output_dir ./results/publication/tables
uv run python scripts/generate_publication_figures.py --results_root ./results/publication --output_dir ./results/publication/figures
uv run python scripts/generate_reproducibility_appendix.py --results_root ./results/publication --output ./results/publication/reproducibility.md
```

Before aggregating, confirm L0 is done. Exclude runs with `structural_passed: false`,
incomplete artifact sets, different splits, and raw-UQ results from the primary table.
Do not exclude a run for poor forecast quality — report it with its advisory flags.

## Completed — kept for reproduction

### 3. CISSN end-to-end, ETTh1-h336, seeds 42/123/456 `[x]`

```powershell
$seeds = 42,123,456
foreach ($seed in $seeds) {
  $setting = "CISSN_ETTh1_M_sl96_pl336_sd5_dm64_fullrevin_lc1_lt0p5_a0p1_per_feature_seed$seed"
  if (Test-Path "./results/validation/$setting/metrics.json") {
    Write-Host "skip seed $seed (already done)"
    continue
  }
  uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 336 --seed $seed --revin --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation
}
```

### 3b.0. Headroom diagnostic `[x]`

```powershell
$seeds = 42,123,456
foreach ($seed in $seeds) {
  $out = "./results/validation/conditioning_headroom_revin_seed$seed.json"
  if (Test-Path $out) {
    Write-Host "skip seed $seed (already done)"
    continue
  }
  $runDir = "./results/validation/CISSN_ETTh1_M_sl96_pl336_sd5_dm64_fullrevin_lc1_lt0p5_a0p1_per_feature_seed$seed"
  uv run python scripts/diagnose_conditioning_headroom.py --run_dir $runDir --output $out
}
```

`variance_decomposition.per_sample_fraction` is 0.0063-0.0068 on all three seeds; it
bounds the *scalar* geometry only, not conditioning as such.

### 3b.2. Conditioning selection `[x]`

```powershell
$seeds = 42,123,456
foreach ($seed in $seeds) {
  $setting = "CISSN_ETTh1_M_sl96_pl336_sd5_dm64_fullrevin_scalecond_lc1_lt0p5_a0p1_per_feature_seed$seed"
  if (Test-Path "./results/selection/$setting/selection.json") {
    Write-Host "skip seed $seed (already done)"
    continue
  }
  uv run python experiments/run_selection.py --data ETTh1 --pred_len 336 --seed $seed --revin --conformal_conditioning scale --scale_geometry per_cell --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
}
```

### 5. Hybrid variants `[x]` runs done, decision pending (L2)

```powershell
$seeds = 42,123,456
$arms = @(
  @{ suffix = 'hybrid_legacy';           args = @('--architecture','hybrid') },
  @{ suffix = 'hybrid_anchored';         args = @('--architecture','hybrid','--state_dynamics','anchored') },
  @{ suffix = 'hybrid_anchored_revin';   args = @('--architecture','hybrid','--state_dynamics','anchored','--state_revin') }
)
foreach ($arm in $arms) {
  foreach ($seed in $seeds) {
    $setting = "CISSN_ETTh1_M_sl96_pl336_sd5_dm64_$($arm.suffix)_lc0_lt0_a0p1_per_feature_seed$seed"
    if (Test-Path "./results/selection/$setting/selection.json") {
      Write-Host "skip $($arm.suffix) seed $seed (already done)"
      continue
    }
    uv run python experiments/run_selection.py @($arm.args) --data ETTh1 --pred_len 336 --seed $seed --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
  }
}
```

### Main grid — CISSN, 52 of 54 cells `[x]`

Ran 2026-08-31 to 2026-09-01 at `fb5231d` through `run_multiseed.py` with `--revin
--conformal_conditioning cluster --scale_geometry scalar --immutable_artifacts
--strict_determinism --evidence_role confirmation`. The remaining two cells are L4.
