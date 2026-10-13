# CISSN — what to run

Updated 2026-10-13, branch `conformal-per-cell-geometry`. This is the only file with run
commands. `RUNBOOK.md` is the protocol and wins on any conflict; `docs/prespecification.md`
fixes the confirmatory analysis. Everything is PowerShell, from the repository root.

## 1. Before you launch

Every run passes `--require_clean_git`, which refuses any uncommitted change. Commit first,
then check:

```powershell
git status --short
git diff --stat efadb63 HEAD -- cissn experiments
```

The first command prints nothing. The second lists only `cissn/evaluation/inference.py` and
`cissn/utils/artifacts.py`; neither changes what a run computes. Do not change code in
`cissn/` or `experiments/` until R1 to R4 finish, or those cells run on different code.

Every block is safe to Ctrl+C and re-paste. Each skips a cell when its last-written artifact
exists: `selection.json` for selection runs, `metrics.json` (found with a `__*` wildcard,
because immutable runs append `__<design hash>`) for confirmation runs. To redo a finished
cell, delete its directory under `results/` and `checkpoints/` first.

## 2. Run order

| Block | What | Cells | GPU time (rough) | Needs |
| --- | --- | --- | --- | --- |
| R1 | Step 4: RevIN selection, ETTh2 + weather | 10 | 1.5-2 h | commit |
| R2 | CISSN weather h192, seeds 42 and 456 | 2 | 0.5 h | commit |
| R3 | `deep_ensemble`, missing cells. **Skip if R1 selects RevIN** (R5 covers it) | 14 | 2-3 h | commit |
| R4 | ECL confirmatory: CISSN h96 + h336, hybrid h336 | 9 | 2.5-3 h | commit |
| R5 | Baselines under RevIN. **Only if R1 selects RevIN** | 216 | 10-14 h | R1 |
| R6 | Ablations, ETTh1-h336 | 18 | 1.5 h | R2-R5 |
| R7 | Analysis, tables, figures | - | none | R2-R6 |

R1, R2, R3, and R4 are independent. Times come from earlier run timestamps and one-epoch
smoke runs; treat them as rough.

## 3. Run blocks

### R1. Step 4: RevIN selection

Validation only; `run_selection.py` never builds the test loader. 2 of 12 cells exist
(RevIN, seed 42, both datasets).

```powershell
$seeds = 42,123,456
foreach ($data in 'ETTh2','weather') {
  foreach ($revin in $true,$false) {
    foreach ($seed in $seeds) {
      $variant = if ($revin) { "_fullrevin" } else { "" }
      $setting = "CISSN_${data}_M_sl96_pl336_sd5_dm64${variant}_lc1_lt0p5_a0p1_per_feature_seed$seed"
      if (Test-Path "./results/selection/$setting/selection.json") { Write-Host "skip $setting"; continue }
      $revinFlag = @(); if ($revin) { $revinFlag = @('--revin') }
      uv run python experiments/run_selection.py --data $data --pred_len 336 --seed $seed @revinFlag --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
    }
  }
}
```

Read the outcome:

```powershell
foreach ($data in 'ETTh2','weather') { foreach ($revin in $true,$false) {
  $tag = if ($revin) { "_fullrevin" } else { "" }
  $runs = foreach ($seed in 42,123,456) { Get-Content "./results/selection/CISSN_${data}_M_sl96_pl336_sd5_dm64${tag}_lc1_lt0p5_a0p1_per_feature_seed$seed/selection.json" -Raw | ConvertFrom-Json }
  [pscustomobject]@{ data = $data; revin = $revin
    val_mse = ($runs.validation_mse | Measure-Object -Average).Average
    primary_coverage = ($runs.interval.coverage | Measure-Object -Average).Average }
} }
```

**Decision**: RevIN is selected only if, on both datasets, its mean `val_mse` is not higher
and its mean `primary_coverage` is not lower than the non-RevIN arm. This decides only
whether R5 runs and R3 is skipped; it changes no endpoint in `docs/prespecification.md`.

### R2. CISSN weather h192

Same flags as the 52 finished cells. Compare the printed configuration with `config.json`
of `CISSN_weather_M_sl96_pl336_..._seed42__cc2b351d0bda`; only `pred_len` and `seed` may differ.

```powershell
foreach ($seed in 42,456) {
  $setting = "CISSN_weather_M_sl96_pl192_sd5_dm64_fullrevin_lc1_lt0p5_a0p1_per_feature_seed$seed"
  if (Test-Path "./results/publication/${setting}__*/metrics.json") { Write-Host "skip weather h192 seed $seed"; continue }
  uv run python experiments/run_benchmark.py --data weather --pred_len 192 --seed $seed --revin --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --conformal_conditioning cluster --scale_geometry scalar --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --checkpoints ./checkpoints/publication --results_dir ./results/publication
}
```

Do not use `run_multiseed.py` here: at a new commit it retrains every cell under a new hash.

### R3. `deep_ensemble`, missing cells (skip if R1 selects RevIN)

Missing: weather h720 seeds 123 and 456, and all 12 exchange-rate cells. Without RevIN, to
match the 40 finished cells.

```powershell
$grid = @(@{ data = 'ETTh1'; h = 24,96,192,336,720 }, @{ data = 'ETTh2'; h = 24,96,192,336,720 },
          @{ data = 'weather'; h = 96,192,336,720 }, @{ data = 'exchange_rate'; h = 96,192,336,720 })
foreach ($cell in $grid) { foreach ($h in $cell.h) { foreach ($seed in 42,123,456) {
  if (Test-Path "./results/publication/BASELINE_deep_ensemble_$($cell.data)_M_sl96_pl${h}_seed${seed}_*/metrics.json") { continue }
  uv run python experiments/run_baseline.py --model deep_ensemble --data $cell.data --pred_len $h --seed $seed --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --multivariate_strategy per_feature --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --checkpoints ./checkpoints/publication --results_dir ./results/publication
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: deep_ensemble $($cell.data) h$h seed $seed (exit $LASTEXITCODE)" }
} } }
```

A nonzero exit is a real failure here, because the skip check already ran.

### R4. ECL confirmatory (prespecified)

Untouched dataset, `--test_origin_stride 24`, seeds 42, 123, 456. Each run writes under 1 GB.
A one-epoch smoke run on the validation split took 2 min 47 s and showed no memory problem.

```powershell
$common = @('--train_epochs','20','--patience','5','--lradj','cosine','--batch_size','128','--conformal_alpha','0.1','--multivariate_strategy','per_feature','--n_clusters','5','--conformal_conditioning','cluster','--test_origin_stride','24','--require_gpu','--require_clean_git','--immutable_artifacts','--strict_determinism','--evidence_role','confirmation','--checkpoints','./checkpoints/publication','--results_dir','./results/publication')

# CISSN, RevIN, scalar geometry: horizons 336 and 96
foreach ($h in 336,96) { foreach ($seed in 42,123,456) {
  $setting = "CISSN_ECL_M_sl96_pl${h}_sd5_dm64_fullrevin_lc1_lt0p5_a0p1_per_feature_seed$seed"
  if (Test-Path "./results/publication/${setting}__*/metrics.json") { Write-Host "skip CISSN ECL h$h seed $seed"; continue }
  uv run python experiments/run_benchmark.py --data ECL --pred_len $h --seed $seed --revin --scale_geometry scalar @common
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: CISSN ECL h$h seed $seed (exit $LASTEXITCODE)" }
} }

# hybrid_anchored, no RevIN, horizon 336
foreach ($seed in 42,123,456) {
  $setting = "CISSN_ECL_M_sl96_pl336_sd5_dm64_hybrid_anchored_lc0_lt0_a0p1_per_feature_seed$seed"
  if (Test-Path "./results/publication/${setting}__*/metrics.json") { Write-Host "skip hybrid ECL seed $seed"; continue }
  uv run python experiments/run_benchmark.py --data ECL --pred_len 336 --seed $seed --architecture hybrid --state_dynamics anchored --lambda_cov 0 --lambda_temp 0 @common
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: hybrid ECL seed $seed (exit $LASTEXITCODE)" }
}
```

Check each run's `environment.json`: `git_dirty` is `false` and `git_commit` is a commit that
contains `docs/prespecification.md`. A run that fails `structural_passed` is rerun once at
the same seed (prespecification, section 3).

### R5. Baselines under RevIN (only if R1 selects RevIN)

`dlinear`, `patchtst`, `mc_dropout`, `deep_ensemble`; never DeepState. The run name gains a
`fullrevin` token, so these never collide with the existing runs.

```powershell
$grid = @(@{ data = 'ETTh1'; h = 24,96,192,336,720 }, @{ data = 'ETTh2'; h = 24,96,192,336,720 },
          @{ data = 'weather'; h = 96,192,336,720 }, @{ data = 'exchange_rate'; h = 96,192,336,720 })
foreach ($model in 'dlinear','patchtst','mc_dropout','deep_ensemble') { foreach ($cell in $grid) { foreach ($h in $cell.h) { foreach ($seed in 42,123,456) {
  if (Test-Path "./results/publication/BASELINE_${model}_$($cell.data)_M_sl96_pl${h}_seed${seed}_*fullrevin__*/metrics.json") { continue }
  uv run python experiments/run_baseline.py --model $model --revin --data $cell.data --pred_len $h --seed $seed --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --multivariate_strategy per_feature --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --checkpoints ./checkpoints/publication --results_dir ./results/publication
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $model $($cell.data) h$h seed $seed (exit $LASTEXITCODE)" }
} } } }
```

When it finishes, move the non-RevIN copies of those four models, so one table never mixes
regimes:

```powershell
$dest = './results/superseded/baselines_no_revin'; New-Item -ItemType Directory -Force $dest | Out-Null
Get-ChildItem ./results/publication -Directory -Filter 'BASELINE_*' | Where-Object { $_.Name -match '^BASELINE_(dlinear|patchtst|mc_dropout|deep_ensemble)_' -and $_.Name -notmatch 'fullrevin' } | Move-Item -Destination $dest
```

Add a row for `baselines_no_revin/` to `results/superseded/README.md`.

### R6. Ablations (ETTh1-h336, exposed split, so diagnostic)

```powershell
foreach ($seed in 42,123,456) {
  $out = "./results/publication/ablations_ETTh1_h336_s$seed.json"
  if (Test-Path $out) { Write-Host "skip ablations seed $seed"; continue }
  uv run python experiments/run_ablation.py --data ETTh1 --pred_len 336 --seed $seed --ablations all --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --multivariate_strategy per_feature --require_gpu --require_clean_git --output $out
}
```

### R7. Analysis (values come from `docs/prespecification.md`, section 8)

```powershell
uv run python scripts/paired_interval_inference.py --results_root ./results/publication --datasets ECL --block_multiplier 1 --n_boot 5000 --seed 0 --output ./results/publication/paired_interval_inference_ECL.json
uv run python scripts/paired_interval_inference.py --results_root ./results/publication --datasets ETTh2 weather exchange_rate --block_multiplier 1 --n_boot 5000 --seed 0 --output ./results/publication/paired_interval_inference_exposed.json
uv run python scripts/paired_point_comparison.py --results_root ./results/publication --checkpoints_root ./checkpoints/publication --data ECL --pred_len 336 --seeds 42 123 456 --output ./results/publication/hybrid_vs_base_ECL.json
uv run python scripts/generate_publication_tables.py --results_root ./results/publication --output_dir ./results/publication/tables
uv run python scripts/generate_publication_figures.py --results_root ./results/publication --output_dir ./results/publication/figures
uv run python scripts/generate_reproducibility_appendix.py --results_root ./results/publication --output ./results/publication/reproducibility.md
```

Read the primary endpoint from the ECL horizon-336 row only, with the reading rule in
`docs/prespecification.md`, section 4. Exclude runs with `structural_passed: false`,
incomplete artifact sets, different splits, and raw-UQ results. Do not exclude a run for
poor forecast quality; report it with its advisory flags.

## 4. What to check after each block

| After | File | Field |
| --- | --- | --- |
| R2, R4 | `metrics.json` | `interval`, `interval_flat_cp`, `interval_cluster_cp`, `interval_state_scaled` all present |
| R2-R5 | `sanity.json` | `structural_passed` is the only field that excludes a run; quality flags are advisory |
| R1 | `selection.json` | `validation_mse` and validation coverage; never a test artifact |

## 5. Recorded decisions (validation data only)

**Step 3b.2: `cluster` stays primary, `--scale_geometry scalar`.** Rule: promote `scale` only
if it beats flat CP and cluster SCCP on validation Winkler on at least 2 of 3 seeds
(ETTh1-h336, RevIN, per-cell geometry). It did on 1.

| Seed | flat CP | cluster SCCP | state-scaled | beats both |
| --- | --- | --- | --- | --- |
| 42 | 6.7219 | 6.7309 | 6.7727 | no |
| 123 | 6.7183 | 6.7104 | 6.6638 | yes |
| 456 | 6.7342 | 6.6947 | 6.7095 | no |

**Step 5: promote `hybrid_anchored`.** Change in validation MSE against its own frozen
DLinear base: -4.15%, -4.11%, -2.81% (mean -3.69%, 3 of 3 seeds). Worst feature +0.73%
against the 5% limit (`results/selection/hybrid_feature_mse.json`). `hybrid_legacy` is
-1.29% (worst feature +4.08%); `hybrid_anchored_revin` is -0.17%. A prior hybrid run gained
4.1% on validation and lost on test (`docs/methodology.md`), so this supports no
point-accuracy claim.

**Step 4** is R1. **Done and not repeated here**: Steps 1 to 3, 3b.0, the exposed-dataset
CISSN grid (52 of 54 cells, `fb5231d`), and 256 of 270 baseline cells. Their commands are
in `RUNBOOK.md`.

## 6. Rules that bind every launch

- Confirmation runs need `--evidence_role confirmation --immutable_artifacts
  --strict_determinism --require_clean_git` together. Without them a run is `development`
  and cannot enter the publication table.
- ETTh2, weather, and exchange-rate test splits were exposed to the pre-RevIN grid, and
  ETTh1 test was exposed during development. State that in any claim; ECL is the only
  untouched confirmatory dataset.
- CISSN ran with RevIN and the existing baselines ran without. Compare point accuracy only
  within one regime (R5), and report intervals and point forecasts separately.
- The two missing weather h192 cells and the R3, R4, R5 runs use a later commit than the 52
  finished CISSN cells (`fb5231d`). Record the commit next to each result.
