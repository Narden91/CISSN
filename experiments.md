# CISSN — experiments left to launch

Written 2026-08-31 from an audit of `results/` on branch `conformal-per-cell-geometry`.
Companion to `commands.md` (every command, including the finished ones) and `RUNBOOK.md`
(the protocol and the authority on *why*). If this file and `RUNBOOK.md` disagree,
`RUNBOOK.md` wins.

Everything below is PowerShell, literal, from the repository root. Every block is safe to
Ctrl+C and re-paste: each one checks for the artifact its runner writes last and skips the
cells already on disk.

## Order

The chain is blocking up to the prespecification — B needs A's artifacts:

```
3 (seeds 123,456) -> 3b.0 -> 3b.2 -> [decision: cluster or scale]
4 (10 cells) ------------------------> [decision: revin default]
5 (6 cells) -------------------------> [decision: hybrid or negative result]
                                              |
                                    prespecification (write it down)
                                              |
                          main grid CISSN  +  main grid baselines
                                              |
                                   ablations -> publication review
```

Steps 4 and 5 do not depend on 3b.x and can run in any order relative to it. Nothing past
the prespecification should start before it is written.

## Before anything: the 41 runs at the repository root

`results/CISSN_*` holds 41 grid runs launched with `revin: false` and
`scale_geometry: scalar` — the configuration `CLAUDE.md` records as having no measured
effect. Flat CP beats the primary mechanism on all four datasets and marginal coverage is
0.72-0.83 against nominal 0.90. They are a pre-RevIN negative result, not the main grid,
and they sit at the repository root rather than under `results/publication/` because the
launch inherited the default `--results_dir ./results/`.

Move them out of the way before relaunching, so no aggregation script picks them up. Add a
line to `results/superseded/README.md` recording why.

```powershell
New-Item -ItemType Directory -Force ./results/superseded/pre_revin_scalar_grid
Get-ChildItem ./results -Directory -Filter 'CISSN_*' | Move-Item -Destination ./results/superseded/pre_revin_scalar_grid
Move-Item ./results/publication/cissn_ETTh2.json,./results/publication/cissn_ETTh2.csv,./results/publication/cissn_weather.json,./results/publication/cissn_weather.csv,./results/publication/cissn_exchange_rate.json,./results/publication/cissn_exchange_rate.csv ./results/superseded/pre_revin_scalar_grid
```

Run the `Get-ChildItem` on its own first and read the list — that filter also matches
anything else named `CISSN_*` added at the root since the audit.

## 3. CISSN end-to-end, seeds 123 and 456 `[~]` 1 of 3 done

Seed 42 is on disk. Needed before 3b.0, which reads these run directories.

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

Confirm each `metrics.json` carries all four of `interval`, `interval_flat_cp`,
`interval_cluster_cp`, `interval_state_scaled`.

## 3b.0. Headroom diagnostic `[ ]` 0 of 3 done

No GPU. Reads the three Step 3 directories, so run it after Step 3 finishes.

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

Read `variance_decomposition.per_sample_fraction` — it bounds the *scalar* geometry only,
not conditioning as such — and the `summary_vs_flat` win counts.

## 3b.2. Conditioning selection, seeds 123 and 456 `[~]` 1 of 3 done

Validation only; `run_selection.py` never constructs the test loader.

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

**Decision**: promote `scale` to the primary mechanism only if `interval_state_scaled`
beats **both** `interval_flat_cp` and `interval_cluster_cp` on Winkler on at least 2 of 3
seeds, read from each run's `selection.json`. Read no test artifact while deciding. Record
the per-seed deltas and the outcome in `commands.md` either way — the main grid's
`--conformal_conditioning` and `--scale_geometry` come from here.

## 4. RevIN selection, ETTh2 + weather `[~]` 2 of 12 done

On disk: ETTh2 RevIN seed 42, weather RevIN seed 42. Outstanding: both non-RevIN arms in
full, and seeds 123/456 of both RevIN arms.

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
validation coverage on either dataset. Otherwise keep it opt-in and pass it per dataset in
the main grid. Either way the main grid runs with `--revin` explicitly — audit finding 1
in `commands.md` is what a grid without it produces.

## 5. Hybrid variants, seeds 123 and 456 `[~]` 3 of 9 done

All three arms exist at seed 42 with exactly the directory names this loop expects.

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

**Decision**: promote a variant only on at least 1% mean validation-MSE improvement over
paired DLinear, better on at least 2 of 3 seeds, no worse than 2% on any seed, and no
feature's mean MSE degraded by more than 5%. Ties within 0.5% go to anchored dynamics
without RevIN. If none qualifies, report the hybrid as a negative result and do not retune.

## Prespecification `[ ]` no command

Write the primary endpoint and the primary horizon into `protocol.json` or an adjacent
prespecification file **before** relaunching the main grid. See `RUNBOOK.md`, "Prespecified
statistics".

Note in the same file that the ETTh2, weather, and exchange-rate test splits already carry
evaluations from the superseded pre-RevIN grid. They are no longer untouched, and a later
reader needs that recorded next to the prespecification rather than discovered in the
artifacts.

## Main grid — CISSN `[ ]` full relaunch

The 41 runs on disk do not count — wrong regime, and ETTh1 never got past `h24`. Three
things differ from the launch that produced them: `--revin`, the conditioning and geometry
Step 3b.2 selected, and `--results_dir ./results/publication` so the runs land beside the
baselines instead of at the repository root.

Set the two variables at the top from the Step 3b.2 decision before running.

```powershell
$conditioning = 'cluster'   # from Step 3b.2
$geometry     = 'scalar'    # 'per_cell' if Step 3b.2 selected the per-cell scale geometry
$datasets = @(
  @{ data = 'ETTh1';         horizons = 24,96,192,336,720 },
  @{ data = 'ETTh2';         horizons = 24,96,192,336,720 },
  @{ data = 'weather';       horizons = 96,192,336,720 },
  @{ data = 'exchange_rate'; horizons = 96,192,336,720 }
)
$seeds = 42,123,456
$condTag = if ($conditioning -eq 'scale') { '_scalecond' } else { '' }
foreach ($cell in $datasets) {
  $allDone = $true
  foreach ($h in $cell.horizons) {
    foreach ($seed in $seeds) {
      $setting = "CISSN_$($cell.data)_M_sl96_pl${h}_sd5_dm64_fullrevin${condTag}_lc1_lt0p5_a0p1_per_feature_seed$seed"
      if (-not (Test-Path "./results/publication/$setting/metrics.json")) { $allDone = $false }
    }
  }
  if ($allDone) {
    Write-Host "skip $($cell.data) (all horizons x seeds already done)"
    continue
  }
  uv run python experiments/run_multiseed.py --data $cell.data --all_horizons --seeds 42,123,456 --allow_partial --revin --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --conformal_conditioning $conditioning --scale_geometry $geometry --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --checkpoints ./checkpoints/publication --results_dir ./results/publication --output "./results/publication/cissn_$($cell.data).json" --raw_csv "./results/publication/cissn_$($cell.data).csv"
}
```

`--allow_partial` is what keeps an already-finished cell from aborting a whole dataset, but
it also lists those cells in `failed_seeds`. It cannot tell "this errored" from "this
already exists", so check any name printed there against
`results/publication/<setting>/metrics.json` afterward — present on disk means it was
already done.

Rough budget from the superseded grid's directory timestamps, so a lower bound at best:
ETTh2 ~45 min, weather ~2 h 15, exchange-rate ~15 min, ETTh1 similar to ETTh2. Call it
4-5 hours for all four datasets, and expect more with RevIN.

## Main grid — baselines `[~]` 54 cells missing, 29 duplicated

`deep_ensemble` never ran, and it is the only model outstanding — this loop covers that one
model rather than re-walking all five.

```powershell
$seeds = 42,123,456
$grid = @(
  @{ data = 'ETTh1';         horizons = 24,96,192,336,720 },
  @{ data = 'ETTh2';         horizons = 24,96,192,336,720 },
  @{ data = 'weather';       horizons = 96,192,336,720 },
  @{ data = 'exchange_rate'; horizons = 96,192,336,720 }
)
$total = 0; $skipped = 0; $ran = 0
foreach ($cell in $grid) {
  foreach ($h in $cell.horizons) {
    foreach ($seed in $seeds) {
      $total++
      uv run python experiments/run_baseline.py --model deep_ensemble --data $cell.data --pred_len $h --seed $seed --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --multivariate_strategy per_feature --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --checkpoints ./checkpoints/publication --results_dir ./results/publication
      if ($LASTEXITCODE -ne 0) {
        Write-Host "skip/fail: deep_ensemble $($cell.data) h$h seed$seed (exit $LASTEXITCODE)"
        $skipped++
      } else {
        $ran++
      }
    }
  }
}
Write-Host "`nDone. $ran ran, $skipped skipped-or-failed, $total total."
```

A nonzero exit is normally "already done" — `require_new_run` raises `FileExistsError`
before training, and the final directory never exists until a run completes. A genuine
crash exits nonzero too and is indistinguishable in that counter, so scan the output above
any reported skip for a traceback.

Deep Ensemble trains several members per cell, so budget well above the ~2-4 min/cell the
single-model baselines took.

**Also outstanding, and not a run**: `dlinear` has 83 directories against 54 real cells.
ETTh1 and ETTh2 were each launched twice, at commits `b8edcc9` and `cd8cb6a`, identical in
visible configuration and separable only by `design_hash`. Decide which commit's cells are
the reported ones and move the other set to `results/superseded/` before aggregation —
`generate_publication_tables.py` has no way to choose between them.

If Step 4 selects RevIN, add `--revin` to `dlinear`, `patchtst`, `mc_dropout`, and
`deep_ensemble` — never DeepState — and re-run those models rather than mixing regimes
within one table.

## Ablations `[ ]`

Run only after the main comparison completes.

```powershell
uv run python experiments/run_ablation.py --data ETTh1 --pred_len 336 --seed 42 --ablations full,no_structured_A,no_disentanglement_loss,flat_cp,no_correction_mlp,state_dim_4 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --multivariate_strategy per_feature --require_gpu --require_clean_git --output ./results/publication/ablations_ETTh1_h336_s42.json
```

Two cells the main grid cannot reach, to add here only if the ablation table needs them:
both scale geometries within a single run (a run fixes one `--scale_geometry`), and any
conditioning mode crossed with an architecture ablation such as `no_structured_A` or
`state_dim_4`.

## Publication review `[ ]`

```powershell
uv run python scripts/generate_publication_tables.py --results_root ./results/publication --output_dir ./results/publication/tables
uv run python scripts/generate_publication_figures.py --results_root ./results/publication --output_dir ./results/publication/figures
uv run python scripts/generate_reproducibility_appendix.py --results_root ./results/publication --output ./results/publication/reproducibility.md
```

Before aggregating, confirm the `dlinear` duplicates are resolved and the superseded
pre-RevIN grid is out of `results/publication/`. Exclude runs with
`structural_passed: false`, incomplete artifact sets, different splits, and raw-UQ results
from the primary table. Do not exclude a run for poor forecast quality — report it with its
advisory flags.
