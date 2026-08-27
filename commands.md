# CISSN launch commands

Every command below is literal — copy and paste as-is, PowerShell, from the repository
root. No placeholders. Run the blocks in order; a later step consumes an earlier one's
artifacts.

`RUNBOOK.md` remains the protocol and the authority on *why* each step exists and what
decision rule it feeds. If the two disagree, `RUNBOOK.md` wins.

Audited 2026-08-27, branch `conformal-per-cell-geometry`. `results/` holds exactly one
usable artifacts (Step 2, and Step 3 seed 42); everything superseded was moved to
`results/superseded/`.

## Resume mechanism

Every loop below is **safe to interrupt (Ctrl+C) and re-paste**. It will skip whatever
already finished and continue from the next cell. Two different skip checks are needed
because the runners behave differently:

- **`run_baseline.py` / `run_multiseed.py` under `--immutable_artifacts`** (the main grid):
  `require_new_run` (`cissn/utils/artifacts.py:82`) raises `FileExistsError` and the
  process exits `1` if the run directory already exists — it never silently overwrites.
  The loops below catch that with `$LASTEXITCODE`, print `skip`, and move on. This is
  the actual immutability guard already in the code, not a heuristic layered on top.
- **`run_benchmark.py` (Step 3) / `run_selection.py`** (no `--immutable_artifacts`): these
  use `mkdir(parents=True, exist_ok=True)` and would silently **overwrite** a finished run
  rather than error. There is no exception to catch, so the loops below check
  `Test-Path` on the file each runner writes last (`metrics.json` /
  `selection.json`) **before** launching, and skip if it exists.

Interrupting mid-run never leaves a corrupt "done" artifact: `metrics.json` /
`selection.json` / `completion.json` are all written only after training and evaluation
finish, so Ctrl+C during training just means that cell restarts from scratch on resume —
it is never marked done and never silently accepted as partial.

If you changed code or config and want a genuine re-run of an already-completed cell,
delete that cell's directory under `results/...` (and `checkpoints/...` for
`run_baseline.py`/`run_benchmark.py`) before resuming — the skip logic has no way to tell
"stale" from "still valid" apart from presence on disk.

## Status

| Exp | Did | ☐ |
| --- | --- | --- |
| 1. Environment, data, tests | **yes** | `[x]` |
| 2. DLinear reference, ETTh1-h336/s42 | **yes** | `[x]` |
| 3. CISSN end-to-end, RevIN, 3 seeds | seed 42 only | `[~]` |
| 3b.0. Headroom diagnostic, ETTh1 | no | `[ ]` |
| 3b.2. Conditioning selection, 3 seeds | no | `[ ]` |
| 4. RevIN selection, ETTh2 + weather, paired | no | `[ ]` |
| 5. Hybrid variants, 3 arms x 3 seeds | no | `[ ]` |
| Prespecification (no command — write it down) | no | `[ ]` |
| Main grid, CISSN, 4 datasets | no | `[ ]` |
| Main grid, baselines, 5 models | no | `[ ]` |
| Ablations | no | `[ ]` |
| Publication review | no | `[ ]` |

---

## 1. Environment, data, tests `[x]` done

Verified 2026-08-27: torch `2.11.0+cu128`, NVIDIA GeForce RTX 5080 Laptop GPU;
`verify_datasets.py` OK for all 10 registered datasets (solar carries no integrity
fingerprint, which does not affect the four locked datasets); 166 tests pass.

Stop if any line fails.

```powershell
uv sync
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
uv run python scripts/verify_datasets.py
uv run python tests/run_tests.py
```

## 2. DLinear reference `[x]` done

Already on disk at `results/validation/BASELINE_dlinear_ETTh1_M_sl96_pl336_seed42`
(clean git, `structural_passed`, test MSE 0.619). Re-run only if you want it refreshed.

```powershell
uv run python experiments/run_baseline.py --model dlinear --data ETTh1 --pred_len 336 --seed 42 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --multivariate_strategy per_feature --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation
```

## 3. CISSN end-to-end, RevIN, current code `[ ]`

Replaces the superseded pre-RevIN run. Needed before 3b.0.

```powershell
uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 336 --seed 42 --revin --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation

uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 336 --seed 123 --revin --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation

uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 336 --seed 456 --revin --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation
```

Writes to `results/validation/CISSN_ETTh1_M_sl96_pl336_sd5_dm64_fullrevin_lc1_lt0p5_a0p1_per_feature_seed{42,123,456}`.
Check `metrics.json` carries all four of `interval`, `interval_flat_cp`,
`interval_cluster_cp`, `interval_state_scaled`.

**Seed 42 done** (2026-08-27), `structural_passed: true`, all four interval keys present:
cluster SCCP Winkler `3.8374` (coverage 0.8865, width 2.5167) against flat CP `3.9497`
(0.8936, 2.5899) and state-scaled scalar `3.9584` — a `-0.112` delta with narrower
intervals. First measurement under the symmetric fitting scheme. Seeds 123 and 456 outstanding.

**Resumable loop** (skips a seed whose `metrics.json` already exists; safe to Ctrl+C and re-paste):

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

**20 epochs is enough — verified, not assumed.** The same config at
`--train_epochs 60 --patience 10` early-stops at epoch 27 with its best validation at
epoch 17; validation past epoch 20 rises (1.5022 to 1.5103). In the 20-epoch run the last
five epochs buy 0.224%. Keep the locked `--train_epochs 20 --patience 5`: changing it now
would break comparability with the Step 2 DLinear reference already on disk.

## 3b.0. Headroom diagnostic `[ ]`

No GPU. Paths below are the exact directories Step 3 writes.

```powershell
uv run python scripts/diagnose_conditioning_headroom.py --run_dir ./results/validation/CISSN_ETTh1_M_sl96_pl336_sd5_dm64_fullrevin_lc1_lt0p5_a0p1_per_feature_seed42 --output ./results/validation/conditioning_headroom_revin_seed42.json

uv run python scripts/diagnose_conditioning_headroom.py --run_dir ./results/validation/CISSN_ETTh1_M_sl96_pl336_sd5_dm64_fullrevin_lc1_lt0p5_a0p1_per_feature_seed123 --output ./results/validation/conditioning_headroom_revin_seed123.json

uv run python scripts/diagnose_conditioning_headroom.py --run_dir ./results/validation/CISSN_ETTh1_M_sl96_pl336_sd5_dm64_fullrevin_lc1_lt0p5_a0p1_per_feature_seed456 --output ./results/validation/conditioning_headroom_revin_seed456.json
```

Read `variance_decomposition.per_sample_fraction` (bounds the *scalar* geometry only) and
`summary_vs_flat` win counts. Re-run this per dataset before spending seeds on it —
conditioning is regime-dependent and ETTh1 does not transfer.

**Resumable loop** (skips a seed whose output JSON already exists):

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

## 3b.2. Conditioning selection `[ ]`

Validation only; the test loader is never constructed.

```powershell
uv run python experiments/run_selection.py --data ETTh1 --pred_len 336 --seed 42 --revin --conformal_conditioning scale --scale_geometry per_cell --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --data ETTh1 --pred_len 336 --seed 123 --revin --conformal_conditioning scale --scale_geometry per_cell --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --data ETTh1 --pred_len 336 --seed 456 --revin --conformal_conditioning scale --scale_geometry per_cell --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
```

**Decision**: promote `scale` to default only if `interval_state_scaled` beats **both**
`interval_flat_cp` and `interval_cluster_cp` on Winkler on at least 2 of 3 seeds, read from
each run's `selection.json`. Read no test artifact while deciding.

**Resumable loop** (skips a seed whose `selection.json` already exists — `run_selection.py`
has no `--immutable_artifacts` guard, so it would silently overwrite without this check):

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

## 4. RevIN selection, ETTh2 + weather `[ ]`

Twelve runs: two datasets x three seeds x with/without `--revin`. The paired comparator is
the same command minus `--revin`.

```powershell
uv run python experiments/run_selection.py --data ETTh2 --pred_len 336 --seed 42 --revin --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --data ETTh2 --pred_len 336 --seed 123 --revin --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --data ETTh2 --pred_len 336 --seed 456 --revin --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --data ETTh2 --pred_len 336 --seed 42 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --data ETTh2 --pred_len 336 --seed 123 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --data ETTh2 --pred_len 336 --seed 456 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --data weather --pred_len 336 --seed 42 --revin --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --data weather --pred_len 336 --seed 123 --revin --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --data weather --pred_len 336 --seed 456 --revin --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --data weather --pred_len 336 --seed 42 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --data weather --pred_len 336 --seed 123 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --data weather --pred_len 336 --seed 456 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
```

**Decision**: flip `--revin` to default only if it regresses neither `validation_mse` nor
validation coverage on either dataset. Otherwise keep it opt-in and pass it per dataset.

**Resumable loop** (skips any of the 12 cells whose `selection.json` already exists):

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

## 5. Hybrid variants `[ ]`

Three arms x three seeds. `--state_revin` changes the epoch-0 gradient scale, so treat that
arm as separate rather than a drop-in ablation.

```powershell
uv run python experiments/run_selection.py --architecture hybrid --data ETTh1 --pred_len 336 --seed 42 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --architecture hybrid --data ETTh1 --pred_len 336 --seed 123 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --architecture hybrid --data ETTh1 --pred_len 336 --seed 456 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --architecture hybrid --state_dynamics anchored --data ETTh1 --pred_len 336 --seed 42 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --architecture hybrid --state_dynamics anchored --data ETTh1 --pred_len 336 --seed 123 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --architecture hybrid --state_dynamics anchored --data ETTh1 --pred_len 336 --seed 456 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --architecture hybrid --state_dynamics anchored --state_revin --data ETTh1 --pred_len 336 --seed 42 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --architecture hybrid --state_dynamics anchored --state_revin --data ETTh1 --pred_len 336 --seed 123 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection

uv run python experiments/run_selection.py --architecture hybrid --state_dynamics anchored --state_revin --data ETTh1 --pred_len 336 --seed 456 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
```

**Decision**: promote a variant only on at least 1% mean validation-MSE improvement over
paired DLinear, better on at least 2 of 3 seeds, no worse than 2% on any seed, and no
feature's mean MSE degraded by more than 5%. Ties within 0.5% go to anchored dynamics
without RevIN. If none qualifies, report the hybrid as a negative result and do not retune.

**Resumable loop** (skips any of the 9 cells whose `selection.json` already exists). The
setting names are derived from `build_setting_name` in `run_benchmark.py:44-68` — verified
against real artifacts for the flat/cluster/scale cases above, but **no hybrid run exists
on disk yet to check these three against**; if the first run's directory name doesn't
match what `Test-Path` expects, the loop just re-runs it once and you'll see it in the
listing — it will not silently duplicate work:

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

## Prespecification `[ ]`

No command. Before any sealed dataset is opened, write the primary endpoint and the primary
horizon into `protocol.json` or an adjacent prespecification file, so the manifest carries
the prespecification and a later reader can check the analysis was not chosen after seeing
the results. See `RUNBOOK.md`, "Prespecified statistics".

## Main grid — CISSN `[ ]`

Sealed confirmation runs. `--conformal_conditioning` below is written as `cluster`, the
current default — **change all four lines (or the `$conditioning` variable in the loop
below) to whatever Step 3b.2 selected** before running.

```powershell
uv run python experiments/run_multiseed.py --data ETTh1 --all_horizons --seeds 42,123,456 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --conformal_conditioning cluster --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --output ./results/publication/cissn_ETTh1.json --raw_csv ./results/publication/cissn_ETTh1.csv

uv run python experiments/run_multiseed.py --data ETTh2 --all_horizons --seeds 42,123,456 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --conformal_conditioning cluster --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --output ./results/publication/cissn_ETTh2.json --raw_csv ./results/publication/cissn_ETTh2.csv

uv run python experiments/run_multiseed.py --data weather --all_horizons --seeds 42,123,456 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --conformal_conditioning cluster --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --output ./results/publication/cissn_weather.json --raw_csv ./results/publication/cissn_weather.csv

uv run python experiments/run_multiseed.py --data exchange_rate --all_horizons --seeds 42,123,456 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --conformal_conditioning cluster --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --output ./results/publication/cissn_exchange_rate.json --raw_csv ./results/publication/cissn_exchange_rate.csv
```

**Resumable loop, with a caveat.** `run_multiseed.py` calls `run_benchmark.py` once per
(horizon, seed) internally and has no per-cell skip check of its own: under
`--immutable_artifacts`, re-running a dataset whose horizons already partly completed hits
`require_new_run`'s `FileExistsError` on every already-done cell (`run_multiseed.py:74`)
and, without `--allow_partial`, aborts the whole dataset. `--allow_partial` prevents that
abort, but it then records an already-completed cell in `failed_seeds` too — it cannot
distinguish "this cell errored" from "this cell already exists". The loop below skips a
**whole dataset** only when every horizon x seed cell for it is already on disk; otherwise
it reruns that dataset's full `run_multiseed.py` invocation with `--allow_partial`, and you
must check the printed `failed_seeds` against `results/publication/**/metrics.json`
yourself afterward — a name appearing there despite being listed as failed means it was
already done, not that it broke:

```powershell
$conditioning = 'cluster'  # set to whatever Step 3b.2 selected
$datasets = @(
  @{ data = 'ETTh1';         horizons = 24,96,192,336,720 },
  @{ data = 'ETTh2';         horizons = 24,96,192,336,720 },
  @{ data = 'weather';       horizons = 96,192,336,720 },
  @{ data = 'exchange_rate'; horizons = 96,192,336,720 }
)
$seeds = 42,123,456
foreach ($cell in $datasets) {
  $allDone = $true
  foreach ($h in $cell.horizons) {
    foreach ($seed in $seeds) {
      $setting = "CISSN_$($cell.data)_M_sl96_pl${h}_sd5_dm64_fullrevin_lc1_lt0p5_a0p1_per_feature_seed$seed"
      if (-not (Test-Path "./results/publication/$setting/metrics.json")) { $allDone = $false }
    }
  }
  if ($allDone) {
    Write-Host "skip $($cell.data) (all horizons x seeds already done)"
    continue
  }
  uv run python experiments/run_multiseed.py --data $cell.data --all_horizons --seeds 42,123,456 --allow_partial --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --conformal_conditioning $conditioning --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --output "./results/publication/cissn_$($cell.data).json" --raw_csv "./results/publication/cissn_$($cell.data).csv"
}
```

Note: the setting name above assumes `--revin`, matching every headline run in this repo.
If a dataset's grid cell is run without RevIN, drop `_fullrevin` from that dataset's
`$setting` string.

## Main grid — baselines `[ ]`

270 runs. Horizons differ per dataset: ETT gets `24,96,192,336,720`, weather and
exchange-rate get `96,192,336,720` (registry-enforced). This loop encodes that.

```powershell
$seeds = 42,123,456
$models = 'dlinear','patchtst','deepstate','mc_dropout','deep_ensemble'
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
        uv run python experiments/run_baseline.py --model $model --data $cell.data --pred_len $h --seed $seed --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --multivariate_strategy per_feature --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --checkpoints ./checkpoints/publication --results_dir ./results/publication
      }
    }
  }
}
```

RevIN applies only to `dlinear`, `patchtst`, `mc_dropout`, `deep_ensemble` — never
DeepState — and only once Step 4 has selected it on validation. If Step 4 selects it, add
`--revin` per model inside the loop rather than assuming a default.

**Resumable version of the same loop.** `--immutable_artifacts` makes `require_new_run`
(`cissn/utils/artifacts.py:82`) raise `FileExistsError` and exit `1` if a cell's directory
already exists (`run_baseline.py:1063`) — it never overwrites. This wraps each call to
catch exactly that and continue, rather than letting a completed cell abort the whole grid,
which is what would otherwise happen on every re-paste after a Ctrl+C:

```powershell
$seeds = 42,123,456
$models = 'dlinear','patchtst','deepstate','mc_dropout','deep_ensemble'
$grid = @(
  @{ data = 'ETTh1';         horizons = 24,96,192,336,720 },
  @{ data = 'ETTh2';         horizons = 24,96,192,336,720 },
  @{ data = 'weather';       horizons = 96,192,336,720 },
  @{ data = 'exchange_rate'; horizons = 96,192,336,720 }
)
$total = 0; $skipped = 0; $ran = 0
foreach ($model in $models) {
  foreach ($cell in $grid) {
    foreach ($h in $cell.horizons) {
      foreach ($seed in $seeds) {
        $total++
        uv run python experiments/run_baseline.py --model $model --data $cell.data --pred_len $h --seed $seed --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --multivariate_strategy per_feature --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --checkpoints ./checkpoints/publication --results_dir ./results/publication
        if ($LASTEXITCODE -ne 0) {
          Write-Host "skip/fail: $model $($cell.data) h$h seed$seed (exit $LASTEXITCODE)"
          $skipped++
        } else {
          $ran++
        }
      }
    }
  }
}
Write-Host "`nDone. $ran ran, $skipped skipped-or-failed, $total total."
```

A nonzero exit is **not ambiguous** here, because of how `--immutable_artifacts` writes
results: `create_temporary_result_root`/`finalize_result_directory`
(`cissn/utils/artifacts.py:90-103`) train into a hidden `.{setting}.tmp-<uuid>` sibling
directory and only `os.replace` it into `results/publication/<setting>/` after
`write_completion_manifest` succeeds. So the final directory **never exists at all** until
a run is completely done — a Ctrl+C mid-training leaves no half-written result directory
behind, only an orphaned `.tmp-<uuid>` folder next to it. That means every nonzero exit in
the loop above really is "this cell's final directory already exists," i.e. already done;
there is nothing to distinguish or clean up on the results side.

The orphaned temp folders are harmless (next run gets a fresh uuid) but can be swept
periodically:

```powershell
Get-ChildItem ./results/publication -Directory -Filter '.*.tmp-*' -Force | Remove-Item -Recurse -Force
```

One real gap remains: a genuine crash (bad flag, OOM, corrupt data) also exits nonzero and
looks identical to "already done" in the loop's own output — the `$skipped` counter above
conflates both. If a run you expected to need work reports as skipped, check the terminal
output above that cell's line for a traceback before assuming it succeeded earlier.

## Ablations `[ ]`

Run only after the main comparison completes.

```powershell
uv run python experiments/run_ablation.py --data ETTh1 --pred_len 336 --seed 42 --ablations full,no_structured_A,no_disentanglement_loss,flat_cp,no_correction_mlp,state_dim_4 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --multivariate_strategy per_feature --require_gpu --require_clean_git --output ./results/publication/ablations_ETTh1_h336_s42.json
```

## Publication review `[ ]`

```powershell
uv run python scripts/generate_publication_tables.py --results_root ./results/publication --output_dir ./results/publication/tables
uv run python scripts/generate_publication_figures.py --results_root ./results/publication --output_dir ./results/publication/figures
uv run python scripts/generate_reproducibility_appendix.py --results_root ./results/publication --output ./results/publication/reproducibility.md
```

---

## What to read after each run

| Step | File | Field |
| --- | --- | --- |
| 2, 3 | `sanity.json` | `structural_passed` — the only field that excludes a run. Quality flags are advisory and never drop a result |
| 3 | `metrics.json` | `interval`, `interval_flat_cp`, `interval_cluster_cp`, `interval_state_scaled` — all four must be present |
| 3b.0 | headroom JSON | `variance_decomposition.per_sample_fraction`, `summary_vs_flat` win counts |
| 3b.2, 4, 5 | `selection.json` | `validation_mse` and validation Winkler. Never a test artifact |

Per-cell "wins on 2 of 3 seeds" is a selection rule, not a significance claim: at `n=3` it
has roughly a 50% false-positive rate per cell under a null of no effect.

## Two constraints that bind every launch

- **Confirmation runs need all four flags**: `--evidence_role confirmation` requires
  `--immutable_artifacts --strict_determinism --require_clean_git` together. Without them a
  run defaults to `development` and is ineligible for the publication table regardless of
  its numbers. No artifact currently on disk carries an evidence role.
- **ETTh2, weather, exchange-rate stay sealed** until the prespecification is written and
  the decision lock recorded. Steps 3b.2, 4, and 5 read validation only —
  `run_selection.py` never constructs the test loader, which is what makes those decisions
  legitimate. Do not substitute `run_benchmark.py`.
