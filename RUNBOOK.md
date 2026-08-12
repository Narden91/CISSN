# CISSN publication runbook

This is the sole launch protocol. Results from earlier protocols are not publication evidence and must not be pooled with these artifacts.

## Locked design

- Datasets: `ETTh1`, `ETTh2`, `weather`, `exchange_rate`.
- Seeds: `42,123,456`.
- Horizons: registry-supported horizons only; use `24,96,192,336,720` for ETT and `96,192,336,720` for weather and exchange-rate.
- Shared training: `--train_epochs 20 --patience 5 --lradj cosine --batch_size 128`.
- Shared calibration: chronological train-tail calibration split, `--cal_fraction 0.2`, `--conformal_alpha 0.1`.
- Primary interval geometry: `--multivariate_strategy per_feature`; report `coverage_primary` (marginal). `max` is a separate simultaneous-coverage analysis.
- Publication safeguards: `--require_gpu --require_clean_git`.

The state partition is learned only from train states, then calibration uses the later calibration split. Serial dependence is documented in artifacts; it is not converted into a coverage guarantee.

All experiment runners show live batch progress for training, validation, partitioning, calibration, and testing. Use `--no_progress` only for CI or captured logs.

## Step 1: environment, data, tests

```powershell
uv sync
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
uv run python scripts/verify_datasets.py
uv run python tests/run_tests.py
```

Stop if any command fails. `verify_datasets.py` must pass for the four locked datasets.

## Step 2: DLinear reference reproduction

```powershell
uv run python experiments/run_baseline.py --model dlinear --data ETTh1 --pred_len 336 --seed 42 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --multivariate_strategy per_feature --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation
```

Review `results/validation/*/sanity.json`, `metrics.json`, `history.json`, and `protocol.json`. Every run writes these artifacts, even when `sanity_passed` is false. Do not include a failed review in publication tables. Record the expected full-train reference and the observed fair split result separately; this protocol intentionally reserves train data for calibration.

## Step 3: CISSN end-to-end

```powershell
uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 336 --seed 42 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation
```

Include the result in publication tables only when `sanity_passed: true`. Check `cluster_stats.json` for fallback clusters and `dependence_diagnostics.json` before interpreting coverage.

## Main grid

Run CISSN with the multi-seed driver for each locked dataset:

```powershell
uv run python experiments/run_multiseed.py --data ETTh1 --all_horizons --seeds 42,123,456 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --require_gpu --require_clean_git --output ./results/publication/cissn_ETTh1.json --raw_csv ./results/publication/cissn_ETTh1.csv
```

Repeat with `ETTh2`, `weather`, and `exchange_rate`. The driver must propagate all safeguards; verify this with `--help` before a long launch.

For baselines, run each model/dataset/horizon/seed with the same shared options:

```powershell
$models = 'dlinear','patchtst','deepstate','mc_dropout','deep_ensemble'
$datasets = 'ETTh1','ETTh2','weather','exchange_rate'
$seeds = 42,123,456
foreach ($model in $models) {
  foreach ($data in $datasets) {
    foreach ($h in 96,192,336,720) {
      foreach ($seed in $seeds) {
        uv run python experiments/run_baseline.py --model $model --data $data --pred_len $h --seed $seed --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --multivariate_strategy per_feature --require_gpu --require_clean_git --checkpoints ./checkpoints/publication --results_dir ./results/publication
      }
    }
  }
}
```

For ETT, add the horizon `24`. For UQ models the default is conformalized intervals; raw output requires explicit `--uq_interval_mode raw` and is secondary.

Run ablations only after the main comparison completes:

```powershell
uv run python experiments/run_ablation.py --data ETTh1 --pred_len 336 --seed 42 --ablations full,no_structured_A,no_disentanglement_loss,flat_cp,no_correction_mlp,state_dim_4 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --multivariate_strategy per_feature --require_gpu --require_clean_git --output ./results/publication/ablations_ETTh1_h336_s42.json
```

## Publication review

Before aggregating, every result must have the full artifact contract and a protocol manifest showing the same split, calibration, and shared training settings for the comparison cell. Exclude incomplete results, results with `sanity_passed: false`, different splits, and raw-UQ results from the primary table.

```powershell
uv run python scripts/generate_publication_tables.py --results_root ./results/publication --output_dir ./results/publication/tables
uv run python scripts/generate_publication_figures.py --results_root ./results/publication --output_dir ./results/publication/figures
uv run python scripts/generate_reproducibility_appendix.py --results_root ./results/publication --output ./results/publication/reproducibility.md
```

Report mean and standard deviation across outer seeds, paired method comparisons, `coverage_primary`, width, Winkler score, MSIS, coverage scope, interval origin, cluster fallbacks, and dependence diagnostics. Do not describe the time-series result as unconditional distribution-free coverage.
