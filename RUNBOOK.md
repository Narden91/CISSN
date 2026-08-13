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

Review `results/validation/*/sanity.json`, `metrics.json`, `history.json`, and `protocol.json`. Every run writes these artifacts. Exclude a run from publication tables only when `structural_passed` is false, which means the artifact is unreadable (empty, non-finite, shape-inconsistent, or inverted interval bounds). Forecast quality never removes a run: `quality.flags` is advisory, and a finite but poor forecast is a valid result that must stay visible. Record the expected full-train reference and the observed fair split result separately; this protocol intentionally reserves train data for calibration.

## Step 3: CISSN end-to-end

```powershell
uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 336 --seed 42 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation
```

Include the result unless `structural_passed` is false. Quality flags in `sanity.json` are advisory and are reported alongside the result, never used to drop it. Check `cluster_stats.json` for fallback clusters and `dependence_diagnostics.json` before interpreting coverage.

## Step 4: hybrid variant selection (validation only)

The legacy architecture routes the whole forecast through the 5-d state, so its
forecast map has rank <= 5 and most of the raw history is unrecoverable. The
hybrid keeps DLinear as the base and gives the state an additive correction:

```text
total = DLinear(history) + LinearCorrectionHead(StateEncoder(history))
```

The correction head is zero-initialised and the base is frozen during
correction training, so correction-stage epoch 0 reproduces the frozen DLinear
exactly and a failed correction stage falls back to it. `runtime.json` records
`base_val_loss` and `correction_improved_on_base` for every hybrid run.

Run the three prespecified variants on ETTh1-h336 with seeds `42,123,456`:

```powershell
uv run python experiments/run_benchmark.py --architecture hybrid --data ETTh1 --pred_len 336 --seed 42 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --strict_artifacts --checkpoints ./checkpoints/selection --results_dir ./results/selection
uv run python experiments/run_benchmark.py --architecture hybrid --state_dynamics anchored --data ETTh1 --pred_len 336 --seed 42 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --strict_artifacts --checkpoints ./checkpoints/selection --results_dir ./results/selection
uv run python experiments/run_benchmark.py --architecture hybrid --state_dynamics anchored --state_revin --data ETTh1 --pred_len 336 --seed 42 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --strict_artifacts --checkpoints ./checkpoints/selection --results_dir ./results/selection
```

Select on **validation** MSE only. Promote a hybrid variant only when it shows
at least `1%` mean improvement over paired DLinear, improves on at least two of
three seeds, is no worse than `2%` on any seed, and degrades no feature's mean
MSE by more than `5%`. Ties within `0.5%` resolve to anchored dynamics without
RevIN.

`--state_revin` rescales the correction by the input-window feature scale. This
preserves epoch-0 exactness but changes the epoch-0 gradient scale, so the RevIN
variant is not comparable to the other two at an equal learning rate; treat it
as a separate arm rather than a drop-in ablation.

If no variant meets these criteria, stop: report the hybrid as a negative result
and do not retune against test data.

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

Before aggregating, every result must have the full artifact contract and a protocol manifest showing the same split, calibration, and shared training settings for the comparison cell. Exclude incomplete results, results with `structural_passed: false`, different splits, and raw-UQ results from the primary table. Do not exclude a run for poor forecast quality; report it with its advisory flags.

```powershell
uv run python scripts/generate_publication_tables.py --results_root ./results/publication --output_dir ./results/publication/tables
uv run python scripts/generate_publication_figures.py --results_root ./results/publication --output_dir ./results/publication/figures
uv run python scripts/generate_reproducibility_appendix.py --results_root ./results/publication --output ./results/publication/reproducibility.md
```

Report mean and standard deviation across outer seeds, paired method comparisons, `coverage_primary`, width, Winkler score, MSIS, coverage scope, interval origin, cluster fallbacks, and dependence diagnostics. Do not describe the time-series result as unconditional distribution-free coverage.
