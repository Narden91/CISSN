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

Include the result unless `structural_passed` is false. Quality flags in `sanity.json` are advisory and are reported alongside the result, never used to drop it. Check `cluster_stats.json` and `scale_stats.json` for fallback clusters and fitted scale coefficients, and `dependence_diagnostics.json`, before interpreting coverage.

## Step 3b: state conditioning mode confirmation

`--conformal_conditioning cluster` (K-Means partition, the historical mechanism) is the
default. `--conformal_conditioning scale` (continuous `sigma(state)`, normalized
conformal score) is an alternative. Every run already calibrates and reports both plus
flat CP regardless of the flag, so this step only decides which one drives the headline
`interval` block and which becomes the CLI default.

**The original evidence for promoting `scale` was withdrawn and then partly replaced.**
The first diagnostic calibrated flat CP on a window twice the size of state-scaled CP's,
so it was not paired. The corrected diagnostic established two things
(`docs/methodology.md`): a **scalar** sigma has essentially no headroom (a scalar oracle
using test labels gains under 1%), and the conditioning signal is a state x cell
interaction that only a **per-cell** sigma or the cluster predictor can express.

Development measurements on ETTh1-h336, three RevIN seeds x four cuts, all methods
sharing the calibration window — mean Winkler delta vs flat CP: cluster `-0.124` (12/12),
scalar sigma `+0.011` (5/12), per-cell sigma `-0.237` (12/12). Under the real protocol on
seed 42 the ordering is cluster `3.5962` < per-cell `3.6916` < flat `3.7869` ~ scalar
`3.7877`.

### Step 3b.0: headroom diagnostic on saved artifacts (do this first, no GPU)

```powershell
uv run python scripts/diagnose_conditioning_headroom.py --run_dir <a RevIN CISSN run dir> --output ./results/validation/conditioning_headroom_revin.json
```

Read `variance_decomposition.per_sample_fraction` (expect ~1%: that bounds the *scalar*
geometry, not conditioning as such) and the `summary_vs_flat` win counts. Run this on
each dataset before spending seeds on it — the conditioning result is regime-dependent
and there is no reason to assume ETTh1's carries over.

### Step 3b.1: protocol confirmation runs

Run both geometries so the comparison stays paired within one model:

```powershell
uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 336 --seed 42 --revin --conformal_conditioning scale --scale_geometry per_cell --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation
```

Repeat for seeds `123,456`. **Decision point**: promote `scale` to the default
conditioning mode only if `interval_state_scaled` beats **both** `interval_flat_cp` and
`interval_cluster_cp` on Winkler on at least 2 of 3 seeds. On the ETTh1 evidence above it
clears the first bar and not the second, so the expected outcome is that `cluster`
remains the default and per-cell state-scaled CP is reported as a mechanism that improves
on flat CP without overtaking the discrete partition. Do not retune against test data.

Record the observed per-seed deltas here either way before proceeding to Step 4.

## Step 4: instance normalisation

Legacy CISSN collapses toward the training mean on ETTh1-h336: the forecast keeps only ~7% of the target's variance, which lowers MSE without tracking the signal. `--revin` removes the level-tracking burden that causes this. Measured over seeds `42,123,456` (see `docs/methodology.md`), test MSE falls `1.280 → 0.771` and coverage moves `0.788 → 0.908` with narrower intervals, with no change to the latent state dimension.

```powershell
uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 336 --seed 42 --revin --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation
```

Check `vali_variance_ratio` in `history.json`. A ratio that falls while validation loss improves is amplitude collapse, not a hard dataset, and must be reported as such rather than attributed to the state bottleneck.

`--revin` is proven only on ETTh1-h336 and remains opt-in (off by default) until
confirmed elsewhere. Before making it the default for `--architecture legacy`, run it
with and without on ETTh2 and weather at h336, 3 seeds each:

```powershell
uv run python experiments/run_benchmark.py --data ETTh2 --pred_len 336 --seed 42 --revin --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation
uv run python experiments/run_benchmark.py --data weather --pred_len 336 --seed 42 --revin --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation
```

Flip the default only if RevIN does not regress test MSE or coverage on either dataset
relative to the non-RevIN arm; if it regresses anywhere, keep it opt-in and pass
`--revin` explicitly per dataset in the main grid rather than defaulting a fix proven on
one dataset/horizon. Record the outcome and per-seed numbers here either way.

## Step 5: hybrid variant selection (validation only)

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

Run CISSN with the multi-seed driver for each locked dataset, using whichever
`--conformal_conditioning` mode Step 3b confirmed (every run still calibrates and
reports both, so this only sets which one is primary):

```powershell
uv run python experiments/run_multiseed.py --data ETTh1 --all_horizons --seeds 42,123,456 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --conformal_conditioning scale --require_gpu --require_clean_git --output ./results/publication/cissn_ETTh1.json --raw_csv ./results/publication/cissn_ETTh1.csv
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

Every `run_benchmark.py` run already reports flat CP, cluster SCCP, and the *selected*
scale geometry paired on the same forecasts, so the main grid covers most conditioning
comparisons without a dedicated ablation arm. Two cells are **not** reachable from it,
and need an added arm if the ablation table requires them: both scale geometries within a
single run (a run fixes one `--scale_geometry`), and any conditioning mode crossed with
an architecture ablation such as `no_structured_A` or `state_dim_4`.

## Publication review

Before aggregating, every result must have the full artifact contract and a protocol manifest showing the same split, calibration, and shared training settings for the comparison cell. Exclude incomplete results, results with `structural_passed: false`, different splits, and raw-UQ results from the primary table. Do not exclude a run for poor forecast quality; report it with its advisory flags.

```powershell
uv run python scripts/generate_publication_tables.py --results_root ./results/publication --output_dir ./results/publication/tables
uv run python scripts/generate_publication_figures.py --results_root ./results/publication --output_dir ./results/publication/figures
uv run python scripts/generate_reproducibility_appendix.py --results_root ./results/publication --output ./results/publication/reproducibility.md
```

`table_paired_comparison.csv` is the central evidence table: per dataset/horizon cell,
the primary conditioning mechanism's mean Winkler delta against flat CP, cluster SCCP,
and state-scaled CP, plus how many seeds it actually won on
(`winkler_delta_vs_*_wins` out of `winkler_delta_vs_*_count`). Report the win count
alongside the mean — a favorable mean can still hide a sign that flips seed to seed, as
it does for cluster-based SCCP against flat CP on some chronological cuts (see
`docs/methodology.md`).

Report mean and standard deviation across outer seeds, paired method comparisons, `coverage_primary`, width, Winkler score, MSIS, coverage scope, interval origin, `worst_slab_coverage`/`max_coverage_deviation` for the conditional-coverage claim, cluster/scale fallbacks, and dependence diagnostics. Do not describe the time-series result as unconditional distribution-free coverage.
