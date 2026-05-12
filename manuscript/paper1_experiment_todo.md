# Paper 1 Experiment Execution Todo

## Locked Scope

- Core datasets only: `ETTh1`, `ETTh2`, `weather`, `exchange_rate`.
- Seeds only: `42`, `123`, `456`.
- Primary interval setting: `conformal_alpha=0.1`, `multivariate_strategy=max`, `n_clusters=5`.
- Baselines only from the current repo: `dlinear`, `patchtst`, `deepstate`, `mc_dropout`, `deep_ensemble`, plus `flat_cp` through the ablation runner.
- Excluded from Paper 1 first pass: `iTransformer`, `TimesNet`, `NHITS/N-HiTS`, `SPCI`, `ARIMA`.

## Current Status

- [x] `experiments/run_baseline.py` implemented for `dlinear`, `patchtst`, `deepstate`, `mc_dropout`, and `deep_ensemble`.
- [x] Baseline runner writes `metrics.json`, `config.json`, `environment.json`, `runtime.json`, `pred.npy`, and `true.npy`, plus `lower.npy` and `upper.npy` for interval-capable baselines.
- [x] Repo tests rerun on 2026-05-12: `uv run python tests/run_tests.py` passed `19/19`.
- [x] CISSN smoke completed: `uv run python experiments/run_benchmark.py --config experiments/configs/etth1_smoke.yaml`.
- [x] Baseline smoke completed: DLinear on `ETTh1`, horizon `24`, seed `42`, `1` epoch. Artifacts were written under `results/smoke_baseline/BASELINE_dlinear_ETTh1_M_sl96_pl24_seed42`.
- [x] Ablation smoke completed: `full,flat_cp` on `ETTh1`, horizon `24`, seed `42`, `1` epoch. Output written to `results/ablations_smoke.json`.

## Phase 0: Pre-Experiment Gate

- [x] Clear the only code blocker: baseline runner exists and is smoke-validated.
- [x] Confirm local data availability and test health before starting large grids.
- [ ] Freeze the exact output roots for publication runs and do not reuse smoke directories.

Recommended publication roots:

```text
./checkpoints/paper1/
./results/paper1/
./results/paper1_baselines/
./results/paper1_uq/
./results/paper1_ablations/
```

## Phase 1: Smoke Checklist

- [x] CISSN smoke: `uv run python experiments/run_benchmark.py --config experiments/configs/etth1_smoke.yaml`
- [x] Baseline smoke: DLinear, `ETTh1`, horizon `24`, seed `42`, `1` epoch
- [x] Ablation smoke: `full,flat_cp`, `ETTh1`, horizon `24`, seed `42`, `1` epoch

Baseline smoke command used:

```powershell
uv run python experiments/run_baseline.py --model dlinear --data ETTh1 --pred_len 24 --seed 42 --train_epochs 1 --patience 1 --batch_size 64 --checkpoints ./checkpoints/smoke_baseline --results_dir ./results/smoke_baseline
```

Ablation smoke command used:

```powershell
uv run python experiments/run_ablation.py --data ETTh1 --pred_len 24 --train_epochs 1 --seed 42 --ablations full,flat_cp --output ./results/ablations_smoke.json
```

## Phase 2: Main CISSN Grid

Target coverage: `4 datasets x 5 horizons x 3 seeds = 60 cells`.

- [ ] Run `ETTh1` full grid.
- [ ] Run `ETTh2` full grid.
- [ ] Run `weather` full grid.
- [ ] Run `exchange_rate` full grid.
- [ ] Verify each dataset writes one aggregate JSON and one raw CSV with mean and standard deviation over the three seeds.

Commands:

```powershell
uv run python experiments/run_multiseed.py --data ETTh1 --all_horizons --seeds 42,123,456 --multivariate_strategy max --conformal_alpha 0.1 --patience 5 --n_clusters 5 --output ./results/paper1/cissn_ETTh1.json --raw_csv ./results/paper1/cissn_ETTh1_raw.csv
uv run python experiments/run_multiseed.py --data ETTh2 --all_horizons --seeds 42,123,456 --multivariate_strategy max --conformal_alpha 0.1 --patience 5 --n_clusters 5 --output ./results/paper1/cissn_ETTh2.json --raw_csv ./results/paper1/cissn_ETTh2_raw.csv
uv run python experiments/run_multiseed.py --data weather --all_horizons --seeds 42,123,456 --multivariate_strategy max --conformal_alpha 0.1 --patience 5 --n_clusters 5 --output ./results/paper1/cissn_weather.json --raw_csv ./results/paper1/cissn_weather_raw.csv
uv run python experiments/run_multiseed.py --data exchange_rate --all_horizons --seeds 42,123,456 --multivariate_strategy max --conformal_alpha 0.1 --patience 5 --n_clusters 5 --output ./results/paper1/cissn_exchange_rate.json --raw_csv ./results/paper1/cissn_exchange_rate_raw.csv
```

## Phase 3: Point Baseline Grid

Target coverage: `3 models x 4 datasets x 5 horizons x 3 seeds = 180 cells`.

- [ ] Run DLinear on the full Paper 1 grid.
- [ ] Run PatchTST on the full Paper 1 grid.
- [ ] Run DeepState on the full Paper 1 grid.
- [ ] Verify each dataset-horizon-seed-model cell creates exactly one result directory with no overwrite.

PowerShell loop:

```powershell
$models = "dlinear", "patchtst", "deepstate"
$datasets = "ETTh1", "ETTh2", "weather", "exchange_rate"
$horizons = 24, 96, 192, 336, 720
$seeds = 42, 123, 456

foreach ($model in $models) {
    foreach ($data in $datasets) {
        foreach ($h in $horizons) {
            foreach ($seed in $seeds) {
                uv run python experiments/run_baseline.py --model $model --data $data --pred_len $h --seed $seed --conformal_alpha 0.1 --patience 5 --checkpoints ./checkpoints/paper1/baselines --results_dir ./results/paper1_baselines
            }
        }
    }
}
```

## Phase 4: UQ Comparison Subset

Target coverage: `2 datasets x 3 horizons x 3 seeds = 18 cells per method`.

- [ ] Run SCCP vs Flat-CP on `ETTh1` and `weather` for horizons `96`, `336`, `720`.
- [ ] Run `mc_dropout` on the same subset.
- [ ] Run `deep_ensemble` on the same subset.
- [ ] Keep `full` and `flat_cp` paired in the same JSON artifact for direct comparison.

SCCP vs Flat-CP loop:

```powershell
$datasets = "ETTh1", "weather"
$horizons = 96, 336, 720
$seeds = 42, 123, 456

foreach ($data in $datasets) {
    foreach ($h in $horizons) {
        foreach ($seed in $seeds) {
            uv run python experiments/run_ablation.py --data $data --pred_len $h --seed $seed --ablations full,flat_cp --output "./results/paper1_uq/${data}_h${h}_s${seed}_sccp_vs_flatcp.json"
        }
    }
}
```

MC-Dropout and Deep Ensemble loop:

```powershell
$models = "mc_dropout", "deep_ensemble"
$datasets = "ETTh1", "weather"
$horizons = 96, 336, 720
$seeds = 42, 123, 456

foreach ($model in $models) {
    foreach ($data in $datasets) {
        foreach ($h in $horizons) {
            foreach ($seed in $seeds) {
                uv run python experiments/run_baseline.py --model $model --data $data --pred_len $h --seed $seed --conformal_alpha 0.1 --patience 5 --ensemble_seeds 42,123,456 --checkpoints ./checkpoints/paper1/uq --results_dir ./results/paper1_uq
            }
        }
    }
}
```

## Phase 5: Ablations

Target coverage: `ETTh1 x 3 horizons x 3 seeds = 9 cells`, each cell containing all six ablation results.

- [ ] Run `full`.
- [ ] Run `no_structured_A`.
- [ ] Run `no_disentanglement_loss`.
- [ ] Run `flat_cp`.
- [ ] Run `no_correction_mlp`.
- [ ] Run `state_dim_4`.

PowerShell loop:

```powershell
$horizons = 96, 336, 720
$seeds = 42, 123, 456
$ablations = "full,no_structured_A,no_disentanglement_loss,flat_cp,no_correction_mlp,state_dim_4"

foreach ($h in $horizons) {
    foreach ($seed in $seeds) {
        uv run python experiments/run_ablation.py --data ETTh1 --pred_len $h --seed $seed --ablations $ablations --output "./results/paper1_ablations/ETTh1_h${h}_s${seed}.json"
    }
}
```

## Verification Rules After Each Batch

- [ ] Before any publication grid, rerun `uv run python tests/run_tests.py`.
- [ ] After each baseline smoke or grid launch, verify `metrics.json` plus `pred.npy` and `true.npy` exist.
- [ ] For interval-producing methods, verify `lower.npy` and `upper.npy` exist.
- [ ] After each grid, verify every expected cell exists exactly once and no result directory was overwritten.
- [ ] Do not start table writing until the corresponding JSON and CSV artifacts exist.

## Publication Acceptance Criteria

- [ ] Main tables report mean plus or minus standard deviation over `3` seeds.
- [ ] Interval tables report coverage, MPIW, Winkler, and calibration error.
- [ ] Ablation table reports MSE, MAE, coverage, MPIW, and calibration error.
- [ ] Every reported result has saved `config.json`, `environment.json`, `runtime.json`, and raw arrays.

## Writing Gates

- [x] Methods section can start now.
- [x] Related work section can start now.
- [x] Theory section can start now.
- [ ] Results section starts only after the main CISSN grid, baseline grid, UQ subset, and ablation artifacts exist.
- [ ] Claims tied to Theorems 3 and 4 stay empirical unless formal proofs are added later.