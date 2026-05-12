# Q1 Journal 1 Experiment Master Plan

This file is the source of truth for the first Q1 journal submission experiment execution. Use it for locked scope, runnable commands, artifact rules, and writing gates. Other planning documents should point here instead of repeating the full execution grid.

## Locked Scope

- Core datasets only: `ETTh1`, `ETTh2`, `weather`, `exchange_rate`.
- Seeds only: `42`, `123`, `456`.
- Primary interval setting: `conformal_alpha=0.1`, `multivariate_strategy=max`, `n_clusters=5`.
- Baselines only from the current repo: `dlinear`, `patchtst`, `deepstate`, `mc_dropout`, `deep_ensemble`, plus `flat_cp` through the ablation runner.
- Excluded from Q1 Journal 1 first pass: `iTransformer`, `TimesNet`, `NHITS/N-HiTS`, `SPCI`, `ARIMA`.
- Theory posture for this pass: Theorems 1 and 2 are formalized; Theorems 3 and 4 remain empirical unless proofs are added later.

## Verified State

Verified on 2026-05-12:

- `uv run python tests/run_tests.py` passed `19/19`.
- Local benchmark data files are present.
- Experiment runners available: `experiments/run_benchmark.py`, `experiments/run_multiseed.py`, `experiments/run_baseline.py`, `experiments/run_ablation.py`.
- CISSN smoke completed: `uv run python experiments/run_benchmark.py --config experiments/configs/etth1_smoke.yaml`.
- Baseline smoke completed: DLinear on `ETTh1`, horizon `24`, seed `42`, `1` epoch.
- Ablation smoke completed: `full,flat_cp` on `ETTh1`, horizon `24`, seed `42`, `1` epoch.

Smoke artifacts already written:

- CISSN smoke: `./results/CISSN_ETTh1_M_sl96_pl24_sd5_dm32_lc1_lt0p5_a0p1_max_seed42/`
- Baseline smoke: `./results/smoke_baseline/BASELINE_dlinear_ETTh1_M_sl96_pl24_seed42/`
- Ablation smoke: `./results/ablations_smoke.json`

## Canonical Output Roots

Do not reuse smoke directories for publication runs.

```text
./checkpoints/paper1/
./checkpoints/paper1/baselines/
./checkpoints/paper1/uq/
./results/paper1/
./results/paper1_baselines/
./results/paper1_uq/
./results/paper1_ablations/
```

## Execution Order

### Phase 0: Pre-Flight

- [x] Baseline runner implemented and smoke-validated.
- [x] Repo tests rerun.
- [ ] Freeze the publication output roots above before launching long grids.
- [ ] Avoid manual renaming or overwriting of completed run directories.

### Phase 1: Smoke Runs

- [x] CISSN smoke.
- [x] Baseline smoke.
- [x] Ablation smoke.

Commands already used:

```powershell
uv run python experiments/run_benchmark.py --config experiments/configs/etth1_smoke.yaml
uv run python experiments/run_baseline.py --model dlinear --data ETTh1 --pred_len 24 --seed 42 --train_epochs 1 --patience 1 --batch_size 64 --checkpoints ./checkpoints/smoke_baseline --results_dir ./results/smoke_baseline
uv run python experiments/run_ablation.py --data ETTh1 --pred_len 24 --train_epochs 1 --seed 42 --ablations full,flat_cp --output ./results/ablations_smoke.json
```

### Phase 2: Main CISSN Grid

Target coverage: `4 datasets x 5 horizons x 3 seeds = 60 cells`.

- [ ] Run `ETTh1` full grid.
- [ ] Run `ETTh2` full grid.
- [ ] Run `weather` full grid.
- [ ] Run `exchange_rate` full grid.
- [ ] Verify one aggregate JSON plus one raw CSV per dataset.

Commands:

```powershell
uv run python experiments/run_multiseed.py --data ETTh1 --all_horizons --seeds 42,123,456 --multivariate_strategy max --conformal_alpha 0.1 --patience 5 --n_clusters 5 --output ./results/paper1/cissn_ETTh1.json --raw_csv ./results/paper1/cissn_ETTh1_raw.csv
uv run python experiments/run_multiseed.py --data ETTh2 --all_horizons --seeds 42,123,456 --multivariate_strategy max --conformal_alpha 0.1 --patience 5 --n_clusters 5 --output ./results/paper1/cissn_ETTh2.json --raw_csv ./results/paper1/cissn_ETTh2_raw.csv
uv run python experiments/run_multiseed.py --data weather --all_horizons --seeds 42,123,456 --multivariate_strategy max --conformal_alpha 0.1 --patience 5 --n_clusters 5 --output ./results/paper1/cissn_weather.json --raw_csv ./results/paper1/cissn_weather_raw.csv
uv run python experiments/run_multiseed.py --data exchange_rate --all_horizons --seeds 42,123,456 --multivariate_strategy max --conformal_alpha 0.1 --patience 5 --n_clusters 5 --output ./results/paper1/cissn_exchange_rate.json --raw_csv ./results/paper1/cissn_exchange_rate_raw.csv
```

### Phase 3: Point Baseline Grid

Target coverage: `3 models x 4 datasets x 5 horizons x 3 seeds = 180 cells`.

- [ ] Run DLinear on the full grid.
- [ ] Run PatchTST on the full grid.
- [ ] Run DeepState on the full grid.
- [ ] Verify each dataset-horizon-seed-model cell creates exactly one result directory.

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

### Phase 4: UQ Comparison Subset

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

### Phase 5: Ablations

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

## Artifact Contract

- Every reported run must save `metrics.json`, `config.json`, `environment.json`, `runtime.json`, `pred.npy`, and `true.npy`.
- Interval-producing methods must also save `lower.npy` and `upper.npy`.
- Main CISSN grids must also retain aggregate JSON and raw CSV outputs from `run_multiseed.py`.
- After each grid, verify every expected dataset-horizon-seed cell exists exactly once.

## Publication Gates

- Methods, related work, and theory drafting can proceed now.
- Results writing starts only after main CISSN, baseline, UQ, and ablation artifacts exist.
- Main tables must report mean plus or minus standard deviation over the three seeds.
- Interval tables must report coverage, MPIW, Winkler, and calibration error.
- Ablation tables must report MSE, MAE, coverage, MPIW, and calibration error.
- Claims tied to Theorems 3 and 4 stay empirical unless formal proofs are added later.

## Document Map

- `document.md`: canonical Q1 Journal 1 experiment execution plan.
- `docs/cissn_intuition.md`: intuition-first overview of how CISSN works.
- `plan.md`: repo readiness summary and implementation status.
- `publication/paper1_framework.md`: theory-facing framework, claims, and proof status.
- `manuscript/README.md`: manuscript structure and writing outline.
- `CLAUDE.md`: repository guide for assistants and future development passes.