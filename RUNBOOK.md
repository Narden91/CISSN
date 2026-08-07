# CISSN Experiment Runbook & Q1 Journal Publication Master Plan

Execution guide for reproducing all experimental results, baselines, ablations, and paper figures for **CISSN** (Conformally Calibrated Interpretable State-Space Networks).

---

## 1. Locked Publication Scope

- **Core Datasets**: `ETTh1`, `ETTh2`, `weather`, `exchange_rate` (plus `ETTm1`, `Electricity`, `Traffic`, `Solar-Energy` for full benchmark evaluation).
- **Random Seeds**: `42`, `123`, `456`.
- **Horizons**: `24`, `96`, `192`, `336`, `720`.
- **Primary Interval Setting**: `conformal_alpha=0.1`, `multivariate_strategy=max`, `n_clusters=5`.
- **Baselines**: `dlinear`, `patchtst`, `deepstate`, `mc_dropout`, `deep_ensemble`, `flat_conformal` (`flat_cp`).
- **Canonical Output Roots**:
  ```text
  ./checkpoints/paper1/
  ./checkpoints/paper1/baselines/
  ./checkpoints/paper1/uq/
  ./results/paper1/
  ./results/paper1_baselines/
  ./results/paper1_uq/
  ./results/paper1_ablations/
  ```

---

## 2. Prerequisites & Environment Setup

### Install Dependencies & Verify Environment
```bash
uv venv
# Windows: .venv\Scripts\Activate.ps1 | Linux/Mac: source .venv/bin/activate
uv pip install -e .
```

### Download Benchmark Datasets
Goal: Fetch raw benchmark CSVs (`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Weather`, `Electricity`, `Traffic`, `ILI`, `Solar-Energy`) into `./data/`.
```bash
uv run python scripts/download_datasets.py
```

### Run Quick Verification Smoke Test
Goal: Ensure data loading, model forward pass, loss computation, SCCP calibration, and evaluation execute cleanly (25/25 tests passing).
```bash
uv run python tests/run_tests.py
uv run python examples/demo_cissn.py
```

---

## 3. Stage 0: Fast Dry-Run / Smoke Test Grid

Goal: Validate experiment scripts and memory allocation with 1 epoch before launching full multi-hour grid.

```bash
# Smoke test main CISSN benchmark runner
uv run python experiments/run_benchmark.py --config experiments/configs/etth1_smoke.yaml

# Smoke test baseline runner
uv run python experiments/run_baseline.py --model dlinear --data ETTh1 --pred_len 24 --seed 42 --train_epochs 1 --patience 1 --batch_size 64 --checkpoints ./checkpoints/smoke_baseline --results_dir ./results/smoke_baseline

# Smoke test ablation runner
uv run python experiments/run_ablation.py --data ETTh1 --pred_len 24 --train_epochs 1 --seed 42 --ablations full,flat_cp --output ./results/ablations_smoke.json
```

---

## 4. Stage 1: Main CISSN Benchmark Grid

Goal: Train CISSN across standard forecasting horizons ($H \in \{24, 96, 192, 336, 720\}$) across multi-seed runs (60 cells total).

### 4.1 Single Dataset Multi-Horizon Runs (ETTh1)
```bash
uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 96 --train_epochs 10 --seed 42
uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 192 --train_epochs 10 --seed 42
uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 336 --train_epochs 10 --seed 42
uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 720 --train_epochs 10 --seed 42
```

### 4.2 Multi-Seed Grid Driver (Publication Tables)
Goal: Automatically run all standard horizons across seeds (42, 123, 456) and compute mean $\pm$ std summary.

```bash
# ETTh1 full horizon grid across 3 seeds
uv run python experiments/run_multiseed.py --data ETTh1 --all_horizons --seeds 42,123,456 --multivariate_strategy max --conformal_alpha 0.1 --patience 5 --n_clusters 5 --output ./results/paper1/cissn_ETTh1.json --raw_csv ./results/paper1/cissn_ETTh1_raw.csv

# ETTh2 full horizon grid across 3 seeds
uv run python experiments/run_multiseed.py --data ETTh2 --all_horizons --seeds 42,123,456 --multivariate_strategy max --conformal_alpha 0.1 --patience 5 --n_clusters 5 --output ./results/paper1/cissn_ETTh2.json --raw_csv ./results/paper1/cissn_ETTh2_raw.csv

# Weather benchmark multi-seed grid
uv run python experiments/run_multiseed.py --data weather --all_horizons --seeds 42,123,456 --multivariate_strategy max --conformal_alpha 0.1 --patience 5 --n_clusters 5 --output ./results/paper1/cissn_weather.json --raw_csv ./results/paper1/cissn_weather_raw.csv

# Exchange Rate benchmark multi-seed grid
uv run python experiments/run_multiseed.py --data exchange_rate --all_horizons --seeds 42,123,456 --multivariate_strategy max --conformal_alpha 0.1 --patience 5 --n_clusters 5 --output ./results/paper1/cissn_exchange_rate.json --raw_csv ./results/paper1/cissn_exchange_rate_raw.csv
```

---

## 5. Stage 2: Point & Probabilistic Baseline Grid

Goal: Evaluate 3 models x 4 datasets x 5 horizons x 3 seeds = 180 cells.

### PowerShell Batch Grid Loop
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

---

## 6. Stage 3: Uncertainty Quantification (UQ) Comparison Subset

Goal: 2 datasets x 3 horizons x 3 seeds = 18 cells per method comparing SCCP vs Flat-CP, MC-Dropout, and Deep Ensemble.

### 6.1 SCCP vs Flat-CP Loop
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

### 6.2 MC-Dropout & Deep Ensemble Loop
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

---

## 7. Stage 4: Ablation Studies

Goal: ETTh1 x 3 horizons x 3 seeds = 9 cells, evaluating all 6 ablation configurations per cell:
1. `full`: Full CISSN model
2. `no_structured_A`: Dense unconstrained transition matrix
3. `no_disentanglement_loss`: $\lambda_{\text{cov}}=0, \lambda_{\text{temp}}=0$
4. `flat_cp`: Marginal conformal prediction without state clustering
5. `no_correction_mlp`: Pure linear encoder without refinement
6. `state_dim_4`: 1D scalar seasonal instead of 2D rotation pair

### PowerShell Ablation Grid Loop
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

---

## 8. Stage 5: Advanced Training & Rolling Protocols

```bash
# Cosine Learning Rate Annealing
uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 96 --lradj cosine --train_epochs 15

# Walk-Forward Rolling Window Evaluation
uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 96 --walk_forward

# Multivariate Conformal Strategy Exploration (per_feature, max, mahalanobis)
uv run python experiments/run_benchmark.py --data ETTh1 --multivariate_strategy mahalanobis
```

---

## 9. Artifact Contract & Publication Gates

### Output Artifacts per Run
Every run directory under `./results/` must contain:
- `metrics.json`: MSE, MAE, PICP, MPIW, Winkler, CRPS, training time.
- `config.json`: Complete hyperparameter CLI dictionary.
- `environment.json`: System hardware, PyTorch version, seed snapshot.
- `pred.npy` & `true.npy`: Prediction and ground truth numpy arrays.
- `lower.npy` & `upper.npy`: Conformal prediction intervals (for interval methods).

### Publication Writing Gates
1. **Methods & Theory**: Draftable immediately.
2. **Results Writing**: Begins only after main CISSN, baseline, UQ, and ablation artifacts exist in `./results/paper1/`.
3. **Table Reporting Rules**:
   - Main tables must report mean $\pm$ standard deviation across seeds 42, 123, 456.
   - UQ tables must report coverage, MPIW, Winkler score, and calibration error.
   - Ablation tables must report MSE, MAE, coverage, MPIW, and calibration error.
