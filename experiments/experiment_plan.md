# Experimental Setup for CISSN ArXiv Publication

To ensure a fast and robust publication path, we must demonstrate that CISSN provides **state-of-the-art accuracy** OR **comparable accuracy with superior interpretability and uncertainty quantification**.

## 1. Datasets (The "Standard" Suite)
We will use the standard **Long-Term Time Series Forecasting (LTSF)** benchmarks. Using these ensures reviewers cannot complain about "cherry-picked" data.

| Dataset | Variables | Frequency | Horizon | Description |
| :--- | :--- | :--- | :--- | :--- |
| **ETT (h1, h2, m1, m2)** | 7 | Hourly/15min | 96, 192, 336, 720 | Transformer temperature data. The "Hello World" of LTSF. |
| **Electricity** | 321 | Hourly | 96, 192, 336, 720 | Hourly electricity consumption of 321 clients. High dim. |
| **Traffic** | 862 | Hourly | 96, 192, 336, 720 | Road occupancy rates. Very high dim, complex spatial limits. |
| **Weather** | 21 | 10min | 96, 192, 336, 720 | Local weather metrics. |

**Strategy:**
- Start with **ETTh1** and **ETTh2** for rapid development and tuning.
- Expand to **Electricity** and **Weather** for the paper.
- **Traffic** is optional if we need more evidence, but computationally expensive.

## 2. Baselines
We need to compare against three categories of models:

### A. SOTA Linear/Transformer Models (Accuracy Benchmarks)
- **PatchTST**: Channel-independent patch Transformer. ✅ Implemented in `cissn/baselines/patchtst.py`.
- **DLinear**: Simple linear baseline. ✅ Implemented in `cissn/baselines/dlinear.py`. CISSN **MUST** beat this to be taken seriously.

### B. Probabilistic/State-Space Models (Direct Competitors)
- **DeepState**: GRU + structured SSM with Gaussian intervals. ✅ Implemented in `cissn/baselines/deepstate.py`.
- **MCDropout / DeepEnsemble**: ✅ Implemented in `cissn/baselines/`.

### C. Interpretable Models
- **Prophet / NeuralProphet**: For visual comparison of trend/seasonality decomposition.

## 3. Evaluation Metrics
We will report metrics in two tables:

### Table 1: Deterministic Accuracy (Point Forecasting)
- **MSE (Mean Squared Error)**: Standard.
- **MAE (Mean Absolute Error)**: Standard.

### Table 2: Probabilistic Calibration & Interpretability (The "Selling Point")
- **CRPS (Continuous Ranked Probability Score)**: Measures distribution quality.
- **MSIS (Mean Scaled Interval Score)**: For interval width and coverage.
- **Coverage Error**: Abs(Target Coverage - Actual Coverage).
- **Disentanglement Score**: (Novel metric) Correlation between learned trend and ground-truth trend (using synthetic data).

## 4. Implementation Roadmap

### Phase 1: Data Pipeline
- [x] Implement `cissn.data.dataset.BaseETTDataset` to download and process ETT datasets.
- [x] Create `DataLoaders` consistent with Autoformer/Informer standards (70/10/20 split).

### Phase 2: Benchmarking Engine
- [x] Create `experiments/run_benchmark.py`: standardised trainer with early stopping.
- [x] Integrate **WandB** for logging (`--use_wandb` flag).
- [x] Set global seeds for reproducibility (`--seed` argument).
- [ ] Implement **Rolling Window Evaluation** (walk-forward, crucial for time series).

### Phase 3: Ablation Studies
- **w/o Structure**: Replace structured SSM with standard GRU.
- **w/o Disentanglement Loss**: Train with only MSE.
- **w/o SCCP**: Use standard Conformal Prediction (EnbPI) instead of State-Conditional.

### Phase 4: Visualization & Paper
- [ ] Generate "Component Decomposition" plots (Level, Trend, Seasonal).
- [ ] Generate "Interval Width vs. State" plots (Show how uncertainty adapts to regimes).

## 5. Directory Structure
```
experiments/
├── datasets/           # Raw data
├── baselines/          # Adapter code for baselines
├── configs/            # YAML configs for each dataset
├── run_benchmark.py    # Main entry point
└── analysis.ipynb      # Visualization notebook
```
