# CISSN Manuscript

**Working Title:** Conformal Inference via State-Conditional Disentangled State-Space Networks for Time-Series Forecasting

**Target Venue:** ICML / NeurIPS / ICLR (primary); IEEE TNNLS / Pattern Recognition (journal fallback)

**Deadline (tentative):** submission window Q3 2026

---

## Research Hypotheses

1. **H1 – Disentanglement improves coverage.** State-conditional conformal intervals are narrower and better-calibrated when the encoder imposes disentangled latent dynamics than when it does not.
2. **H2 – State conditioning beats marginal conformal.** Conditioning on the latent state at prediction time yields tighter intervals than a flat (marginal) conformal baseline, at equal empirical coverage.
3. **H3 – Scalability.** CISSN scales to multivariate long-horizon settings (prediction length ≥ 336) without degradation in calibration or point accuracy compared to Transformer-based baselines.

---

## IMRAD Structure

### 1. Introduction

- [ ] Motivate reliable uncertainty quantification in time-series forecasting (finance, health, energy)
- [ ] Gap: most deep-learning forecasting models produce point estimates; conformal prediction is distribution-free but does not exploit latent structure
- [ ] Claim: structured disentanglement + state-conditional conformal = principled, efficient UQ
- [ ] Contributions (bullet list):
  - Novel Disentangled State-Space Encoder (CISSN encoder) with a structured-SSM dynamics layer
  - State-Conditional Conformal Prediction (SCCP) framework that leverages encoder hidden states
  - Theoretical coverage guarantee (exchangeability argument) under distribution shift bounded by state similarity
  - Empirical evaluation on ETT, Exchange, Weather, Traffic, ILI benchmarks

### 2. Related Work

- [ ] State-space models for sequences (S4, Mamba, LRU)
- [ ] Disentangled representation learning (β-VAE, FactorVAE; time-series variants)
- [ ] Conformal prediction (Vovk et al.; split CP; adaptive CP; time-series CP – SPCI, EnbPI)
- [ ] Uncertainty in forecasting (MC-Dropout, Deep Ensembles, Bayesian RNNs)

### 3. Methods

#### 3.1 Problem Formulation
- [ ] Define multivariate time series, look-back window, prediction horizon
- [ ] Define point forecast and interval forecast objectives

#### 3.2 Disentangled State-Space Encoder
- [ ] Structured dynamics matrix (local, trend, global, cross components)
- [ ] Input projection + recurrent update rule
- [ ] Disentanglement losses: covariance penalty + temporal independence penalty
- [ ] Complexity analysis (O(L·d) per sample)

#### 3.3 Forecast Head
- [ ] Linear direct-decoding path
- [ ] MLP refinement residual
- [ ] Contribution decomposition (linear vs. refinement per horizon/output cell)

#### 3.4 State-Conditional Conformal Prediction
- [ ] Calibration: residual computation per state cluster / KDE weighting
- [ ] Multivariate strategies: `max`, `scalar`, `per_output`
- [ ] Coverage guarantee proof sketch (finite-sample, distribution-free)
- [ ] Computational cost of SCCP at inference

#### 3.5 Training Procedure
- [ ] Loss: MSE + α·covariance_loss + β·temporal_loss
- [ ] Optimizer, LR schedule, early stopping criteria
- [ ] Conformal calibration split (separate from validation set)

### 4. Experiments

#### 4.1 Datasets and Setup
- [ ] ETTh1, ETTh2, ETTm1, ETTm2 (hourly/minute electricity transformer temperature)
- [ ] Exchange-Rate, Weather, Traffic (standard long-term benchmarks)
- [ ] ILI (short-series, long-horizon)
- [ ] Splits: train 60%, val 20%, test 20% (or dataset-standard splits)
- [ ] Metrics: MSE, MAE (point); Coverage Rate, Interval Width, PICP, MPIW (interval)

#### 4.2 Baselines
- [ ] Point forecast: PatchTST, iTransformer, TimesNet, DLinear, NHITS
- [ ] UQ: Flat conformal (same encoder, marginal CP), MC-Dropout, Deep Ensemble (3-seed), SPCI

#### 4.3 Main Results
- [ ] Table: MSE/MAE vs. baselines across all datasets and horizons {24, 96, 192, 336, 720}
- [ ] Table: Coverage + Width vs. UQ baselines at nominal coverage {90%, 95%}

#### 4.4 Ablations
- [ ] No disentanglement loss (α=0, β=0)
- [ ] No MLP refinement (linear head only)
- [ ] Marginal CP vs. state-conditional CP (same encoder, SCCP disabled)
- [ ] Multivariate strategy: max vs. scalar vs. per_output
- [ ] Encoder depth (hidden_dim ∈ {32, 64, 128, 256})

#### 4.5 Analysis
- [ ] State-space visualization (t-SNE / PCA of latent states, colored by regime)
- [ ] Contribution attribution heatmap (linear vs. refinement per horizon × output)
- [ ] Coverage calibration curve (reliability diagram) per dataset
- [ ] Runtime vs. sequence length plot (encoder + SCCP)

### 5. Discussion

- [ ] Interpret H1–H3 in light of results
- [ ] Limitations: exchangeability assumption, calibration set size, computational cost of KDE-based SCCP
- [ ] Failure modes: heavy distribution shift, very short calibration sets
- [ ] Future work: online/streaming conformal, multi-step state conditioning, plug-in to other SSMs

### 6. Conclusion

- [ ] One-paragraph summary of contributions and findings

### Appendix

- [ ] A: Full proof of coverage guarantee
- [ ] B: Dataset statistics table
- [ ] C: Hyperparameter sensitivity (learning rate, α, β sweep)
- [ ] D: Extended results tables (all horizons × all datasets)
- [ ] E: Reproducibility checklist

---

## Experiments TODO

### E1 – Codebase / Pipeline Readiness

- [ ] Verify 13/13 unit tests pass on GPU build (`python tests/run_tests.py`)
- [ ] Add dataset download script for ETTh2, ETTm1, ETTm2, Weather, Exchange, Traffic, ILI
- [ ] Unify config schema: single `config.yaml` drives all experiments (dataset, model, training, conformal)
- [ ] Implement multi-dataset experiment runner (loop over dataset × horizon grid)
- [ ] Add checkpoint versioning (include dataset name + horizon + seed in checkpoint path)
- [ ] Implement early stopping with patience=5 on validation MSE

### E2 – Baselines

- [ ] Integrate DLinear baseline (single linear layer; no encoder)
- [ ] Integrate PatchTST inference wrapper (load pretrained or re-train from scratch)
- [ ] Integrate iTransformer inference wrapper
- [ ] Flat conformal wrapper: same CISSN encoder, replace SCCP with marginal split-CP
- [ ] MC-Dropout wrapper: add dropout to forecast head, sample N=50 forward passes at test time
- [ ] Deep Ensemble wrapper: train 3 seeds, average predictions, use std as uncertainty proxy
- [ ] SPCI baseline: sequential predictive conformal intervals (Xu & Xie 2023)

### E3 – Evaluation Metrics

- [ ] Point metrics: MSE, MAE (already in run_benchmark.py)
- [ ] Interval metrics: empirical coverage rate (ECR), mean prediction interval width (MPIW), PICP
- [ ] Winkler score (combines coverage + width in one scalar)
- [ ] Conditional coverage: ECR stratified by quantile of |state| norm
- [ ] Calibration curve: plot empirical coverage vs. nominal α for α ∈ [0.05, 0.5]

### E4 – Main Benchmark Run

- [ ] ETTh1: horizons {24, 96, 192, 336, 720} × 3 random seeds
- [ ] ETTh2: same grid
- [ ] ETTm1: same grid
- [ ] ETTm2: same grid
- [ ] Weather: horizons {96, 192, 336, 720}
- [ ] Exchange: horizons {96, 192, 336, 720}
- [ ] Traffic: horizons {96, 192, 336, 720}
- [ ] ILI: horizons {24, 36, 48, 60}
- [ ] Aggregate results into LaTeX tables (mean ± std over seeds)

### E5 – Ablations

- [ ] CISSN (full) vs. no-disentanglement (α=0, β=0): ETTh1, all 5 horizons
- [ ] CISSN (full) vs. linear-head-only (no MLP refinement): ETTh1, ETTm1
- [ ] Full SCCP vs. marginal CP: ETTh1, Weather (coverage + width)
- [ ] SCCP strategy: max vs. scalar vs. per_output on multivariate datasets
- [ ] Hidden dim sweep {32, 64, 128, 256}: ETTh1 horizon=96
- [ ] α sweep {0, 0.01, 0.1, 1.0}: covariance loss weight
- [ ] β sweep {0, 0.01, 0.1, 1.0}: temporal loss weight

### E6 – Analysis / Visualization

- [ ] PCA of latent states on ETTh1 test set, colored by time-of-day / season
- [ ] t-SNE of latent states on Weather test set
- [ ] Contribution heatmap (linear vs. refinement) for a representative ETTh1 test window
- [ ] Coverage reliability diagram for CISSN-SCCP vs. flat-CP on ETTh1 and Weather
- [ ] Runtime scaling: encoder inference time vs. L ∈ {24, 96, 192, 336, 720} on GPU

### E7 – Robustness / Stress Tests

- [ ] Distribution shift: train on ETTh1 first 60%, test on last 20% with 2-year gap (no val leakage)
- [ ] Calibration set size sensitivity: vary |D_cal| ∈ {50, 100, 200, 500, 1000}; measure ECR and MPIW
- [ ] Noise robustness: add Gaussian noise σ ∈ {0.01, 0.05, 0.1} to test inputs; measure ECR degradation
- [ ] Missing data: randomly mask 10%/20% of input features; impute with mean; measure MSE and ECR

### E8 – Reproducibility

- [ ] Seed all experiments: `torch.manual_seed`, `numpy.random.seed`, `random.seed` in config
- [ ] Log all hyperparameters to JSON alongside checkpoints
- [ ] Provide `requirements.txt`/`pyproject.toml` with pinned versions
- [ ] Write a `reproduce.sh` (or `reproduce.ps1`) that re-runs all main table experiments from scratch
- [ ] Upload checkpoints for best model per dataset to a public store (Zenodo / HuggingFace Hub)

---

## Benchmark Datasets

Standard long-term forecasting benchmark suite following Autoformer / PatchTST / iTransformer conventions. All datasets are publicly available (links below).

### Dataset Statistics

| # | Name | Domain | Freq | Features (M) | Total length | Train | Val | Test | Horizons |
|---|------|--------|------|:---:|---:|---:|---:|---:|---:|
| 1 | **ETTh1** | Electricity (transformer temp.) | 1 h | 7 | 17 420 | 8 640 | 2 880 | 2 880 | 24 / 96 / 192 / 336 / 720 |
| 2 | **ETTh2** | Electricity (transformer temp.) | 1 h | 7 | 17 420 | 8 640 | 2 880 | 2 880 | 24 / 96 / 192 / 336 / 720 |
| 3 | **ETTm1** | Electricity (transformer temp.) | 15 min | 7 | 69 680 | 34 560 | 11 520 | 11 520 | 24 / 96 / 192 / 336 / 720 |
| 4 | **ETTm2** | Electricity (transformer temp.) | 15 min | 7 | 69 680 | 34 560 | 11 520 | 11 520 | 24 / 96 / 192 / 336 / 720 |
| 5 | **Weather** | Meteorology (21 weather vars.) | 10 min | 21 | 52 696 | 36 887 | 5 270 | 10 540 | 96 / 192 / 336 / 720 |
| 6 | **Exchange-Rate** | Finance (8 sovereign FX rates) | 1 day | 8 | 7 588 | 5 120 | 760 | 1 517 | 96 / 192 / 336 / 720 |
| 7 | **Electricity (ECL)** | Energy (utility consumption) | 1 h | 321 | 26 304 | 18 413 | 2 633 | 5 261 | 96 / 192 / 336 / 720 |
| 8 | **Traffic** | Transport (road occupancy rate) | 1 h | 862 | 17 544 | 12 281 | 1 757 | 3 509 | 96 / 192 / 336 / 720 |
| 9 | **ILI** | Epidemiology (influenza-like illness) | 1 week | 7 | 966 | 676 | 97 | 193 | 24 / 36 / 48 / 60 |
| 10 | **Solar-Energy** | Renewable energy (PV production) | 10 min | 137 | 52 560 | 36 792 | 5 256 | 10 512 | 96 / 192 / 336 / 720 |

> **Split convention**: ETT datasets use the Autoformer fixed splits (12/4/4 months for ETTh; 48/16/16 months for ETTm). All other datasets use a 7:1:2 ratio by chronological order. No shuffling.

### Dataset Notes

| # | Key characteristics | Challenges |
|---|---|---|
| 1–4 | ETT family: 6 oil temperature features + 1 target (OT); H = hourly, M = 15-min granularity | Intra-day seasonality, seasonal regime shifts between summer and winter |
| 5 | Weather: 21 vars including temperature, humidity, wind speed, CO₂ | High-dimensional correlations, missing noon/night patterns |
| 6 | Exchange-Rate: 8 daily FX rates (AUD, GBP, CAD, CHF, CNY, JPY, NZD, SGD) | Near-random-walk behavior at short horizons; near-zero autocorrelation |
| 7 | ECL: 321 clients' hourly electricity load; large multivariate setup | Extreme dimensionality; week-day/weekend patterns; sparse cross-feature correlation |
| 8 | Traffic: 862 San Francisco freeway sensors; very wide matrix | Very high-dimensional; spatio-temporal correlations; strong intra-day cycle |
| 9 | ILI: CDC weekly influenza surveillance; only 966 observations total | Very short series; long relative horizon (H/T up to 0.06); high variability |
| 10 | Solar-Energy: Alabama 137-station 10-min PV data | Strong diurnal cycle; near-zero at night (distribution shift between day/night) |

### Download Sources

- **ETT (E1–E4)**: [github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) — `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`
- **Weather**: [drive.google.com/…](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) (Autoformer repo)
- **Exchange-Rate**: same Autoformer Google Drive link (`exchange_rate.csv`)
- **Electricity (ECL)**: [archive.ics.uci.edu — ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) → preprocessed `electricity.csv` from Autoformer repo
- **Traffic**: [PeMS D7 via LSTF-Linear repo](https://github.com/cure-lab/LTSF-Linear) (`traffic.csv`)
- **ILI**: [CDC FluView (CDC.gov)](https://www.cdc.gov/flu/weekly/fluviewinteractive.htm) → preprocessed `national_illness.csv` from Autoformer repo
- **Solar-Energy**: [LSTNet repo](https://github.com/laiguokun/multivariate-time-series-data) (`solar_AL.txt`)

### Priority for E4 Benchmark Grid

Recommended run order (difficulty / time cost ascending):

1. ETTh1, ETTh2 — fast (7 features, small), already have ETTh1 in `data/ETT/`
2. ETTm1, ETTm2 — 4× longer than ETTh but same structure
3. Exchange-Rate — daily, small, good sanity check for long horizons
4. ILI — short but tricky; tests short-series regime
5. Weather — medium size, 21 features
6. Solar-Energy — 137 features, requires LSTNet preprocessing script
7. Electricity (ECL) — 321 features, GPU memory pressure at large batch
8. Traffic — 862 features, most expensive; consider `hidden_dim=32` first

---

## Hyperparameter Reference

| Parameter | Search Range | Default |
|-----------|-------------|---------|
| `hidden_dim` | 32, 64, 128, 256 | 64 |
| `seq_len` | 96, 192, 336 | 96 |
| `pred_len` | 24, 96, 192, 336, 720 | 96 |
| `label_len` | 0, seq_len//2 | 48 |
| `batch_size` | 16, 32, 64 | 32 |
| `learning_rate` | 1e-4, 5e-4, 1e-3 | 1e-3 |
| `disentangle_alpha` | 0, 0.01, 0.1, 1.0 | 0.1 |
| `disentangle_beta` | 0, 0.01, 0.1, 1.0 | 0.1 |
| `conformal_alpha` | 0.05, 0.10 | 0.10 |
| `conformal_strategy` | max, scalar, per_output | max |
| `num_epochs` | — | 10 (early-stop) |
| `patience` | — | 5 |

---

## File Checklist

```
manuscript/
├── README.md           ← this file (outline + TODO)
├── paper.tex           ← main LaTeX source (to be created)
├── figures/            ← generated plots (PNG/PDF)
│   ├── state_pca.pdf
│   ├── contribution_heatmap.pdf
│   ├── calibration_curve.pdf
│   └── runtime_scaling.pdf
├── tables/             ← auto-generated LaTeX table fragments
│   ├── main_results.tex
│   ├── ablation_disentangle.tex
│   └── ablation_conformal.tex
└── supplement/
    └── proofs.tex
```
