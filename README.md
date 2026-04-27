# CISSN — Conformally Calibrated Interpretable State-Space Networks

**Hybrid deep learning for time-series forecasting with structured interpretability and distribution-free uncertainty quantification.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Overview

CISSN unifies three capabilities that are usually in tension:

| Capability | CISSN Approach |
|------------|---------------|
| **Accuracy** | Neural state-space encoder + hybrid linear/nonlinear forecast head |
| **Interpretability** | 5-dimensional disentangled latent state (Level, Trend, Seasonal pair, Residual) with per-component contribution analysis |
| **Uncertainty** | State-Conditional Conformal Prediction (SCCP) — distribution-free finite-sample coverage guarantees |

The encoder uses a structured block-diagonal transition matrix with physically meaningful constraints (near-unit root for level, 2D rotation for seasonality, fast decay for residuals). A small neural correction refines the linear dynamics. The forecast head combines an interpretable linear projection with a learnable non-linear refinement path. SCCP clusters latent states during calibration to produce adaptive prediction intervals — narrow in stable regimes, wide in volatile ones.

---

## Installation

```bash
git clone https://github.com/Narden91/CISSN.git
cd CISSN
uv venv
uv pip install -e .
```

**Requirements**: Python ≥ 3.9, PyTorch ≥ 2.0.0. Uses `uv` for dependency management.

---

## Quick Start

```python
import torch
from cissn.models import DisentangledStateEncoder, ForecastHead
from cissn.conformal import StateConditionalConformal
from cissn.explanations import ForecastExplainer

# Initialize components
encoder = DisentangledStateEncoder(input_dim=7, state_dim=5)
head = ForecastHead(state_dim=5, output_dim=1, horizon=24)
conformal = StateConditionalConformal(alpha=0.1, n_clusters=5)
explainer = ForecastExplainer(head)

# Forward pass
x = torch.randn(1, 96, 7)    # (batch, seq_len, features)
state = encoder(x)             # (1, 5) — Level, Trend, Seasonal_0, Seasonal_1, Residual
forecast = head(state)         # (1, 24, 1)

# Calibrate uncertainty (on validation data)
conformal.fit(cal_states, cal_residuals)
lower, upper = conformal.predict(state, forecast)

# Explain the forecast
explanations = explainer.explain(state)
print(f"Level: {explanations[0].level_contribution:.3f}, "
      f"Trend: {explanations[0].trend_contribution:.3f}, "
      f"Seasonal: {explanations[0].seasonal_contribution:.3f}")
```

Run the full demo:
```bash
uv run examples/demo_cissn.py
```

Run a benchmark experiment:
```bash
uv run experiments/run_benchmark.py --data ETTh1 --train_epochs 10 --seed 42
```

---

## Architecture

```
Input Sequence (B, T, D)
        │
        ▼
┌──────────────────────────────────┐
│  Disentangled State Encoder      │
│  s_t = A·s_{t-1} + B(x_t)       │  ← Structured physics
│        + correction_scale·MLP(…) │  ← Small neural refinement
│  State: [Level, Trend,           │
│          Seasonal cos, sin,      │
│          Residual]               │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Forecast Head                   │
│  ŷ = W·s + b                    │  ← Interpretable linear
│    + refine_scale·MLP(s)        │  ← Learnable non-linear
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  SCCP (Conformal Prediction)     │
│  1. KMeans(state) → cluster     │
│  2. Lookup quantile qₖ          │
│  3. [ŷ − qₖ, ŷ + qₖ]          │
└──────────────────────────────────┘
```

The 5-dimensional state:

| Dim | Component | Dynamics | Constraint |
|-----|-----------|----------|------------|
| 0 | Level | Near-unit root | α ∈ [0.85, 1.0] |
| 1 | Trend | Persistent | α ∈ [0.70, 0.95] |
| 2 | Seasonal cos | 2D rotation | γ ∈ [0.80, 1.0] |
| 3 | Seasonal sin | | ω learnable |
| 4 | Residual | Fast decay | α ∈ [0.0, 0.4] |

---

## Supported Datasets

| Dataset | Domain | Frequency | Features |
|---------|--------|-----------|----------|
| ETTh1, ETTh2 | Transformer temperature | Hourly | 7 |
| ETTm1, ETTm2 | Transformer temperature | 15-min | 7 |
| Weather | Meteorology | 10-min | 21 |
| Exchange-Rate | Finance | Daily | 8 |
| ECL | Electricity | Hourly | 321 |
| Traffic | Transport | Hourly | 862 |
| ILI | Epidemiology | Weekly | 7 |
| Solar-Energy | Renewable energy | 10-min | 137 |

Download all datasets:
```bash
uv run scripts/download_datasets.py
```

---

## Baselines

All six comparison models are implemented in `cissn/baselines/` and share the same `forward(x)` interface:

| Baseline | Reference | Key idea |
|----------|-----------|----------|
| `DLinear` | Zeng et al., AAAI 2023 | Moving-average decomposition + two linear layers |
| `FlatConformal` | — | Marginal conformal prediction (no state-conditioning) |
| `MCDropout` | Gal & Ghahramani, ICML 2016 | Stochastic dropout at inference for uncertainty |
| `DeepEnsemble` | Lakshminarayanan et al., NeurIPS 2017 | Ensemble of independently trained models |
| `PatchTST` | Nie et al., ICLR 2023 | Channel-independent patch-based Transformer |
| `DeepState` | Rangapuram et al., NeurIPS 2018 | GRU encoder + structured SSM + Gaussian intervals |

## Evaluation Metrics

**Point forecast**: MSE, MAE, RMSE, MAPE

**Uncertainty**: Empirical coverage @ 90%, mean prediction interval width

**Disentanglement quality** (per epoch): Mean off-diagonal correlation (should approach 0), per-dimension variance

**Interpretability**: Refinement ratio (fraction of prediction from non-linear path; < 0.5 means linear dominates)

---

## Repository Structure

```
cissn/              # Package
├── models/         # DisentangledStateEncoder, ForecastHead
├── conformal/      # StateConditionalConformal (SCCP)
├── losses/         # DisentanglementLoss
├── explanations/   # ForecastExplainer
├── baselines/      # DLinear, FlatConformal, MCDropout, DeepEnsemble, PatchTST, DeepState
└── data/           # BaseETTDataset, data loader factory

experiments/        # Benchmark runner + experiment plan
examples/           # Demo script
tests/              # 13 unit tests (4 test files)
scripts/            # Dataset download utility
docs/               # Technical specification, flow diagram, datasets info
architecture/       # Architecture documentation, publication strategy
manuscript/         # Paper outline (IMRaD) + experiment checklist
```

---

## Testing

```bash
uv run tests/run_tests.py     # 13 tests, all passing
```

Tests cover: encoder/head shapes and integration, ForecastExplainer structure, dataset inheritance, DataLoader evaluation policies, split validation, partial batches, conformal predictor scalar/per-feature broadcast, and incompatible shape rejection.

---

## Documentation

| Document | Contents |
|----------|----------|
| `document.md` | Intuition and how CISSN works |
| `CLAUDE.md` | Complete repository reference for AI assistants |
| `architecture/architecture_and_flow.md` | Detailed architecture with formulas |
| `architecture/publication_strategy.md` | Target venues, baselines, ablation plan |
| `docs/cissn_technical_specification.md` | Full theoretical specification + pseudocode |
| `docs/flow_diagram.md` | Mermaid architecture diagram |
| `manuscript/README.md` | Paper outline + experiment TODO checklist |

---

## Citation

```bibtex
@software{cissn2025,
  author = {Emanuele Nardone},
  title = {CISSN: Conformally Calibrated Interpretable State-Space Networks},
  year = {2025},
  url = {https://github.com/Narden91/CISSN}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
