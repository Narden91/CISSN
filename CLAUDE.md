# CLAUDE.md — CISSN Repository Guide

## Project Identity

**CISSN** (Conformally Calibrated Interpretable State-Space Networks) is a hybrid deep-learning framework for time-series forecasting that combines structured state-space dynamics with conformal prediction for distribution-free uncertainty quantification.

- **Package**: `cissn` (formerly BSSNN)
- **Version**: 0.1.0
- **Python**: >= 3.9
- **Framework**: PyTorch >= 2.0.0
- **Package manager**: uv (with pyproject.toml)
- **License**: MIT

## Quick Commands

```bash
uv run examples/demo_cissn.py      # Run demo
uv run tests/run_tests.py           # Run all tests (13 tests)
uv run python benchmark_encoder.py  # Benchmark encoder
uv run python scripts/download_datasets.py  # Download benchmark data
uv run python experiments/run_benchmark.py --data ETTh1 --train_epochs 10  # Train + evaluate
```

## Architecture

### Five-Dimensional Disentangled State

The encoder maps input sequences to a **5-dimensional** latent state with explicit physical meaning:

| Index | Component | Symbol | Dynamics | Parameter Range |
|-------|-----------|--------|----------|-----------------|
| 0 | Level | s^(0) | Near-unit root (slow varying) | α_L ∈ [0.85, 1.0] |
| 1 | Trend | s^(1) | Persistent, smooth | α_T ∈ [0.70, 0.95] |
| 2 | Seasonal cos | s^(2) | 2D rotation (cos component) | γ ∈ [0.80, 1.0] |
| 3 | Seasonal sin | s^(3) | 2D rotation (sin component) | ω (learnable freq) |
| 4 | Residual | s^(4) | Fast-decaying noise | α_R ∈ [0.0, 0.4] |

### State Transition (per timestep)

```
s_t = A · s_{t-1}  +  B(x_t)  +  β · tanh(MLP(A·s_{t-1} + B(x_t), h_t))
      ⎣ structured ⎦ ⎣innov.⎦    ⎣        small tanh-MLP correction       ⎦
```

The matrix A is block-diagonal with constrained eigenvalues enforced by sigmoid gates. The correction MLP receives the linear prediction `s_linear = A·s_{t-1} + B(x_t)` and the projected input `h_t = input_proj(x_t)`. Its output is scaled by a softplus-gated learnable parameter β (init 0.01). All linear layers in the correction MLP are spectrally normalized (`nn.utils.spectral_norm`) guaranteeing `||J_MLP||₂ ≤ 1`. Combined with `||A||₂ ≤ 1`, the per-step Jacobian is bounded by `1 + β`.

### Core Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `DisentangledStateEncoder` | `cissn/models/encoder.py` | Maps sequences → disentangled state via structured SSM |
| `ForecastHead` | `cissn/models/forecast_head.py` | Linear projection + MLP refinement → multi-horizon forecasts |
| `StateConditionalConformal` | `cissn/conformal/state_conditional.py` | K-Means state clustering → per-cluster quantile → prediction intervals |
| `DisentanglementLoss` | `cissn/losses/disentangle_loss.py` | Covariance regularization + temporal consistency loss |
| `ForecastExplainer` | `cissn/explanations/state_attribution.py` | Per-component contribution decomposition |
| `BaseETTDataset` | `cissn/data/dataset.py` | ETT benchmark data loading with date-aware splits |
| `get_data_loader` | `cissn/data/data_loader.py` | DataLoader factory for 10 standard benchmarks |

### Data Flow

```
Input X (B, T, D_in)
  → InputProj (Linear + LayerNorm + GELU)
  → Recurrent steps: innovation + structured A transition + tanh correction
  → State s_T (B, 5)
  → ForecastHead: linear(W·s + b) + refine_scale * MLP(s)
  → Output Ŷ (B, H, D_out)
  → SCCP: cluster(s_T) → quantile → [ŷ - q, ŷ + q]
```

### Conformal Prediction (SCCP)

1. **Calibration**: Encode validation set → cluster states via K-Means → per-cluster quantile of absolute residuals
2. **Finite-sample correction**: q_k = Quantile(R_k, ⌈(n_k+1)(1-α)⌉ / n_k) with clipping at 1.0
3. **ACF-aware quantile inflation** (Theorem 1b): When within-cluster ACF(1) exceeds 0.3, quantiles are automatically inflated by factor `1 + (√((1+|ρ|)/(1-|ρ|)) - 1) / √n_k` to compensate for reduced effective sample size. Enabled by default via `correct_acf=True`.
4. **Empty-cluster fallback**: Max-of-nonempty-cluster-quantile ensures coverage ≥ 1-α regardless of assignment
5. **Inference**: Encode new input → assign cluster → retrieve (potentially corrected) quantile → [ŷ - q, ŷ + q]
6. **Validation**: `check_exchangeability()` reports per-cluster ACF(1) values and applied correction factors

### Loss Function

Total loss = MSE(forecast, target) + λ_cov · CovLoss + λ_temp · TemporalLoss

- **CovLoss**: Frobenius norm of off-diagonal state covariance (encourages independence)
- **TemporalLoss**: Deviation from expected structured transitions (encourages proper per-dimension dynamics)

### Monitored Metrics (per epoch)

- MSE loss (train/val/test)
- Mean absolute off-diagonal correlation (disentanglement quality)
- Per-dimension variance
- Refinement ratio (fraction of prediction from non-linear path)

### Test Evaluation

- MSE, MAE, RMSE, MAPE
- Empirical coverage @ 90% (conformal intervals)
- Mean prediction interval width

## Package Structure

```
cissn/
├── __init__.py              # Version (0.1.0)
├── models/
│   ├── __init__.py          # Exports DisentangledStateEncoder, ForecastHead
│   ├── encoder.py           # Structured SSM encoder (127 lines)
│   └── forecast_head.py     # Linear + MLP forecast head (117 lines)
├── conformal/
│   ├── __init__.py          # Exports StateConditionalConformal
│   └── state_conditional.py # SCCP with K-Means clustering (251 lines)
├── losses/
│   ├── __init__.py          # Exports DisentanglementLoss
│   └── disentangle_loss.py  # Cov + temporal regularization (157 lines)
├── explanations/
│   ├── __init__.py          # Exports ForecastExplainer, ExplanationResult
│   └── state_attribution.py # Contribution decomposition (72 lines)
└── data/
    ├── __init__.py          # Exports datasets + get_data_loader
    ├── dataset.py           # BaseETTDataset + ETT_hour/minute/Custom (282 lines)
    └── data_loader.py       # Factory for 10 benchmarks (98 lines)
```

## Key Files (Root)

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata + dependencies |
| `experiments/run_benchmark.py` | Main training + evaluation script (440 lines) |
| `examples/demo_cissn.py` | End-to-end demo of all components |
| `benchmark_encoder.py` | Encoder performance benchmarking |
| `sanity_check.py` | Quick integration check with dummy data |
| `scripts/download_datasets.py` | Download all benchmark datasets |
| `tests/run_tests.py` | Test runner (discovers test_*.py) |
| `tests/test_model.py` | Encoder + ForecastHead shape tests |
| `tests/test_components.py` | Explainer + dataset tests |
| `tests/test_training.py` | DataLoader + Experiment pipeline tests |
| `tests/test_utils.py` | Conformal predictor contract tests |
| `architecture/architecture_and_flow.md` | Detailed architecture documentation |
| `architecture/publication_strategy.md` | Target venues, baselines, experimental plan |
| `docs/cissn_technical_specification.md` | Full theoretical spec + pseudocode |
| `docs/flow_diagram.md` | Mermaid architecture diagram |
| `manuscript/README.md` | Paper outline + IMRaD structure + TODO checklist |

## Conventions

- **Imports**: `from cissn.models import DisentangledStateEncoder, ForecastHead`
- **State dim**: Always 5 (hard constraint in encoder)
- **Horizon slicing**: `outputs[:, -pred_len:, f_dim:]` where `f_dim = -1` for MS, `0` otherwise
- **Seed**: Set via `--seed` CLI arg (default 42); all torch/numpy/random/cudnn seeds
- **Device**: `cuda` if available, else `cpu`; non_blocking=True for transfers
- **Hooks**: `torch.compile` used on non-Windows systems for encoder (in `__init__`)
- **Checkpoints**: Two files — `checkpoint.pth` (encoder) + `checkpoint_head.pth` (head)
- **Results**: Saved to `./results/{setting}/` as `.npy` files
- **Logging**: Optional wandb via `--use_wandb`

## Known Limitations

- Example files `adapter.py`, `basic_classification.py`, `handwriting_classification.py` are legacy BSSNN code with `ImportError` guards
- No rolling-window evaluation yet (single train/val/test split)
- No baseline wrapper implementations (PatchTST, DLinear, DeepState)
- Conformal coverage guarantee depends on within-cluster exchangeability (tested via ACF check but not guaranteed for time series)

## Testing

13 tests in 4 files, all passing:
- `test_model.py` (4 tests): Encoder/head shapes, integration
- `test_components.py` (2 tests): Explainer structure, dataset inheritance
- `test_training.py` (4 tests): DataLoader policies, split validation, partial batches, variable batch concatenation
- `test_utils.py` (3 tests): Conformal scalar/per-feature broadcast, incompatible shape rejection
