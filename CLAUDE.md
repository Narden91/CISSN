# CLAUDE.md — CISSN Repository Guide

## Project Identity

**CISSN** (Conformally Calibrated Interpretable State-Space Networks) is a hybrid deep-learning framework for time-series forecasting that combines structured state-space dynamics with conformal prediction for distribution-free uncertainty quantification.

- **Package**: `cissn`
- **Version**: 0.1.0
- **Python**: >= 3.9
- **Framework**: PyTorch >= 2.0.0
- **Package manager**: uv (with pyproject.toml — `requirements.txt` is not used)
- **License**: MIT

## Quick Commands

```bash
uv run examples/demo_cissn.py                                                   # End-to-end demo
uv run python tests/run_tests.py                                                 # Run all 25 tests
uv run python benchmark_encoder.py                                               # Encoder throughput benchmark
uv run python scripts/download_datasets.py                                       # Download ETT/Weather/ILI datasets
uv run python experiments/run_benchmark.py --data ETTh1 --train_epochs 10       # Train + evaluate CISSN
uv run python experiments/run_baseline.py --model dlinear --data ETTh1 --pred_len 24 --train_epochs 1  # Baseline smoke / grid cell
uv run python experiments/run_multiseed.py --data ETTh1 --all_horizons --seeds 42,123,456              # Publication grid driver
uv run python experiments/run_benchmark.py --data ETTh1 --lradj cosine          # Train with cosine LR schedule
uv run python experiments/run_benchmark.py --data ETTh1 --walk_forward          # Walk-forward rolling evaluation
```

## Architecture

### Five-Dimensional Disentangled State

The encoder maps input sequences to a **5-dimensional** latent state with explicit physical meaning. `STRUCTURED_STATE_DIM = 5` is defined once in `cissn/constants.py` and imported everywhere — it is a hard architectural constraint.

| Index | Component | Symbol | Dynamics | Parameter Range |
|-------|-----------|--------|----------|-----------------|
| 0 | Level | s⁰ | Near-unit root (slow varying) | α_L ∈ [0.85, 1.00] |
| 1 | Trend | s¹ | Persistent, smooth | α_T ∈ [0.70, 0.95] |
| 2 | Seasonal cos | s² | 2D rotation (cos component) | γ ∈ [0.80, 1.00] |
| 3 | Seasonal sin | s³ | 2D rotation (sin component) | ω (learnable freq) |
| 4 | Residual | s⁴ | Fast-decaying noise | α_R ∈ [0.00, 0.40] |

The constrained decay parameters (sigmoid gates) are implemented once in `StructuredDecayMixin` (`cissn/models/_dynamics.py`) and inherited by both `DisentangledStateEncoder` and `DeepState`.

### State Transition (per timestep)

```
s_t = A · s_{t-1}  +  B(x_t)  +  β · tanh(MLP(A·s_{t-1} + B(x_t), h_t))
      ⎣ structured ⎦  ⎣innov.⎦    ⎣        small tanh-MLP correction       ⎦
```

The matrix A is block-diagonal with constrained eigenvalues. The correction MLP is spectrally normalised (`nn.utils.spectral_norm`) bounding each linear layer to `‖W‖₂ ≤ 1`, but GELU has Lipschitz constant `L_G ≈ 1.77`, so `‖J_MLP‖₂ ≤ L_G`. Combined with `‖A‖₂ ≤ 1`, the per-step Jacobian is bounded by `1 + L_G·β` (`L_G ≈ 1.77`, the GELU Lipschitz constant; β is initialised at 0.01 via softplus).

### Core Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `DisentangledStateEncoder` | `cissn/models/encoder.py` | Maps sequences → disentangled 5-d state via structured SSM |
| `ForecastHead` | `cissn/models/forecast_head.py` | Linear projection + MLP refinement → multi-horizon forecasts |
| `StateConditionalConformal` | `cissn/conformal/state_conditional.py` | K-Means state clustering → per-cluster quantile → adaptive intervals |
| `DisentanglementLoss` | `cissn/losses/disentangle_loss.py` | Covariance regularisation + temporal consistency loss |
| `ForecastExplainer` | `cissn/explanations/state_attribution.py` | Per-component contribution decomposition |
| `BaseETTDataset` | `cissn/data/dataset.py` | ETT benchmark data loading with date-aware splits |
| `get_data_loader` | `cissn/data/data_loader.py` | DataLoader factory for 10 standard benchmarks |
| `EarlyStopping` | `cissn/utils/early_stopping.py` | Validation-loss-based early stopping with checkpoint saving |

### Data Flow

```
Input X (B, T, D_in)
  → InputProj (Linear + LayerNorm + GELU)
  → Recurrent steps: innovation + structured A transition + tanh correction
  → State s_T (B, 5)
  → ForecastHead: linear(W·s + b) + refine_scale * MLP(s)
  → Output Ŷ (B, H, D_out)
  → SCCP: cluster(s_T) → quantile → [ŷ − q, ŷ + q]
```

### Conformal Prediction (SCCP)

1. **Calibration**: Encode validation set → cluster states via K-Means → per-cluster quantile of absolute residuals
2. **Finite-sample correction**: q_k = Quantile(R_k, ⌈(n_k+1)(1−α)⌉ / (n_k+1)) with clipping at 1.0
3. **ACF-aware quantile inflation** (Theorem 1b): When within-cluster ACF(1) exceeds 0.3, quantiles are inflated by `1 + (√((1+|ρ|)/(1−|ρ|)) − 1) / √n_k` to compensate for reduced effective sample size. Enabled by default via `correct_acf=True`. The correction formula is centralised in `_compute_acf_correction(rho, n_k)`.
4. **Multivariate strategies**: `per_feature` (default), `max`, `mean`, or `mahalanobis` (Mahalanobis distance + covariance back-projection)
5. **Empty-cluster fallback**: max-of-nonempty-cluster-quantile ensures coverage ≥ 1−α
6. **Validation**: `check_exchangeability()` reports per-cluster ACF(1) values; internally calls `_prepare_residuals()` so the ACF is computed on the same reduced data that was used during `fit()`

### Loss Function

```
Total = MSE(ŷ, y) + λ_cov · CovLoss + λ_temp · TemporalLoss
```

- **CovLoss**: Frobenius norm of off-diagonal state covariance (encourages independence)
- **TemporalLoss**: Deviation from expected structured transitions (encourages proper per-dimension dynamics)

### Learning Rate Schedules (`--lradj`)

| Policy | Behaviour |
|--------|-----------|
| `type1` (default) | Halve LR every epoch: `lr × 0.5^(epoch−1)` |
| `type2` | Fixed milestone table: 2→5e-5, 4→1e-5, 6→5e-6, … |
| `cosine` | Cosine annealing to 1% of initial LR over `train_epochs` |

## Package Structure

```
cissn/
├── __init__.py              # Version (0.1.0); re-exports STRUCTURED_STATE_DIM
├── constants.py             # STRUCTURED_STATE_DIM = 5 (single source of truth)
├── models/
│   ├── __init__.py          # Exports DisentangledStateEncoder, ForecastHead
│   ├── _dynamics.py         # StructuredDecayMixin (shared sigmoid-gated decay params)
│   ├── encoder.py           # Structured SSM encoder with SN correction
│   └── forecast_head.py     # Linear + MLP forecast head with refinement_ratio metric
├── conformal/
│   ├── __init__.py          # Exports StateConditionalConformal
│   └── state_conditional.py # SCCP: K-Means clustering + ACF-aware quantile inflation
├── losses/
│   ├── __init__.py          # Exports DisentanglementLoss
│   └── disentangle_loss.py  # Covariance + temporal consistency regularisation
├── explanations/
│   ├── __init__.py          # Exports ForecastExplainer, ExplanationResult
│   └── state_attribution.py # Per-component contribution decomposition
├── utils/
│   ├── __init__.py          # Exports EarlyStopping
│   └── early_stopping.py    # Validation-loss early stopping + checkpoint saving
├── baselines/
│   ├── __init__.py          # Exports all six baselines
│   ├── dlinear.py           # Channel-independent DLinear (Zeng et al., AAAI 2023)
│   ├── flat_conformal.py    # Marginal conformal baseline (no state-conditioning)
│   ├── mc_dropout.py        # MC Dropout uncertainty (Gal & Ghahramani, ICML 2016)
│   ├── deep_ensemble.py     # Deep Ensemble uncertainty (Lakshminarayanan et al., NeurIPS 2017)
│   ├── patchtst.py          # Channel-independent PatchTST (Nie et al., ICLR 2023)
│   ├── deepstate.py         # GRU + StructuredDecayMixin SSM (Rangapuram et al., NeurIPS 2018)
│   └── training.py          # Shared baseline training/evaluation helpers
├── evaluation/
│   ├── __init__.py          # Exports all metrics + plot functions
│   ├── metrics.py           # MSE, MAE, RMSE, MAPE, PICP, MPIW, Winkler, CRPS, calibration error
│   └── plots.py             # Calibration curve, reliability diagram, decomposition, interval width
└── data/
    ├── __init__.py          # Exports datasets + get_data_loader
    ├── dataset.py           # BaseETTDataset + ETT_hour/minute/Custom; uses _read_data()
  ├── data_loader.py       # DataLoader factory for 10 benchmarks; logging via logging module
  └── registry.py          # Canonical dataset metadata and defaults
```

## Key Files (Root)

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata + all dependencies (including `scipy>=1.10.0`) |
| `document.md` | Canonical Q1 Journal 1 experiment master plan |
| `experiments/run_benchmark.py` | Main training + evaluation script; imports EarlyStopping from cissn.utils |
| `experiments/run_baseline.py` | Unified runner for implemented point and UQ baselines |
| `experiments/run_multiseed.py` | Multi-seed CISSN grid runner with JSON and raw CSV aggregation |
| `experiments/run_ablation.py` | Ablation runner for full vs. component-disabled variants |
| `examples/demo_cissn.py` | End-to-end demo of all components |
| `benchmark_encoder.py` | Encoder throughput benchmarking (eager vs. torch.compile) |
| `scripts/download_datasets.py` | Download all benchmark datasets |
| `tests/run_tests.py` | Test runner (discovers test_*.py) |
| `tests/test_model.py` | Encoder + ForecastHead shape tests |
| `tests/test_components.py` | Explainer structure + dataset inheritance tests |
| `tests/test_training.py` | DataLoader policies, split validation, partial batches |
| `tests/test_utils.py` | Conformal predictor contract tests |
| `architecture/architecture_and_flow.md` | Detailed architecture documentation |
| `architecture/publication_strategy.md` | Target venues, baselines, experimental plan |
| `docs/cissn_intuition.md` | Intuition-first explanation of the model |
| `docs/cissn_technical_specification.md` | Full theoretical spec + pseudocode |
| `docs/flow_diagram.md` | Mermaid architecture diagram |
| `manuscript/README.md` | Paper outline + writing gates |

## Conventions

- **Imports**: `from cissn.models import DisentangledStateEncoder, ForecastHead`
- **State dim**: `from cissn import STRUCTURED_STATE_DIM` (always 5 — hard architectural constraint)
- **Horizon slicing**: `outputs[:, -pred_len:, f_dim:]` where `f_dim = -1` for MS, `0` otherwise
- **Forward helper**: `Experiment._forward_and_slice(batch_x, batch_y, return_all_states=False)` — use `return_all_states=True` during training for the disentanglement loss
- **Seed**: Set via `--seed` CLI arg (default 42); all torch/numpy/random/cudnn seeds
- **Device**: `cuda` if available, else `cpu`; `non_blocking=True` for device transfers
- **torch.compile**: Applied on non-Windows systems for the encoder's inner sequence loop
- **Checkpoints**: `torch.load(..., weights_only=True)` — two files per run: `checkpoint.pth` (encoder) + `checkpoint_head.pth` (head)
- **Results**: Saved to `./results/{setting}/` as `.npy` files
- **Logging**: Library code uses `logging.getLogger(__name__)` — no `print()` calls in `cissn/`; `--use_wandb` for experiment tracking
- **Early stopping**: Imported from `cissn.utils`; tracks `val_loss_min` and saves checkpoints on improvement

## Known Limitations

- Walk-forward rolling window evaluation (`--walk_forward`) silently drops trailing samples when `len(test_data) % pred_len != 0`; a `UserWarning` is now emitted with the exact count.
- ACF-aware quantile correction addresses AR(1) autocorrelation (Theorem 1b); stronger temporal dependence (long-memory) may require block-conformal extensions.
- `torch.compile` is skipped on Windows unless `cl.exe` (MSVC) is on PATH — run `benchmark_encoder.py` to verify.

## Testing

25 tests in 5 files, all passing:

- `test_model.py` (4 tests): Encoder/head shapes, integration
- `test_components.py` (4 tests): Explainer structure, dataset inheritance, Solar loader behavior, MS target ordering
- `test_experiment_runners.py` (4 tests): Multi-seed arg propagation, baseline interval metric scopes
- `test_training.py` (6 tests): DataLoader policies, split validation, no test access during train, partial batches, variable batch concatenation, epoch diagnostics
- `test_utils.py` (7 tests): Conformal scalar/per-feature broadcast, cluster reset, constant-residual ACF, requested single cluster, incompatible shape rejection, coverage property test
