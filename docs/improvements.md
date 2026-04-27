# Future Improvements for CISSN

## Completed (in CISSN)

- **State-space transitions**: Implemented as structured block-diagonal dynamics with 5-dimensional latent state (Level, Trend, Seasonal cos/sin pair, Residual).
- **Uncertainty quantification**: State-Conditional Conformal Prediction (SCCP) provides distribution-free finite-sample coverage guarantees.
- **Interpretability**: ForecastExplainer provides per-component linear contribution decomposition. DisentanglementLoss enforces independence via covariance regularization and temporal consistency.
- **Benchmarking**: Standard LTSF datasets (ETT, Weather, Exchange, ECL, Traffic, ILI, Solar) are supported via `cissn.data`.
- **Reproducibility**: Seeds set globally in `run_benchmark.py`; wandb integration for experiment tracking.
- **Baseline comparisons**: All six baselines implemented in `cissn/baselines/`:
  - `DLinear` — decomposition-linear (Zeng et al., AAAI 2023)
  - `FlatConformal` — marginal conformal baseline (no state-conditioning)
  - `MCDropout` — Monte Carlo dropout uncertainty (Gal & Ghahramani, ICML 2016)
  - `DeepEnsemble` — ensemble uncertainty (Lakshminarayanan et al., NeurIPS 2017)
  - `PatchTST` — channel-independent patch Transformer (Nie et al., ICLR 2023)
  - `DeepState` — GRU + structured SSM with Gaussian intervals (Rangapuram et al., NeurIPS 2018)

## Planned

- **Rolling window evaluation**: Implement walk-forward validation for true out-of-sample assessment.
- **Hyperparameter optimization**: Integrate Optuna or similar for automatic tuning.
- **Causal reasoning**: Extend the structural state-space to support counterfactual queries.
- **Online/streaming conformal**: Adapt SCCP for streaming data with periodic recalibration.
- **Multivariate output strategies**: Advanced multivariate conformal (copula-based, Mahalanobis).
- **Deployment guides**: TorchServe / ONNX export pipelines for production serving.
