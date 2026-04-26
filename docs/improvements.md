# Future Improvements for CISSN

Status: several items from the original BSSNN roadmap are now implemented in CISSN.

## Completed (in CISSN)

- **State-space transitions**: Implemented as structured block-diagonal dynamics with 5-dimensional latent state (Level, Trend, Seasonal cos/sin pair, Residual).
- **Uncertainty quantification**: State-Conditional Conformal Prediction (SCCP) provides distribution-free finite-sample coverage guarantees.
- **Interpretability**: ForecastExplainer provides per-component linear contribution decomposition. DisentanglementLoss enforces independence via covariance regularization and temporal consistency.
- **Benchmarking**: Standard LTSF datasets (ETT, Weather, Exchange, ECL, Traffic, ILI, Solar) are supported via `cissn.data`.

## In Progress

- **Reproducibility**: Seeds now set globally in `run_benchmark.py`; wandb integration for experiment tracking.
- **Baseline comparisons**: Data loader and experiment runner ready; baseline wrappers pending (PatchTST, DLinear, DeepState).

## Planned

- **Rolling window evaluation**: Implement walk-forward validation for true out-of-sample assessment.
- **Hyperparameter optimization**: Integrate Optuna or similar for automatic tuning.
- **Causal reasoning**: Extend the structural state-space to support counterfactual queries.
- **Online/streaming conformal**: Adapt SCCP for streaming data with periodic recalibration.
- **Multivariate output strategies**: Advanced multivariate conformal (copula-based, Mahalanobis).
- **Deployment guides**: TorchServe / ONNX export pipelines for production serving.
