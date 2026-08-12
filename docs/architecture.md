# CISSN architecture

CISSN maps an input window `(batch, sequence, feature)` to a five-dimensional latent state with `DisentangledStateEncoder`. `ForecastHead` maps the final state to the forecast horizon. Training optimizes forecast MSE plus the configured covariance and temporal disentanglement penalties.

For intervals, the trained model is frozen. The runner learns a K-Means state partition from train-split states only, then calibrates state-conditional residual quantiles on the later chronological calibration split. It never refits the partition during calibration or test evaluation. Sparse calibration clusters use the global calibration quantile and are recorded in `cluster_stats.json`.

`per_feature` is the publication default: each horizon-feature cell has its own absolute-residual score and marginal coverage is primary. `max` uses a maximum score over the full horizon-feature block and its primary metric is simultaneous coverage.

The flat conformal comparator uses the same score geometry without state clustering. MC-Dropout and Deep Ensemble retain their raw predictive scale but are conformalized on the same calibration split by default; raw intervals are explicitly labelled secondary.

The implementation is in `cissn/models/`, `cissn/conformal/state_conditional.py`, and `cissn/baselines/`. Launch commands are in `RUNBOOK.md`.
