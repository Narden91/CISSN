# CISSN architecture

CISSN maps an input window `(batch, sequence, feature)` to a five-dimensional latent state with `DisentangledStateEncoder`. `ForecastHead` maps the final state to the forecast horizon. Training optimizes forecast MSE plus the configured covariance and temporal disentanglement penalties.

For intervals, the trained model is frozen. `--conformal_conditioning` selects the primary state conditioning mechanism, fit on train-split states only and calibrated on the later chronological calibration split; it never refits during calibration or test evaluation:

- `scale` (`StateScaledConformal`): a log-linear regression of residual scale on the (standardised) state gives a continuous `sigma(state)`, and quantiles are calibrated on residuals normalized by that scale. This is the paper's primary contribution.
- `cluster` (`StateConditionalConformal`): learns a K-Means partition of the state and calibrates one quantile per cluster. Sparse calibration clusters use the global calibration quantile and are recorded in `cluster_stats.json`. Reported as the ablation demonstrating what discretising the state loses relative to `scale` (see `docs/methodology.md`).

Every run calibrates and reports both, plus flat CP, regardless of which is primary (`interval_flat_cp`, `interval_cluster_cp`, `interval_state_scaled` in `metrics.json`), so the comparison is always paired against the same trained model, calibration residuals, and test forecasts.

`per_feature` is the publication default: each horizon-feature cell has its own absolute-residual score and marginal coverage is primary. `max` uses a maximum score over the full horizon-feature block and its primary metric is simultaneous coverage.

The flat conformal comparator uses the same score geometry without state conditioning. MC-Dropout and Deep Ensemble retain their raw predictive scale but are conformalized on the same calibration split by default; raw intervals are explicitly labelled secondary.

Conditional-coverage claims are checked against `worst_slab_coverage` and `max_coverage_deviation` (`cissn/evaluation/metrics.py`), computed on prespecified, method-agnostic state-space bins fit on train data — not on each method's own partition, which would make the comparison unfair.

The implementation is in `cissn/models/`, `cissn/conformal/state_conditional.py`, and `cissn/baselines/`. Launch commands are in `RUNBOOK.md`.
