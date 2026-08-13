# CISSN architecture

CISSN maps an input window `(batch, sequence, feature)` to a five-dimensional latent state with `DisentangledStateEncoder`. `ForecastHead` maps the final state to the forecast horizon. Training optimizes forecast MSE plus the configured covariance and temporal disentanglement penalties.

For intervals, the trained model is frozen. `--conformal_conditioning` selects the primary state conditioning mechanism, fit on train-split states only and calibrated on the later chronological calibration split; it never refits during calibration or test evaluation:

- `scale` (`StateScaledConformal`): a log-linear regression of residual scale on the (standardised) state gives a continuous `sigma(state)`, and quantiles are calibrated on residuals normalized by that scale. `--scale_geometry` sets sigma's shape: `scalar` (default) fits one scale per sample; `per_cell` fits one per horizon-feature cell against a shared Gram factorisation. The geometry matters more than the mechanism — the conditioning signal is a state x cell interaction, so the scalar geometry cannot express it and measures as no improvement over flat CP, while `per_cell` does. See `docs/methodology.md`.
- `cluster` (`StateConditionalConformal`): learns a K-Means partition of the state and calibrates one quantile per cluster, so each cluster gets its own full quantile *surface* — which is why it captures the same state x cell interaction. Sparse calibration clusters use the global calibration quantile and are recorded in `cluster_stats.json`. This is the default primary mode and currently the strongest under the real protocol.

Both are reported with their measured numbers on every run. State conditioning is regime-dependent: it improves on flat CP in the RevIN regime and does not without it. `scripts/diagnose_conditioning_headroom.py` scores flat CP, cluster SCCP, and state-scaled CP on a shared calibration window and adds an oracle upper bound plus a residual-variance decomposition; run it on a run's saved artifacts before interpreting any conditioning result.

Every run calibrates and reports both, plus flat CP, regardless of which is primary (`interval_flat_cp`, `interval_cluster_cp`, `interval_state_scaled` in `metrics.json`), so the comparison is always paired against the same trained model, calibration residuals, and test forecasts.

`per_feature` is the publication default: each horizon-feature cell has its own absolute-residual score and marginal coverage is primary. `max` uses a maximum score over the full horizon-feature block and its primary metric is simultaneous coverage.

The flat conformal comparator uses the same score geometry without state conditioning. MC-Dropout and Deep Ensemble retain their raw predictive scale but are conformalized on the same calibration split by default; raw intervals are explicitly labelled secondary.

Conditional-coverage claims are checked against `worst_slab_coverage` and `max_coverage_deviation` (`cissn/evaluation/metrics.py`), computed on prespecified, method-agnostic state-space bins fit on train data — not on each method's own partition, which would make the comparison unfair.

The implementation is in `cissn/models/`, `cissn/conformal/state_conditional.py`, and `cissn/baselines/`. Launch commands are in `RUNBOOK.md`.
