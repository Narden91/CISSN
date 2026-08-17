# CISSN architecture

> Evidence status: saved ETTh1 test artifacts are exploratory development evidence. The headroom script reuses those arrays, so its label-informed reference is neither an oracle nor an upper bound and cannot choose a default.

CISSN maps an input window `(batch, sequence, feature)` to a five-dimensional latent state with `DisentangledStateEncoder`. `ForecastHead` maps the final state to the forecast horizon. Training optimizes forecast MSE plus the configured covariance and temporal disentanglement penalties.

For intervals, the trained model is frozen. `--conformal_conditioning` selects the primary state conditioning mechanism, calibrated on the later chronological calibration split; it never refits during calibration or test evaluation. Both mechanisms' conditioning-fit windows are now the same: `_build_conditioning_predictors`/`_calibrate_conformal` (`run_benchmark.py:690-753`) fit `cluster`'s partition and `scale`'s sigma regression on the same `conditioning_states` — the first half of the calibration split — so neither has an in-sample or sample-size advantage. (An earlier version fit the cluster partition on train-split states instead, a ~9x asymmetry against `scale`'s calibration-half fit; see `docs/methodology.md` for that history and why the previously measured ordering between the two mechanisms needs re-running under the fixed code before it can be trusted.)

- `scale` (`StateScaledConformal`): a log-linear regression of residual scale on the (standardised) state gives a continuous `sigma(state)`, and quantiles are calibrated on residuals normalized by that scale. `--scale_geometry` sets sigma's shape: `scalar` (default) fits one scale per sample; `per_cell` fits one per horizon-feature cell against a shared Gram factorisation. The geometry matters more than the mechanism — the conditioning signal is a state x cell interaction, so the scalar geometry cannot express it and measures as no improvement over flat CP, while `per_cell` does. See `docs/methodology.md`.
- `cluster` (`StateConditionalConformal`): learns a K-Means partition of the state and calibrates one quantile per cluster, so each cluster gets its own full quantile *surface* — which is why it captures the same state x cell interaction. Sparse calibration clusters use the global calibration quantile and are recorded in `cluster_stats.json`. This is the default primary mode; its previously measured edge over per-cell state-scaled CP on ETTh1-h336 predates the fitting-set-asymmetry fix above and has not yet been re-measured under equal fitting sets — see `docs/methodology.md`.

Both are reported with their measured numbers on every run. State conditioning is regime-dependent and must be established in sealed confirmation, not inferred from historical ETTh1 diagnostics. `scripts/diagnose_conditioning_headroom.py` is explicitly exploratory: it reloads evaluation artifacts, compares shared-window calibrators, and adds a label-informed scalar reference plus residual-variance decomposition. It cannot select defaults or support confirmatory claims.

Every run calibrates and reports both, plus flat CP, regardless of which is primary (`interval_flat_cp`, `interval_cluster_cp`, `interval_state_scaled` in `metrics.json`), so the comparison is always paired against the same trained model, calibration residuals, and test forecasts.

`per_feature` is the publication default: each horizon-feature cell has its own absolute-residual score and marginal coverage is primary. `max` uses a maximum score over the full horizon-feature block and its primary metric is simultaneous coverage.

The flat conformal comparator uses the same score geometry without state conditioning. MC-Dropout and Deep Ensemble retain their raw predictive scale but are conformalized on the same calibration split by default; raw intervals are explicitly labelled secondary.

MC-Dropout and Deep Ensemble are UQ-method ablations on the CISSN backbone, not
independent baseline architectures: `run_baseline.py:build_backbone` constructs both from
`DisentangledStateEncoder` + `ForecastHead` and hard-requires `state_dim=5`, so their
point forecasts are CISSN's by construction. This is a legitimate, arguably
better-controlled design for the UQ comparison, but a table listing "CISSN vs MC-Dropout
vs Deep Ensemble vs DLinear vs PatchTST" implies five independent architectures where
three exist. Deep Ensemble also receives roughly 3x the training budget of a single
member (one full training run per ensemble member); report that alongside any comparison.

Conditional-coverage claims are checked against `worst_slab_coverage` and `max_coverage_deviation` (`cissn/evaluation/metrics.py`), computed on prespecified, method-agnostic state-space bins — not on each method's own partition, which would make the comparison unfair. `run_benchmark.py` fits these bin edges (`fit_coverage_bin_edges`) on the same calibration-half `conditioning_states` used above, not on train states; `tests/test_experiment_runners.py::TestConditioningCalibrationDataSource` pins this call site.

The implementation is in `cissn/models/`, `cissn/conformal/state_conditional.py`, and `cissn/baselines/`. Launch commands are in `RUNBOOK.md`.
