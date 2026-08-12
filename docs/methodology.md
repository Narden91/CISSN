# Scientific scope and reporting

The interval procedure is split conformal under its stated exchangeability assumptions. Overlapping time-series windows are dependent, so unconditional finite-sample coverage is not claimed for the deployed time-series protocol. The runner records within-cluster lag-1 residual autocorrelation in `dependence_diagnostics.json`; it does not apply an unsupported ACF inflation factor.

The locked comparison uses the same chronological calibration split, alpha, score geometry, preprocessing, validation budget, and reporting metrics for CISSN and baselines. Main tables report MSE, MAE, primary coverage, mean interval width, Winkler score, calibration error, and MSIS. Every result includes predictions, targets, bounds where available, training history, sanity report, runtime, configuration, environment, and protocol manifest.

Treat all findings as empirical until the predeclared multi-seed, multi-dataset analyses complete. Report mean and standard deviation across independent outer seeds, paired uncertainty comparisons, and failures as well as successes.
