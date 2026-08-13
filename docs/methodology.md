# Scientific scope and reporting

The interval procedure is split conformal under its stated exchangeability assumptions. Overlapping time-series windows are dependent, so unconditional finite-sample coverage is not claimed for the deployed time-series protocol. The runner records within-cluster lag-1 residual autocorrelation in `dependence_diagnostics.json`; it does not apply an unsupported ACF inflation factor.

The locked comparison uses the same chronological calibration split, alpha, score geometry, preprocessing, validation budget, and reporting metrics for CISSN and baselines. Main tables report MSE, MAE, primary coverage, mean interval width, Winkler score, calibration error, and MSIS. Every result includes predictions, targets, bounds where available, training history, sanity report, runtime, configuration, environment, and protocol manifest.

Treat all findings as empirical until the predeclared multi-seed, multi-dataset analyses complete. Report mean and standard deviation across independent outer seeds, paired uncertainty comparisons, and failures as well as successes.

## Forecast collapse under MSE

Every run records `var(pred)/var(true)` on the validation split per epoch (`vali_variance_ratio` in `history.json`) and on the test split (`variance_ratio` in `metrics.json`). This separates two failure modes that point metrics alone cannot distinguish:

- An **information bottleneck** caps the forecast's *rank*: the forecast stays appropriately dispersed but cannot express enough independent directions.
- **Mean-shrinkage under MSE** caps the forecast's *amplitude*: with a weakly predictable target, attenuating toward the training mean lowers MSE, so the model converges to a near-constant while its loss curve still improves.

The two call for different fixes, so they must not be conflated. A variance ratio near 1 is well-dispersed; below 0.5 is under-dispersed and raises an advisory quality flag; below 0.1 is effectively constant.

Measured on ETTh1, horizon 336, seeds `42,123,456`, identical configuration apart from `--revin`:

| | no RevIN | RevIN |
| --- | --- | --- |
| test MSE | 1.280 ± 0.081 | 0.771 ± 0.085 |
| variance ratio | 0.073 ± 0.009 | 0.540 ± 0.095 |
| corr(pred, true) | 0.295 ± 0.072 | 0.641 ± 0.054 |
| coverage @90% | 0.788 ± 0.011 | 0.908 ± 0.003 |

(mean ± sample standard deviation over the three seeds.)

All three seeds improve on every column. The latent state dimension is unchanged across both arms, so the recovered accuracy is not explained by added forecast rank; the failure was amplitude collapse, not lost capacity. Interval coverage moved from `0.788` to nominal with *narrower* intervals, which indicates the earlier under-coverage came from calibrating a collapsed forecaster rather than from a defect in the conformal procedure.

These are development observations on one dataset and horizon; they justify the diagnostic, not a publication claim.

## Paired flat-CP comparison

The contribution under test is *state conditioning*, not conformal prediction itself. Every benchmark run therefore calibrates `FlatConformal` on the **same calibration residuals from the same trained model** and scores it on the **same test forecasts**, writing the result to `interval_flat_cp` in `metrics.json`. Fitting the flat comparator in a separate training run would confound the partition's effect with training variance, so the two calibrators must share a model.

Winkler score is the primary comparison because it penalises width and coverage jointly: a method can trivially raise coverage by widening intervals, and only a scoring rule that charges for width detects that.

ETTh1, horizon 336, `--revin`, seeds `42,123,456`:

| seed | SCCP Winkler | flat CP Winkler | delta | SCCP width | flat width |
| --- | --- | --- | --- | --- | --- |
| 42 | 3.5715 | 3.7452 | -0.174 | 2.5618 | 2.6431 |
| 123 | 3.5739 | 3.7710 | -0.197 | 2.5365 | 2.6085 |
| 456 | 3.8902 | 3.9869 | -0.097 | 2.7989 | 2.7870 |

Mean delta `-0.156 ± 0.053`, a `4.1%` relative improvement, negative on 3/3 seeds, achieved with narrower intervals at comparable coverage. The direction also holds without RevIN (delta `-0.147` at seed 42), so the advantage is not an artefact of the collapse fix.

With `n = 3` this is descriptive evidence of a consistent sign, not a confirmatory test; the predeclared multi-seed, multi-dataset analysis remains the basis for any published claim.
