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

## Cluster discretisation discards most of the state's conditioning signal

The K-Means partition in `StateConditionalConformal` was the original mechanism for
conditioning intervals on the latent state. Measured on a saved ETTh1-h336 run
(`results/validation/CISSN_ETTh1_..._seed42`, pre-RevIN, `n_clusters=5`):

| quantity | value |
| --- | --- |
| R^2 of residual scale on the continuous 5-d state (linear regression) | 0.733 |
| R^2 of residual scale on K-Means cluster membership (one-way ANOVA) | 0.173 |
| test-split cluster occupancy (5 requested clusters) | `[94, 482, 1958, 0, 11]` |

77% of test windows fall in a single cluster and one cluster receives zero test samples,
so at test time the discrete predictor is close to flat CP with extra machinery. The
continuous state carries roughly 4x the linearly-recoverable information about residual
scale that the discretisation preserves.

This motivated `StateScaledConformal`: a log-linear regression of residual scale on the
(standardised) state gives `sigma(s)`, and conformal quantiles are calibrated on
`residual / sigma(s)` rather than per cluster. Reconstructed on the same saved test
residuals (not the real train/cal split — a diagnostic, not a protocol run), with a
proper fit/calibrate separation at four chronological cut points:

| cut | flat Winkler | cluster SCCP | state-scaled |
| --- | --- | --- | --- |
| 0.3 | 5.389 | 5.527 (loses to flat) | 5.213 |
| 0.4 | 5.284 | 5.425 (loses to flat) | 5.071 |
| 0.5 | 5.112 | 5.037 | 4.822 |
| 0.6 | 5.084 | 5.019 | 4.789 |

State-scaled beats flat CP at all four cuts (3-6% Winkler improvement); cluster SCCP
loses to flat CP at two of the four. An ablation on the same diagnostic found the
seasonal-rotation state coordinate alone (the coordinate most correlated with residual
scale, r=0.73) recovers nearly all of the state-scaled predictor's gain, so the
conditioning signal is concentrated in one or two coordinates rather than spread evenly
across the five-dimensional state.

This is why the paper's primary conditioning mechanism is `StateScaledConformal`
(`--conformal_conditioning scale`), with `StateConditionalConformal` (`cluster`) reported
as the ablation demonstrating what the discretisation loses. Every benchmark run
calibrates and reports both plus flat CP, regardless of which is primary
(`interval_flat_cp`, `interval_cluster_cp`, `interval_state_scaled` in `metrics.json`),
so this is paired evidence, not two separately-run comparisons. This diagnostic must be
reproduced under the real protocol (RUNBOOK.md Step 3b) before it supports a publication
claim; treat the numbers above as the reason the mechanism was built, not as the
evidence for it.

## Why CISSN trails DLinear on point accuracy

The gap is a rank constraint, not a tuning failure, and it is intrinsic to the architecture.

On ETTh1-h336 the target block spans `336 x 7 = 2352` cells. Measured on the test split, DLinear's own forecasts need **rank 96** to reach 99% of their energy and already hold 86.8% by rank 5. CISSN maps every window through a five-dimensional state, so its forecast can never exceed **rank 5** regardless of head capacity.

Three interventions were tested and none closed the gap, which is what a hard rank constraint predicts:

| variant | test MSE |
| --- | --- |
| CISSN + RevIN | 0.723 |
| \+ `--lambda_refinement 0.1` | 0.759 |
| \+ `--dropout 0.3` | 0.730 |
| \+ `--no_refinement` (linear head only) | 0.818 |
| DLinear reference | 0.619 |

Removing the refinement MLP makes accuracy *worse*, so the MLP is compensating for the bottleneck rather than merely memorising. Note that an oracle rank-5 linear projection of the target reaches MSE `0.422`; CISSN does not approach that because it must *learn* its basis through a recurrent encoder rather than being handed the optimal one.

The consequence for reporting: CISSN's five-state constraint should be presented as an interpretability/calibration design choice with a measured accuracy cost, not as a competitive point forecaster. The hybrid architecture exists precisely to keep full-rank accuracy in the base while the state supplies a structured correction.

## Validation-to-test transfer on ETTh1

Checkpoint selection on validation does not reliably transfer to test on this dataset. Standardised by train statistics, split means are `train 0.000`, `val -0.108`, `test -0.049`, with test standard deviation `1.123` against train `1.000`: validation and test differ from train *and from each other*.

Observed directly in the hybrid run at horizon 336: correction-stage validation loss improved from `1.160` to `1.112` (-4.1%) and the epoch-0 fallback correctly did not trigger, yet test MSE was `0.799` against the frozen DLinear base's `0.619`. The fallback machinery behaved exactly as specified; the validation signal itself was misleading.

Treat single-seed, single-horizon validation improvements on ETTh1 as weak evidence, and require the multi-seed multi-dataset protocol before promoting any architecture on validation deltas alone.

## Simultaneous coverage

`per_feature` geometry reports `coverage_joint = 0.0000` at horizon 336: it calibrates each of the 2352 horizon-feature cells to its own marginal quantile, so simultaneous coverage of every cell is not what the method targets and near-zero is the expected reading, not a defect.

`--multivariate_strategy max` is the geometry that targets simultaneous coverage. Measured on ETTh1-h336 with `--revin`, seed 42:

| geometry | marginal | joint | mean width | Winkler |
| --- | --- | --- | --- | --- |
| `per_feature` | 0.9071 | 0.0000 | 2.56 | 3.57 |
| `max` | 0.9953 | 0.4923 | 8.10 | 8.16 |

`max` produces genuine simultaneous coverage but reaches only `0.49` against a nominal `0.90`, at roughly `3.2x` the interval width. The shortfall is not a finite-sample artefact — the calibration split holds 1393 origins and the exact-rank index `k = 1255` is attainable. It reflects test-time dependence across cells: under independence, per-cell coverage of `0.9953` would give a joint rate near `1.5e-05`, so the observed `0.49` indicates strong positive dependence, and residual distribution shift between calibration and test then costs the remainder.

Do not report `max` as a validated simultaneous-coverage guarantee. Either report it as an honest negative result with the numbers above, or restrict simultaneous claims to a smaller, prespecified cell block where the target is attainable.
