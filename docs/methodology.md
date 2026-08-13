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

**Resolved: the conflict with the headroom diagnostic was RevIN, not the split.** An
earlier revision flagged this table as irreconcilable with the headroom diagnostic below
(which found cluster SCCP losing to flat CP on 4/4 cuts of a pre-RevIN run). Re-running
`scripts/diagnose_conditioning_headroom.py` on freshly regenerated RevIN artifacts for
all three seeds settles it — with RevIN held fixed, both agree:

| geometry | mean Winkler delta vs flat CP | better on |
| --- | --- | --- |
| cluster SCCP | −0.124 | 12/12 seed-cut cells |
| state-scaled CP, scalar sigma | +0.011 | 5/12 |
| state-scaled CP, per-cell sigma | −0.237 | 12/12 |

(ETTh1-h336, seeds `42,123,456` x cuts `0.3,0.4,0.5,0.6`, every method calibrated on the
same `[cut/2:cut]` window.) The pre-RevIN run remains a genuine negative for all three
mechanisms — under amplitude collapse the residual structure that conditioning exploits
is not present. State conditioning is therefore **regime-dependent**: it helps in the
RevIN regime and does not help without it. Report it that way rather than as an
unconditional property of the method.

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
`residual / sigma(s)` rather than per cluster.

### Correction: the original diagnostic was not a paired comparison

An earlier revision of this document reported that state-scaled CP beat flat CP at all
four chronological cut points by 3-6% Winkler. **That comparison was invalid and its
conclusion does not hold.** The two columns were computed on different calibration
windows: state-scaled CP fit `sigma(s)` on `[0:cut/2]` and calibrated on `[cut/2:cut]`,
while flat CP was calibrated on the whole `[0:cut]` window. The methods were therefore
not calibrated on the same data, so the difference between the columns confounded the
conditioning mechanism with calibration sample size.

Re-run with every method calibrated on the same `[cut/2:cut]` window
(`scripts/diagnose_conditioning_headroom.py`, same saved run, `n_clusters=5`):

| cut | flat Winkler | cluster SCCP | state-scaled | oracle per-sample sigma |
| --- | --- | --- | --- | --- |
| 0.3 | 5.154 | 5.572 | 5.213 | 5.110 |
| 0.4 | 5.031 | 5.155 | 5.071 | 4.993 |
| 0.5 | 4.836 | 4.846 | 4.822 | 4.857 |
| 0.6 | 4.801 | 4.851 | 4.789 | 4.821 |

Mean Winkler delta against flat CP: cluster SCCP `+0.151` (better on 0/4 cuts),
state-scaled CP `+0.019` (better on 2/4 cuts). Under a like-for-like comparison
**neither state conditioning mechanism improves on flat CP on this run**, and
state-scaled CP shows exactly the sign-flipping pattern the earlier text attributed
only to the cluster mechanism.

### A scalar sigma is the wrong geometry: the signal is in the cell, not the sample

The `oracle per-sample sigma` column above is an upper bound on any learned *scalar*
`sigma(s)`: it sets sigma to the true per-sample test residual scale, using test labels,
which no deployable method can do. It gains at most `0.85%` Winkler and loses at two of
the four cuts. The reason is structural — decomposing test residual variance:

| axis | share of residual variance (pre-RevIN / RevIN) |
| --- | --- |
| per-sample mean (all a **scalar** `sigma(s)` can reach) | 1.19% / 0.73% |
| per-cell mean (already captured by `per_feature` quantiles) | 21.19% / 18.11% |

A per-window scalar can only exploit the first row, and under `per_feature` geometry the
second row is already handled by cell-wise quantiles — so a scalar sigma competes for
about one percent of the variation. Consistent with this, the fitted scalar `sigma(s)`
has a coefficient of variation of `0.016-0.040` — intervals barely adapt — even though it
correlates `r ~ 0.78` with the true per-sample scale. Inflating its dynamic range makes
Winkler *worse* at every cut, so the small spread is not an attenuation artefact.

**This bounds the scalar geometry, not state conditioning.** On the RevIN runs cluster
SCCP wins 12/12 despite the per-sample share falling to `0.73%`, which is only possible
if its gain comes from somewhere other than the per-sample axis. It does: decomposing the
fitted per-cluster quantile arrays, the per-cluster *level* varies by a coefficient of
variation of only `0.029`, while the normalised per-cluster *shape* deviates by
`0.20-0.26`. The conditioning signal is a **state x cell interaction** — the state
changes which cells are hard, not how hard the window is overall. A scalar sigma
structurally cannot represent that; a per-cell sigma can.

`StateScaledConformal(scale_geometry='per_cell')` fits one log-linear regression per
horizon-feature cell against the same design matrix (one shared Gram factorisation, so
the cost over the scalar geometry is a single extra matrix multiply). Over three RevIN
seeds x four cuts it is `-0.237` Winkler against flat CP, better on 12/12, at matched
coverage — roughly double cluster SCCP's `-0.124`, and better than the scalar *oracle*.

Controls (RevIN, cut 0.5): the result is insensitive to ridge over `[1e-4, 1e2]`
(3.489 → 3.426); coverage matches flat CP (0.82 both), so the Winkler win is not bought
by under-covering; and permuting the states before fitting collapses it back to flat
(3.813 vs flat 3.797, against 3.403 unpermuted), so the gain genuinely comes from the
state. On the pre-RevIN run per-cell sigma *loses* badly (6.5 vs flat 5.15), the same
regime-dependence as every other conditioning mechanism here.

### Protocol result and its remaining gap

Under the real protocol — where sigma is fit on **train** states/residuals, not a
held-out chronological window — the per-cell geometry improves on flat CP but recovers
only part of the diagnostic's margin (ETTh1-h336, seed 42, RevIN, all three calibrated on
the same residuals from the same model):

| mechanism | Winkler | coverage | width |
| --- | --- | --- | --- |
| flat CP | 3.7869 | 0.9086 | 2.6022 |
| state-scaled, scalar sigma | 3.7877 | 0.9116 | 2.6379 |
| state-scaled, per-cell sigma | 3.6916 | 0.9163 | 2.6233 |
| cluster SCCP | 3.5962 | 0.9058 | 2.5625 |

The cause of the gap is measurable: the protocol fits sigma on in-sample train residuals,
which the model was trained to shrink. The per-cell coefficient spread across cells is
`0.076` when fit on train residuals against `0.137` when fit on out-of-sample residuals —
the in-sample fit sees roughly half the true state-to-cell coupling. (That the spread
across cells, `0.076`, dwarfs the mean coefficient magnitude, `0.007`, independently
confirms the effect is cell-specific rather than a shared level term.)

This is a limitation of fitting the difficulty estimator on data the forecaster has
already fit, not of the per-cell geometry. Closing it would mean giving sigma its own
held-out fitting window — a change to the split contract, which is protocol work, not a
tuning change, and is not undertaken here.

### What this means for the claim

State conditioning helps in the RevIN regime and does not help without it. Cluster SCCP
is currently the strongest mechanism under the real protocol; per-cell state-scaled CP is
second and beats both flat CP and the scalar geometry it replaces. `scale_geometry` is
off by default (`scalar`), so existing runs and artifacts are unchanged.

None of this is yet a protocol result across datasets: everything above is ETTh1-h336.
Before any of it is published it needs the locked multi-dataset grid. Do not retune
`sigma(s)` against these numbers.

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

### Open: capacity against sample count

A separate, uninvestigated concern sits alongside the rank constraint. The ETTh1-h336
train split holds 6481 samples at a parameter-per-sample ratio of roughly 15, and the
training curves show the expected signature: train loss falls `1.17 → 0.55` while
validation stays flat near `1.73` from epoch 1. This is distinct from the fan-in
initialisation fix already applied to the refinement head.

No regularisation sweep has been run against it, and the interventions in the table above
(`--lambda_refinement`, `--dropout`) made accuracy worse rather than better. Two defensible
options: run a dedicated sweep, or report the overfitting evidence above as a known
limitation without further tuning. The second is consistent with the rank analysis — if
the binding constraint is rank 5, extra regularisation cannot recover the gap — but the
choice has not been made.

## Validation-to-test transfer on ETTh1

Checkpoint selection on validation does not reliably transfer to test on this dataset. Standardised by train statistics, split means are `train 0.000`, `val -0.108`, `test -0.049`, with test standard deviation `1.123` against train `1.000`: validation and test differ from train *and from each other*.

Observed directly in the hybrid run at horizon 336: correction-stage validation loss improved from `1.160` to `1.112` (-4.1%) and the epoch-0 fallback correctly did not trigger, yet test MSE was `0.799` against the frozen DLinear base's `0.619`. The fallback machinery behaved exactly as specified; the validation signal itself was misleading.

Treat single-seed, single-horizon validation improvements on ETTh1 as weak evidence, and require the multi-seed multi-dataset protocol before promoting any architecture on validation deltas alone.

The current mitigation is procedural, not structural: `RUNBOOK.md` Step 5 requires agreement across at least two of three seeds precisely because one ETTh1 validation delta is not trustworthy. An open alternative, not adopted, is to drop ETTh1 from the *selection* stage entirely and keep it only in the main reporting grid. That would remove the weakest selection signal at the cost of selecting on datasets the headline table also reports; it is a scope decision that has not been made, and nothing below assumes it.

## Simultaneous coverage

`per_feature` geometry reports `coverage_joint = 0.0000` at horizon 336: it calibrates each of the 2352 horizon-feature cells to its own marginal quantile, so simultaneous coverage of every cell is not what the method targets and near-zero is the expected reading, not a defect.

`--multivariate_strategy max` is the geometry that targets simultaneous coverage. Measured on ETTh1-h336 with `--revin`, seed 42:

| geometry | marginal | joint | mean width | Winkler |
| --- | --- | --- | --- | --- |
| `per_feature` | 0.9071 | 0.0000 | 2.56 | 3.57 |
| `max` | 0.9953 | 0.4923 | 8.10 | 8.16 |

`max` produces genuine simultaneous coverage but reaches only `0.49` against a nominal `0.90`, at roughly `3.2x` the interval width. The shortfall is not a finite-sample artefact — the calibration split holds 1393 origins and the exact-rank index `k = 1255` is attainable. It reflects test-time dependence across cells: under independence, per-cell coverage of `0.9953` would give a joint rate near `1.5e-05`, so the observed `0.49` indicates strong positive dependence, and residual distribution shift between calibration and test then costs the remainder.

Do not report `max` as a validated simultaneous-coverage guarantee. Either report it as an honest negative result with the numbers above, or restrict simultaneous claims to a smaller, prespecified cell block where the target is attainable.
