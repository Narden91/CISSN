# CISSN publication runbook

This is the sole launch protocol. Results from earlier protocols are not publication evidence and must not be pooled with these artifacts.

> Evidence note: ETTh1 test artifacts have already informed development. They remain diagnostics only. Selection must use chronological pre-test folds; ETTh2, weather, and exchange-rate tests stay sealed until a decision lock is written.

## Execution status

Checkbox meaning: `[x]` an artifact on disk satisfies the step under the *current* code;
`[ ]` the step must be launched. Runs that exist but cannot satisfy their step have been
moved to `results/superseded/` with a per-run reason in that directory's `README.md`; they
stay available as diagnostics and are never pooled into a publication table. Audited
against `results/` on 2026-08-27, branch `conformal-per-cell-geometry`.

`results/` now holds exactly one usable artifact (Step 2). Launch commands for everything
outstanding are in `commands.md`.

| Step | State | Evidence on disk |
| --- | --- | --- |
| 1 environment, data, tests | `[x]` | verified 2026-08-27: CUDA available (torch `2.11.0+cu128`, RTX 5080), `verify_datasets.py` OK on all 10 datasets, 163 tests pass |
| 2 DLinear reference (ETTh1-h336/s42) | `[x]` | `results/validation/BASELINE_dlinear_ETTh1_M_sl96_pl336_seed42` — clean git, `structural_passed`, no quality flags, test MSE `0.619` |
| 3 CISSN end-to-end (ETTh1-h336/s42) | `[ ]` | superseded run moved to `results/superseded/CISSN_ETTh1_h336_seed42_preRevIN/` — pre-RevIN, and its `metrics.json` holds only `interval` (no `interval_flat_cp`/`_cluster_cp`/`_state_scaled`), so it predates the three-mechanism contract |
| 3b.0 headroom diagnostic, ETTh1-h336 RevIN | `[ ]` | superseded, moved to `results/superseded/headroom/` and `.../percell/` — `git_dirty: true` at commit `d749709`, i.e. before the fitting-set fix, so every cluster-vs-scale ordering from them is confounded |
| 3b.0 headroom diagnostic, other datasets | `[ ]` | ETTh2 / weather / exchange-rate not run |
| 3b.2 validation-only selection runs | `[ ]` | `results/selection/` does not exist; no `selection.json` anywhere |
| 4 RevIN selection (ETTh2, weather, 3 seeds, paired) | `[ ]` | RevIN evidence to date is ETTh1 **test**, which is diagnostic and cannot select |
| 5 hybrid variant selection (3 variants x 3 seeds) | `[ ]` | no hybrid run directories |
| Prespecified statistics recorded | `[ ]` | primary endpoint and primary horizon not written into `protocol.json` or an adjacent prespecification file |
| Main grid — CISSN, 4 datasets | `[ ]` | `results/publication/` does not exist |
| Main grid — baselines, 5 models x 4 datasets | `[ ]` | none |
| Ablations | `[ ]` | none |
| Publication review (tables/figures/appendix) | `[ ]` | none |

Two findings constrain what can be launched next:

- **No run on disk carries an evidence role.** Every `protocol.json` has keys
  `config/dataset/protocol/protocol_hash/source` and no `evidence` block, so all six runs
  are development artifacts. None is eligible for the confirmation grid regardless of its
  numbers.
- **Every existing run is ETTh1-h336.** The three sealed datasets have never been opened,
  which is the intended state — and it also means no step past 3b has any evidence.

Recommended order: Step 1, then re-run Step 3 under `--revin` on current code, then Step
3b.0 against those fresh artifacts, then the Step 3b.2 and Step 4 selection runs, then the
decision lock, then the prespecification, then the main grid.

## Locked design

- Datasets: `ETTh1`, `ETTh2`, `weather`, `exchange_rate`.
- Seeds: `42,123,456`.
- Horizons: registry-supported horizons only; use `24,96,192,336,720` for ETT and `96,192,336,720` for weather and exchange-rate.
- Shared training: `--train_epochs 20 --patience 5 --lradj cosine --batch_size 128`.
- Shared calibration: chronological train-tail calibration split, `--cal_fraction 0.2`, `--conformal_alpha 0.1`.
- Primary interval geometry: `--multivariate_strategy per_feature`; report `coverage_primary` (marginal). `max` is a separate simultaneous-coverage analysis.
- Publication safeguards: `--require_gpu --require_clean_git`.

The cluster state partition (`StateConditionalConformal.fit_partition`) and the state-scaled sigma regression (`StateScaledConformal.fit_scale`) are both learned from the same window — the first half of the calibration split — then quantile calibration uses the second half. This equalises the two mechanisms' fitting-set size and in-sample status; see `docs/methodology.md` for the earlier asymmetric version and why prior cluster-vs-scale orderings need re-measuring under the current code. Serial dependence is documented in artifacts; it is not converted into a coverage guarantee.

All experiment runners show live batch progress for training, validation, partitioning, calibration, and testing. Use `--no_progress` only for CI or captured logs.

## Step 1: environment, data, tests `[x]`

```powershell
uv sync
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
uv run python scripts/verify_datasets.py
uv run python tests/run_tests.py
```

Stop if any command fails. `verify_datasets.py` must pass for the four locked datasets.

## Step 2: DLinear reference reproduction `[x]` ETTh1-h336/seed 42

```powershell
uv run python experiments/run_baseline.py --model dlinear --data ETTh1 --pred_len 336 --seed 42 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --multivariate_strategy per_feature --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation
```

Review `results/validation/*/sanity.json`, `metrics.json`, `history.json`, and `protocol.json`. Every run writes these artifacts. Exclude a run from publication tables only when `structural_passed` is false, which means the artifact is unreadable (empty, non-finite, shape-inconsistent, or inverted interval bounds). Forecast quality never removes a run: `quality.flags` is advisory, and a finite but poor forecast is a valid result that must stay visible. Record the expected full-train reference and the observed fair split result separately; this protocol intentionally reserves train data for calibration.

## Step 3: CISSN end-to-end `[ ]`

```powershell
uv run python experiments/run_benchmark.py --data ETTh1 --pred_len 336 --seed 42 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --require_gpu --require_clean_git --checkpoints ./checkpoints/validation --results_dir ./results/validation
```

Include the result unless `structural_passed` is false. Quality flags in `sanity.json` are advisory and are reported alongside the result, never used to drop it. Check `cluster_stats.json` and `scale_stats.json` for fallback clusters and fitted scale coefficients, and `dependence_diagnostics.json`, before interpreting coverage.

## Step 3b: state conditioning mode confirmation `[ ]`

> Selection override: this legacy section is development-only because its diagnostic and commands read test artifacts. Do not use any outcome here to choose conditioning. The choice belongs to `experiments/run_selection.py` (`--evidence_role selection`), which trains and calibrates identically but scores every mechanism on the validation split only -- the test loader is never constructed -- followed by a decision lock. See Step 3b.2 below.

`--conformal_conditioning cluster` (K-Means partition, the historical mechanism) is the
default. `--conformal_conditioning scale` (continuous `sigma(state)`, normalized
conformal score) is an alternative. Every run already calibrates and reports both plus
flat CP regardless of the flag, so this step only decides which one drives the headline
`interval` block and which becomes the CLI default.

**The original evidence for promoting `scale` was withdrawn and then partly replaced.**
The first diagnostic calibrated flat CP on a window twice the size of state-scaled CP's,
so it was not paired. The corrected diagnostic established two things
(`docs/methodology.md`): a **scalar** sigma has essentially no headroom (a scalar
label-informed per-sample reference using test labels gains under 1%), and the
conditioning signal is a state x cell interaction that only a **per-cell** sigma or the
cluster predictor can express.

Development measurements on ETTh1-h336, three RevIN seeds x four nested cuts (cut 0.3's
window strictly contains cut 0.6's, so this is not 12 independent trials — effective n is
closer to 3, and the seeds share one dataset/split, so a population-level claim has an
effective n closer to 1) — mean Winkler delta vs flat CP: cluster `-0.124` (better on all
3 seeds x 4 cuts), scalar sigma `+0.011` (better on 5 of 12), per-cell sigma `-0.237`
(better on all 12). Under the real protocol on seed 42 the ordering was cluster `3.5962` <
per-cell `3.6916` < flat `3.7869` ~ scalar `3.7877` — but that ordering was measured while
cluster and state-scaled CP had a ~9x conditioning-fit sample-size asymmetry (see
`docs/methodology.md`). `run_benchmark.py` now fits both mechanisms on the same
calibration-half window, so this ordering must be re-measured under the current code
before it can be used to select a mode.

### Step 3b.0: headroom diagnostic on saved artifacts (do this first, no GPU) `[ ]`

The diagnostic needs a run directory containing `states.npy`, `residuals.npy`,
`pred.npy`, and `true.npy` — any completed `run_benchmark.py` CISSN run writes all four
(baseline runs do not, since they have no latent state). Point it at a `--revin` run for
the dataset under test:

```powershell
uv run python scripts/diagnose_conditioning_headroom.py --run_dir ./results/validation/CISSN_<dataset>_..._fullrevin_..._seed42 --output ./results/validation/conditioning_headroom_revin.json
```

Read `variance_decomposition.per_sample_fraction` (expect ~1%: that bounds the *scalar*
geometry, not conditioning as such) and the `summary_vs_flat` win counts. Run this on
each dataset before spending seeds on it — the conditioning result is regime-dependent
and there is no reason to assume ETTh1's carries over.

ETTh1-h336 was measured and is recorded in `docs/methodology.md`, but under the earlier
asymmetric fitting scheme; those artifacts now sit under `results/superseded/headroom/`
and `results/superseded/percell/` (see that directory's `README.md`). They are development
runs, not `--require_clean_git` publication evidence — reuse them for diagnostics, never
pool them into publication tables, and re-run this diagnostic against fresh Step 3
artifacts before citing any cluster-vs-scale ordering.

### Step 3b.2: validation-only selection runs `[ ]`

Run both geometries through `experiments/run_selection.py` (`--evidence_role selection`
is forced automatically) so the comparison stays paired within one model, on validation
only -- the test loader is structurally never constructed by this runner:

```powershell
uv run python experiments/run_selection.py --data ETTh1 --pred_len 336 --seed 42 --revin --conformal_conditioning scale --scale_geometry per_cell --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
```

Repeat for seeds `123,456`. Read `interval`, `interval_flat_cp`, and `interval_cluster_cp`
from each run's `selection.json` (same field names as `metrics.json`, scored on
`validation_mse`/validation Winkler instead of test). **Decision point**: promote `scale`
to the default conditioning mode only if `interval_state_scaled` beats **both**
`interval_flat_cp` and `interval_cluster_cp` on Winkler on at least 2 of 3 seeds. Do not
retune against test data, and do not read any test artifact while making this decision.

Record the observed per-seed deltas here either way, then run the confirmation cell (Step
3b.1 above showed the ETTh1 test numbers for context only -- they are diagnostic and
cannot be used to make this decision) before proceeding to Step 4.

## Step 4: instance normalisation `[ ]`

> Selection override: historical ETTh1 test values below are diagnostic context only, not
> a default-selection procedure. Choose RevIN from `experiments/run_selection.py`
> (validation only, test loader never constructed) before opening any final test.

Legacy CISSN collapses toward the training mean on ETTh1-h336: the forecast keeps only ~7% of the target's variance, which lowers MSE without tracking the signal. `--revin` removes the level-tracking burden that causes this. Measured over seeds `42,123,456` on the ETTh1 test split (see `docs/methodology.md`; diagnostic only, not the selection evidence), test MSE falls `1.280 → 0.771` and coverage moves `0.788 → 0.908` with narrower intervals, with no change to the latent state dimension.

```powershell
uv run python experiments/run_selection.py --data ETTh1 --pred_len 336 --seed 42 --revin --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
```

Check `vali_variance_ratio` in `history.json`. A ratio that falls while validation loss improves is amplitude collapse, not a hard dataset, and must be reported as such rather than attributed to the state bottleneck.

`--revin`'s effect is proven only on ETTh1-h336 test artifacts and remains opt-in (off by
default) until confirmed on validation elsewhere. Before making it the default for
`--architecture legacy`, run it with and without on ETTh2 and weather at h336, 3 seeds
each, through the selection controller:

```powershell
uv run python experiments/run_selection.py --data ETTh2 --pred_len 336 --seed 42 --revin --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
uv run python experiments/run_selection.py --data weather --pred_len 336 --seed 42 --revin --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
```

Repeat each without `--revin` for the paired comparator. Flip the default only if RevIN
does not regress `validation_mse` or validation coverage (`selection.json`) on either
dataset relative to the non-RevIN arm; if it regresses anywhere, keep it opt-in and pass
`--revin` explicitly per dataset in the main grid rather than defaulting a fix proven on
one dataset/horizon. Record the outcome and per-seed numbers here either way.

## Step 5: hybrid variant selection (validation only) `[ ]`

The legacy architecture routes the forecast through a five-dimensional latent bottleneck.
Without RevIN this caps forecast rank at 5. With RevIN's side statistics the cap loosens
to roughly `5 + 2C`, and measured effective rank on RevIN runs is 7-8 (see
`docs/methodology.md`, "Rank and the DLinear gap under RevIN") — a real but smaller
deficit than the hard-5 case, not the absence of one. The
hybrid keeps DLinear as the base and gives the state an additive correction:

```text
total = DLinear(history) + LinearCorrectionHead(StateEncoder(history))
```

The correction head is zero-initialised and the base is frozen during
correction training, so correction-stage epoch 0 reproduces the frozen DLinear
exactly and a failed correction stage falls back to it. `runtime.json` records
`base_val_loss` and `correction_improved_on_base` for every hybrid run.

Run the three prespecified variants on ETTh1-h336 with seeds `42,123,456` through the
selection controller, which trains identically but never constructs the test loader:

```powershell
uv run python experiments/run_selection.py --architecture hybrid --data ETTh1 --pred_len 336 --seed 42 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
uv run python experiments/run_selection.py --architecture hybrid --state_dynamics anchored --data ETTh1 --pred_len 336 --seed 42 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
uv run python experiments/run_selection.py --architecture hybrid --state_dynamics anchored --state_revin --data ETTh1 --pred_len 336 --seed 42 --lambda_cov 0 --lambda_temp 0 --train_epochs 20 --patience 5 --lradj cosine --require_gpu --require_clean_git --checkpoints ./checkpoints/selection --results_dir ./results/selection
```

Select on **validation** MSE only (`validation_mse` in each run's `selection.json`, or
`vali_loss` in `history.json`). Promote a hybrid variant only when it shows
at least `1%` mean improvement over paired DLinear, improves on at least two of
three seeds, is no worse than `2%` on any seed, and degrades no feature's mean
MSE by more than `5%`. Ties within `0.5%` resolve to anchored dynamics without
RevIN.

`--state_revin` rescales the correction by the input-window feature scale. This
preserves epoch-0 exactness but changes the epoch-0 gradient scale, so the RevIN
variant is not comparable to the other two at an equal learning rate; treat it
as a separate arm rather than a drop-in ablation.

If no variant meets these criteria, stop: report the hybrid as a negative result
and do not retune against test data.

## Prespecified statistics (write before opening the sealed datasets) `[ ]`

`RUNBOOK.md` locks 4 datasets x 4-5 horizons x 3 seeds, with per-cell comparisons of the
primary mechanism against flat CP and against the secondary mechanism -- on the order of
17-19 cells per comparison. Reporting all of them as a grid of means with no correction
and no prespecified primary endpoint invites a favourable-looking cell to be read as the
result. Before ETTh2, weather, or exchange-rate test splits are opened, record here:

- **Primary endpoint**: mean Winkler delta of the primary conditioning mechanism (as
  selected in Step 3b.2) vs flat CP, aggregated across the full locked grid (all
  datasets/horizons/seeds), with a paired test over cells (e.g. a sign test or Wilcoxon
  signed-rank across the per-cell mean deltas). This is the one number a paper abstract
  may state as confirmatory.
- **Primary horizon**: h336 is the de facto candidate, simply because every development
  and selection measurement so far has used h336 (`docs/methodology.md`,
  `results/superseded/`). That is a selected horizon, not a neutral
  default, unless prespecified here explicitly. Record the chosen primary horizon and
  the reason before running the grid at other horizons.
- **Per-cell win-count rule is descriptive, not confirmatory**. `n=3` seeds per cell means
  a "wins on 2 of 3 seeds" reading (Step 3b.2, Step 5) has roughly a 50% false-positive
  rate per cell under a null of no effect. Report per-cell win counts alongside means as
  descriptive detail; do not present them as a per-cell significance claim. The same
  applies to any "N/M nested cuts" figure quoted from `diagnose_conditioning_headroom.py`
  (see `docs/methodology.md`) -- the cuts are nested, not independent trials.
- **Every other cell/comparison in the grid is secondary/descriptive**, reported as means
  with win counts, not as individually confirmatory results.

Record the filled-in choices above in `protocol.json` (or an adjacent prespecification
file) before Main grid runs begin, so the manifest carries the prespecification and a
later reader can check the analysis was not chosen after seeing the sealed results.

## Main grid `[ ]`

This is the sealed confirmation grid: every command below must pass
`--evidence_role confirmation`, which `enforce_evidence_contract` requires be paired with
`--immutable_artifacts --strict_determinism --require_clean_git` (all three, not just
`--require_gpu --require_clean_git`). This is what makes a run a sealed, reproducible,
non-overwritable confirmation artifact rather than a development run; without
`--evidence_role confirmation` a run defaults to `development` and is not eligible for the
publication table regardless of its numbers.

Run CISSN with the multi-seed driver for each locked dataset, using whichever
`--conformal_conditioning` mode Step 3b.2 selected on validation (every run still
calibrates and reports both, so this only sets which one is primary):

```powershell
uv run python experiments/run_multiseed.py --data ETTh1 --all_horizons --seeds 42,123,456 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --n_clusters 5 --multivariate_strategy per_feature --conformal_conditioning scale --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --output ./results/publication/cissn_ETTh1.json --raw_csv ./results/publication/cissn_ETTh1.csv
```

Repeat with `ETTh2`, `weather`, and `exchange_rate`. The driver must propagate all safeguards; verify this with `--help` before a long launch.

For baselines, run each model/dataset/horizon/seed with the same shared options. RevIN
applies only to `dlinear`, `patchtst`, `mc_dropout`, `deep_ensemble` (DeepState is
excluded -- see `docs/methodology.md`), and only once Step 4 has selected it on
validation; pass it explicitly per dataset rather than assuming a default:

```powershell
$models = 'dlinear','patchtst','deepstate','mc_dropout','deep_ensemble'
$datasets = 'ETTh1','ETTh2','weather','exchange_rate'
$seeds = 42,123,456
foreach ($model in $models) {
  foreach ($data in $datasets) {
    foreach ($h in 96,192,336,720) {
      foreach ($seed in $seeds) {
        uv run python experiments/run_baseline.py --model $model --data $data --pred_len $h --seed $seed --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --multivariate_strategy per_feature --require_gpu --require_clean_git --immutable_artifacts --strict_determinism --evidence_role confirmation --checkpoints ./checkpoints/publication --results_dir ./results/publication
      }
    }
  }
}
```

For ETT, add the horizon `24`. For UQ models the default is conformalized intervals; raw output requires explicit `--uq_interval_mode raw` and is secondary.

Run ablations only after the main comparison completes:

```powershell
uv run python experiments/run_ablation.py --data ETTh1 --pred_len 336 --seed 42 --ablations full,no_structured_A,no_disentanglement_loss,flat_cp,no_correction_mlp,state_dim_4 --train_epochs 20 --patience 5 --lradj cosine --batch_size 128 --conformal_alpha 0.1 --multivariate_strategy per_feature --require_gpu --require_clean_git --output ./results/publication/ablations_ETTh1_h336_s42.json
```

Every `run_benchmark.py` run already reports flat CP, cluster SCCP, and the *selected*
scale geometry paired on the same forecasts, so the main grid covers most conditioning
comparisons without a dedicated ablation arm. Two cells are **not** reachable from it,
and need an added arm if the ablation table requires them: both scale geometries within a
single run (a run fixes one `--scale_geometry`), and any conditioning mode crossed with
an architecture ablation such as `no_structured_A` or `state_dim_4`.

## Publication review `[ ]`

Before aggregating, every result must have the full artifact contract and a protocol manifest showing the same split, calibration, and shared training settings for the comparison cell. Exclude incomplete results, results with `structural_passed: false`, different splits, and raw-UQ results from the primary table. Do not exclude a run for poor forecast quality; report it with its advisory flags.

```powershell
uv run python scripts/generate_publication_tables.py --results_root ./results/publication --output_dir ./results/publication/tables
uv run python scripts/generate_publication_figures.py --results_root ./results/publication --output_dir ./results/publication/figures
uv run python scripts/generate_reproducibility_appendix.py --results_root ./results/publication --output ./results/publication/reproducibility.md
```

`table_paired_comparison.csv` is the central evidence table: per dataset/horizon cell,
the primary conditioning mechanism's mean Winkler delta against flat CP, cluster SCCP,
and state-scaled CP, plus how many seeds it actually won on
(`winkler_delta_vs_*_wins` out of `winkler_delta_vs_*_count`). Report the win count
alongside the mean — a favorable mean can still hide a sign that flips seed to seed, as
it does for cluster-based SCCP against flat CP on some chronological cuts (see
`docs/methodology.md`).

Report mean and standard deviation across outer seeds, paired method comparisons, `coverage_primary`, width, Winkler score, MSIS, coverage scope, interval origin, `worst_prespecified_bin_coverage`/`max_coverage_deviation`, cluster/scale fallbacks, and dependence diagnostics. Do not describe the time-series result as unconditional distribution-free coverage.
