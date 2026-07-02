# CISSN arXiv v1 — Experiment Plan

**Title (working):** CISSN: Interpretable Structured State-Space Forecasting with
State-Conditional Conformal Intervals

**Headline result:** Competitive point-forecast accuracy *and* tighter, calibrated
prediction intervals *and* an interpretable Level / Trend / Seasonal / Residual
decomposition on standard time-series benchmarks.

> **Prerequisite:** The conformal-quantile divisor fix (A1) must be committed before
> any results are recorded. All MPIW figures reflect the corrected
> `⌈(n+1)(1−α)⌉/(n+1)` level, not the old over-covering `/n_k` variant.

---

## Scope (locked)

| Axis | Values |
|------|--------|
| Datasets | `ETTh1`, `weather` |
| Horizons | `96`, `336`, `720` |
| Seeds | `42`, `123`, `456` |
| Conformal | `alpha=0.1`, `strategy=max`, `n_clusters=5` |

Rationale: ETTh1 is the standard mid-frequency ETT benchmark; weather is high-dimensional
(21 variates) and widely used in recent LTSF literature. Two datasets keep the preprint
tractable while covering different regimes. The full Q1-journal grid (ETTh1/ETTh2/weather/
exchange_rate, horizons 96–720) is documented in `document.md`.

---

## Cell Counts

| Component | Formula | Cells |
|-----------|---------|-------|
| CISSN grid | 2 datasets × 3 horizons × 3 seeds | **18** |
| Point baselines | 3 models × 2 datasets × 3 horizons × 3 seeds | **54** |
| UQ: MC-Dropout | 1 method × 2 datasets × 3 horizons × 3 seeds | **36** |
| UQ: FlatCP | 1 method × 2 datasets × 3 horizons × 3 seeds | **18** |
| Ablations (ETTh1 only) | 5 variants × 3 horizons × 3 seeds | **45** |
| **Total** | | **171** |

Ablation variants (ETTh1): `full`, `no_structured_A`, `no_disentanglement_loss`,
`no_correction_mlp`, `state_dim_4`. The `flat_cp` variant is excluded here — it runs
as a dedicated H2 UQ baseline on both datasets via `--ablations flat_cp` (see below).

---

## Comparisons

### Point baselines
- **DLinear** (Zeng et al., AAAI 2023) — channel-independent linear decomposition
- **PatchTST** (Nie et al., ICLR 2023) — patch-based Transformer
- **DeepState** (Rangapuram et al., NeurIPS 2018) — GRU + structured SSM

### UQ baselines
- **FlatCP** — marginal conformal baseline, no state-conditioning (via `run_ablation.py`)
- **MCDropout** (Gal & Ghahramani, ICML 2016) — stochastic inference

> Deep Ensemble (Lakshminarayanan et al., NeurIPS 2017) is deferred to the journal
> version — it requires 5× the training compute and adds no new scientific dimension
> for the preprint.

---

## Hypotheses Tested

| ID | Hypothesis | Coverage |
|----|-----------|---------|
| H2 | State-conditional intervals are tighter than marginal FlatCP at equal empirical coverage | ✅ Full — MPIW SCCP vs FlatCP, same alpha, same data |
| H3 | CISSN accuracy is competitive with DLinear/PatchTST/DeepState | ✅ Full — MSE/MAE tables |
| H1 | Each state dimension captures its intended dynamics (Level/Trend/Seasonal/Residual) | Partial — ablation `no_structured_A` + `no_disentanglement_loss` quantify contribution |

---

## Metrics & Tables

**Point accuracy** (mean ± std over 3 seeds):
- MSE, MAE

**Interval quality** (mean ± std over 3 seeds):
- Empirical coverage (PICP) — must be ≥ 0.90 at α=0.10
- MPIW (mean prediction interval width) — primary efficiency metric for H2
- Winkler score
- Calibration error

**Per-cluster coverage breakdown** (ETTh1, horizon 96):
- Coverage per K-Means cluster → interpretability figure

---

## Runnable Commands

### CISSN grid

```bash
# ETTh1 all horizons, 3 seeds
uv run python experiments/run_multiseed.py \
  --data ETTh1 --all_horizons --seeds 42,123,456 \
  --checkpoints ./checkpoints/arxiv_v1 \
  --results_dir ./results/arxiv_v1 \
  --n_clusters 5 --conformal_alpha 0.1 --multivariate_strategy max \
  --train_epochs 30 --lradj cosine \
  --output ./results/arxiv_v1/cissn_ETTh1.json \
  --raw_csv ./results/arxiv_v1/cissn_ETTh1_raw.csv

# weather all horizons, 3 seeds
uv run python experiments/run_multiseed.py \
  --data weather --all_horizons --seeds 42,123,456 \
  --checkpoints ./checkpoints/arxiv_v1 \
  --results_dir ./results/arxiv_v1 \
  --n_clusters 5 --conformal_alpha 0.1 --multivariate_strategy max \
  --train_epochs 30 --lradj cosine \
  --output ./results/arxiv_v1/cissn_weather.json \
  --raw_csv ./results/arxiv_v1/cissn_weather_raw.csv
```

### Point baselines

```bash
for MODEL in dlinear patchtst deepstate; do
  for DATA in ETTh1 weather; do
    for HORIZON in 96 336 720; do
      for SEED in 42 123 456; do
        uv run python experiments/run_baseline.py \
          --model $MODEL --data $DATA --pred_len $HORIZON --seed $SEED \
          --checkpoints ./checkpoints/arxiv_v1 \
          --results_dir ./results/arxiv_v1 \
          --train_epochs 30 --lradj cosine
      done
    done
  done
done
```

### UQ baselines

**MC-Dropout** (via `run_baseline.py`):

```bash
for DATA in ETTh1 weather; do
  for HORIZON in 96 336 720; do
    for SEED in 42 123 456; do
      uv run python experiments/run_baseline.py \
        --model mc_dropout --data $DATA --pred_len $HORIZON --seed $SEED \
        --conformal_alpha 0.1 \
        --checkpoints ./checkpoints/arxiv_v1 \
        --results_dir ./results/arxiv_v1 \
        --train_epochs 30 --lradj cosine
    done
  done
done
```

**FlatCP** (via `run_ablation.py --ablations flat_cp`; same CISSN encoder/head, marginal conformal):

```bash
for DATA in ETTh1 weather; do
  for HORIZON in 96 336 720; do
    for SEED in 42 123 456; do
      uv run python experiments/run_ablation.py \
        --data $DATA --pred_len $HORIZON --seed $SEED \
        --ablations flat_cp \
        --train_epochs 30 \
        --output "./results/arxiv_v1/flat_cp/${DATA}_h${HORIZON}_s${SEED}.json"
    done
  done
done
```

### Ablations (ETTh1 only)

```bash
for HORIZON in 96 336 720; do
  for SEED in 42 123 456; do
    uv run python experiments/run_ablation.py \
      --data ETTh1 --pred_len $HORIZON --seed $SEED \
      --ablations full,no_structured_A,no_disentanglement_loss,no_correction_mlp,state_dim_4 \
      --train_epochs 30 \
      --output "./results/arxiv_v1/ablations/ETTh1_h${HORIZON}_s${SEED}.json"
  done
done
```

### Smoke test (single cell, fast)

```bash
# Verify the corrected conformal code calibrates and coverage >= 0.9
uv run python experiments/run_benchmark.py \
  --data ETTh1 --pred_len 96 --seed 42 \
  --train_epochs 3 --n_clusters 5 \
  --conformal_alpha 0.1 --multivariate_strategy max \
  --checkpoints ./checkpoints/arxiv_v1_smoke \
  --results_dir ./results/arxiv_v1_smoke
```

---

## Artifact Layout

```
results/arxiv_v1/
  CISSN_ETTh1_M_sl96_pl96_*/   metrics.json, preds.npy, trues.npy
  CISSN_weather_M_sl96_pl96_*/
  ...
  ablations/
    ETTh1_full_pl96_seed42/
    ETTh1_no_structured_A_pl96_seed42/
    ...

checkpoints/arxiv_v1/
  CISSN_ETTh1_M_sl96_pl96_*/   checkpoint.pth, checkpoint_head.pth
  ...
```

---

## Writing Gates

1. All 18 CISSN cells complete → draft Table 1 (point metrics) and Table 2 (interval metrics)
2. All 54 point-baseline + 36 MC-Dropout + 18 FlatCP cells complete → finalize comparison tables; compute H2/H3 conclusions
3. All 45 ablation cells complete → draft Table 3 (ablation); conclude H1 partial claim
4. Coverage ≥ 0.90 on all CISSN cells → confirm conformal guarantee holds post-fix
5. SCCP MPIW < FlatCP MPIW on ≥ 4/6 (dataset × horizon) pairs → H2 confirmed

---

## Compute Requirements

CISSN is a compact model (~25–35 K trainable parameters), but the arXiv preprint requires
approximately 171 training cells across the full grid, ablations, and UQ baselines. Each
cell involves up to 720-step horizon forecasting over multivariate sequences (21 variates
for weather), with conformal calibration requiring a full forward pass over the validation
split. MC-Dropout adds 30 stochastic forward passes per test sample. A single **NVIDIA
A100 40 GB** (floor: RTX 4090 24 GB) comfortably accommodates all batch_size-32
long-horizon cells and MC-Dropout ensemble passes within GPU memory. On Linux,
`torch.compile` accelerates the recurrent encoder loop further. With one such card the
full 171-cell grid runs in approximately 2–3 days; without GPU it would require weeks.
The binding memory constraint is the horizon-720 multivariate case (weather, 21 variates,
batch 32), not raw FLOPs — an A100 40 GB provides sufficient headroom. Three seeds are
required for publication-quality mean ± std reporting, and the conformal calibration pass
is non-negotiable (each cell uses the full validation set for cluster quantile estimation),
making distributed CPU fallback impractical at scale.
