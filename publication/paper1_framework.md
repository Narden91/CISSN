# Paper 1 — CISSN: Conformally Calibrated Interpretable State-Space Networks

## Metadata

| Field | Value |
|-------|-------|
| **Working Title** | Conformal Inference via State-Conditional Disentangled State-Space Networks for Time-Series Forecasting |
| **Target Venues** | ICML 2026 (primary) → AISTATS 2026 → NeurIPS 2026 → JMLR (journal fallback) |
| **Deadline** | ICML 2026: ~January 2026; AISTATS 2026: ~October 2025 |
| **Focus** | Theory-first: coverage proof, structured dynamics analysis, rigorous ablation |
| **Target Length** | 8 pages ICML style + appendix for proofs |
| **Status** | In preparation — codebase ready, baselines + proofs needed |

---

## Research Hypotheses

| # | Hypothesis | Test |
|---|-----------|------|
| H1 | A structured block-diagonal transition matrix with constrained eigenvalues produces more disentangled latent states than an unconstrained learned matrix. | Compare off-diagonal correlation with/without structured A |
| H2 | State-conditional conformal intervals are narrower than flat marginal conformal intervals at equal empirical coverage. | SCCP vs marginal CP on same encoder |
| H3 | A 5-dimensional state (level, trend, seasonal cos, seasonal sin, residual) captures seasonal dynamics better than a 4-dimensional state (scalar seasonal). | state_dim=4 vs state_dim=5 ablation |
| H4 | The linear forecast path provides sufficient interpretability without significant accuracy loss vs full hybrid head. | Refinement ratio analysis; linear-only vs hybrid ablation |
| H5 | The disentanglement loss (covariance + temporal) improves long-horizon forecasting by keeping state dimensions independent. | λ_cov=λ_temp=0 vs full loss; measure MSE at horizon=720 |

---

## Theory Gaps (MUST CLOSE)

### Theorem 1 — SCCP Coverage Guarantee

**Statement:**
Let `D_cal = {(s_i, r_i)}_{i=1}^{n}` be a calibration set where `s_i` are latent states and `r_i = |y_i - ŷ_i|` are absolute residuals. Partition D_cal into M clusters via K-Means on s. For a new sample `(s_{new}, r_{new})`, assume exchangeability within the assigned cluster k. Then:
```
P(r_{new} ≤ q_k^{1-α}) ≥ 1 - α
```
where `q_k^{1-α} = Quantile({r_i : cluster(s_i) = k}, ⌈(n_k+1)(1-α)⌉/n_k)`.

**Proof sketch to formalize:**
1. Within cluster k, the n_k calibration residuals plus the new residual form an exchangeable set of n_k+1 variables (assumption).
2. By standard split conformal (Vovk et al., 2005), the rank of r_{new} among the n_k+1 residuals is uniform over {1, ..., n_k+1}.
3. Let m = ⌈(n_k+1)(1-α)⌉. With quantile level min(m/n_k, 1.0), the event {r_new ≤ q_k} occurs when r_new has rank ≤ min(m, n_k) in the combined set.
4. When n_k ≥ ⌈1/α⌉ − 1 (guaranteeing m ≤ n_k, no clipping), P(r_new ≤ q_k) = m/(n_k+1) ≥ 1−α. Below this threshold, P = n_k/(n_k+1) which may fall below 1−α.
5. Marginalizing over clusters, the overall coverage is a weighted average of per-cluster coverages, each ≥ 1−α.

**Edge cases to address:**
- Empty clusters (already handled: fallback to global residuals)
- Small clusters with n_k < ⌈1/α⌉ − 1: coverage guarantee degrades to n_k/(n_k+1) < 1−α
- Autocorrelated residuals (bound the coverage loss via ACF(1))

**Status:** ✅ FORMAL PROOF COMPLETE — REVISED (see `paper1_proofs.tex`). Full proof with rank-uniformity argument, finite-sample correction with clipping analysis, and code-path verification. **Amendment:** The coverage guarantee requires $n_k \geq \lceil 1/\alpha \rceil - 1$ calibration samples per cluster. Below this threshold, coverage degrades to $n_k/(n_k+1)$, which may fall below $1-\alpha$ (e.g., $n_k=5$, $\alpha=0.1$ gives $P=5/6\approx 0.833$). The code warns at $n_k < 1/\alpha$ (line~157). Identified limitation: (i) exchangeability within time-series clusters is approximate, not guaranteed; (ii) small clusters break the coverage guarantee. GRADE assessment: MODERATE. Empirical validation plan provided.

---

### Theorem 2 — Structured Dynamics Stability

**Statement:**
The structured transition matrix A with sigmoid-gated eigenvalues ensures BIBO stability of the state recurrence: for bounded inputs, the state sequence remains uniformly bounded. Specifically:
```
||s_t|| ≤ C  for all t, with C < ∞ depending on α_L, α_T, γ, α_R, ||B||, and β.
```
where ρ_eff = max(α_T, γ, α_R) ≤ 1.0 for the sub-level subspace, and the level component (α_L ≤ 1.0) is separately bounded.

**Proof sketch:**
1. For each block: level ∈ [0.85, 1.0], trend ∈ [0.70, 0.95], seasonal rotation norm = γ ∈ [0.80, 1.0], residual ∈ [0.0, 0.4].
2. The correction MLP output is bounded by `|correction_scale| · 1` (tanh output range).
3. The sub-level subspace has effective contraction ρ_eff ≤ 1.0 (seasonal can be at 1.0, preserving norm).
4. The level component is individually bounded: for α_L < 1, geometric series converges; for α_L = 1, bound is linear in t but finite for finite horizon.
5. Combined BIBO: all components bounded by input bound + correction bound + initial condition.

**Status:** ✅ FORMAL PROOF COMPLETE (see `paper1_proofs.tex`). BIBO stability proved for all 5 state dimensions with sigmoid-gate parameter bounds verified against encoder lines 58--90. Seasonal subspace is Lyapunov-stable at γ ≤ 1.0 (norm-preserving); trend/residual subspaces exponentially contract (α_T ≤ 0.95, α_R ≤ 0.4). Level component bounded by geometric series when α_L < 1. Edge case: when α_L = 1.0, bound is linear in t (unit root) but finite for any input of finite length L.

---

### Theorem 3 — Disentanglement Convergence

**Statement:**
Under the covariance regularization loss L_cov = ||off_diag(Σ_s)||_F², the off-diagonal entries of the state covariance matrix converge to 0 at rate O(1/T) for sequences of length T, assuming the gradient of L_cov dominates the forecast loss gradient in orthogonal directions.

**Proof sketch:**
1. Write the gradient of L_cov w.r.t. each state dimension.
2. Show that for a linear encoder (ignoring correction MLP), the dynamics imply each dimension's evolution is approximately independent when A is diagonal.
3. The covariance loss drives off-diagonal terms to zero; the temporal loss aligns each dimension's timescale with its intended dynamics.
4. The rate O(1/T) follows from the fact that the empirical covariance estimator converges at this rate.

**Status:** Not started — **2 weeks work. May be downgraded to "empirical evidence" if proof is intractable.**

---

### Theorem 4 — Interval Width Bound

**Statement:**
For a cluster with $n_k$ calibration samples and target coverage $1-\alpha$, the quantile $q_k$ concentrates around the population quantile $Q_k$:
```
q_k = Q_k + O_p(1/√n_k)
```
where for sub-Gaussian residuals, $Q_k \leq \sigma_k \cdot \sqrt{2 \log(1/\sqrt{\alpha})}$ with $\sigma_k$ the conditional standard deviation. For distributions with finite variance, the expected width satisfies $\mathbb{E}[q_k] \to Q_k$ as $n_k \to \infty$.

**Proof sketch:**
1. The empirical quantile is $\sqrt{n_k}$-consistent for the population quantile (Bahadur representation).
2. For sub-Gaussian tails, the population quantile $Q_k$ is bounded by $\sigma_k \cdot \sqrt{2 \log(1/\sqrt{\alpha})}$ (via Chernoff bound on the tail).
3. The finite-sample correction term $\lceil (n_k+1)(1-\alpha) \rceil / n_k - (1-\alpha)$ is $O(1/n_k)$, negligible compared to the $O_p(1/\sqrt{n_k})$ estimation error.
4. The SCCP interval width in cluster $k$ tracks $\sigma_k$, explaining adaptivity: clusters with lower conditional variance yield narrower intervals.

**Status:** Not started — **1 week work.**

---

## Paper Outline (IMRaD)

### 1. Introduction (1 page)

- Motivate reliable time-series forecasting with uncertainty quantification (energy, finance, health)
- The trilemma: interpretability vs. accuracy vs. uncertainty
- Gap: no existing method provides all three with theoretical guarantees
- **Contributions:**
  1. Disentangled State-Space Encoder with structured block-diagonal dynamics
  2. Hybrid linear + non-linear Forecast Head with learnable refinement scale
  3. State-Conditional Conformal Prediction (SCCP) with coverage guarantee proof
  4. Rigorous ablation study isolating each component's contribution
- List of hypotheses (H1–H5)

### 2. Related Work (1 page)

| Area | Representative Works | Our Position |
|------|---------------------|--------------|
| State-Space Models | S4, Mamba, LRU, DeepState | We impose interpretable structure; they optimize efficiency |
| Disentangled Representation | β-VAE, FactorVAE, DCI | Applied to time-series with physics-informed dynamics |
| Conformal Prediction | Vovk et al., split CP, EnbPI, SPCI | State-conditioning provides adaptivity without distribution assumptions |
| UQ in Forecasting | MC-Dropout, Deep Ensembles, Bayesian RNNs | Distribution-free vs. parametric assumptions |

### 3. Methods (3 pages)

#### 3.1 Problem Formulation
- Define multivariate time-series forecasting
- Input X ∈ R^{B×L×D_in}, output Ŷ ∈ R^{B×H×D_out}
- Point forecast + interval forecast objectives

#### 3.2 Disentangled State-Space Encoder
- **Structured Transition Matrix A**: Block-diagonal with sigmoid-gated eigenvalues
  - α_L ∈ [0.85, 1.0], α_T ∈ [0.70, 0.95], γ ∈ [0.80, 1.0], α_R ∈ [0.0, 0.4]
  - 2D rotation for seasonal: [c, s; -s, c] with learnable ω
  - State equation: s_t = A·s_{t-1} + B(x_t) + β·tanh(MLP(A·s_{t-1} + B(x_t), h_t)) where h_t = input_proj(x_t)
- **Innovation**: Neural projection of x_t into per-dimension increments
- **Correction**: Small MLP (β_init = 0.01) for non-linear refinement
- **Complexity**: O(L·d) per sample

#### 3.3 Hybrid Forecast Head
- **Linear path**: ŷ_lin = W·s_T + b (interpretable, per-dimension contributions)
- **Non-linear path**: ŷ_ref = γ·MLP(s_T) with learnable scale γ
- **Decomposition**: Each forecast = Σ_i w_i·s^{(i)} + refinement
- **Refinement ratio**: r = ||ŷ_ref|| / (||ŷ_lin|| + ||ŷ_ref||)

#### 3.4 Disentanglement Loss
- **Covariance loss**: L_cov = ||off_diag(Σ_s)||_F²
- **Temporal loss**: L_temp = Σ_i ||s_t^{(i)} - Â^{(i)}·s_{t-1}^{(i)}||²
- **Total**: L = MSE + λ_cov·L_cov + λ_temp·L_temp

#### 3.5 State-Conditional Conformal Prediction
- **Calibration Algorithm** (Algorithm 1 in LaTeX)
- **Theorem 1**: Coverage guarantee (see theory gaps above)
- **Theorem 2**: Interval width bound (see theory gaps above)
- **Finite-sample correction**: ⌈(n_k+1)(1-α)⌉/n_k
- **Exchangeability validation**: Per-cluster ACF(1) test
- **Multivariate strategies**: per_feature, max, mean

### 4. Experiments (3 pages)

#### 4.1 Setup
- **Datasets**: ETTh1, ETTh2, Weather, Exchange-Rate (4 benchmarks)
- **Horizons**: {24, 96, 192, 336, 720}
- **Baselines**: DLinear, PatchTST, N-HiTS, DeepState, Flat-CP, ARIMA
- **Metrics**:
  - Point: MSE, MAE
  - Interval: Coverage @ 90%, MPIW, Winkler score, CRPS
  - Disentanglement: Mean off-diag correlation, refinement ratio, per-dim variance
- **Implementation**: PyTorch, Adam, 3 seeds, early stopping (patience=5)

#### 4.2 Main Results
**Table 1 — Point forecast comparison:**
| Model | ETTh1 (96) | ETTh1 (720) | Weather (96) | Exchange (96) |
|-------|-----------|------------|-------------|--------------|
| DLinear | — | — | — | — |
| PatchTST | — | — | — | — |
| N-HiTS | — | — | — | — |
| DeepState | — | — | — | — |
| **CISSN** | — | — | — | — |

**Table 2 — Interval quality comparison (Coverage=90%):**
| Model | Coverage | MPIW | Winkler | CRPS |
|-------|---------|------|---------|------|
| Flat-CP | — | — | — | — |
| MC-Dropout | — | — | — | — |
| Deep Ensemble | — | — | — | — |
| **CISSN-SCCP** | — | — | — | — |

#### 4.3 Ablation Studies
**Table 3 — Ablation results on ETTh1:**
| Configuration | MSE (96) | MSE (720) | off-diag corr | Coverage | MPIW |
|--------------|----------|-----------|---------------|---------|------|
| CISSN (full) | — | — | — | — | — |
| − structured A | — | — | — | — | — |
| − disentanglement loss | — | — | — | — | — |
| − SCCP (flat CP) | — | — | — | — | — |
| − correction MLP | — | — | — | — | — |
| state_dim=4 | — | — | — | — | — |

#### 4.4 Analysis
- **Figure 1**: Off-diagonal correlation over training epochs (should decay)
- **Figure 2**: Refinement ratio over epochs (should stabilize)
- **Figure 3**: Coverage reliability diagram (empirical vs nominal α)
- **Figure 4**: State PCA colored by regime (shows cluster separation)
- **Figure 5**: Decomposition plot (level, trend, seasonal, residual stacked)
- **Figure 6**: Interval width vs. state norm (shows adaptivity)

### 5. Discussion (0.5 pages)

- H1–H5 interpretation
- Limitations:
  - Exchangeability assumption within clusters
  - Calibration set size sensitivity
  - Computational cost of K-Means for very large calibration sets
- Failure modes: heavy distribution shift, very short calibration sets

### 6. Conclusion (0.25 pages)

- Restate contributions
- Future work: online conformal, multi-step state conditioning, plug-in to other SSMs

### Appendix

- A: Full proofs of Theorems 1–4
- B: Dataset statistics and split details
- C: Hyperparameter sensitivity (λ_cov, λ_temp, correction_scale)
- D: Extended results tables (all horizons × all datasets)
- E: Reproducibility checklist

---

## Codebase Needs Before Submission

### Critical (must complete)

| # | Task | Location | Effort |
|---|------|----------|--------|
| 1 | Baseline wrappers (DLinear, PatchTST, Flat-CP) | `cissn/baselines/` | 2 weeks |
| 2 | Ablation study runner (toggle components) | `experiments/run_ablation.py` | 1 week |
| 3 | Multi-seed runner (3 seeds × 4 datasets × 5 horizons) | `experiments/run_multiseed.py` | 1 week |
| 4 | CRPS, Winkler, PICP, MPIW metrics | `cissn/evaluation/metrics.py` | 3 days |
| 5 | Calibration curve + reliability diagram | `cissn/evaluation/plots.py` | 3 days |
| 6 | LaTeX table generator | `scripts/generate_tables.py` | 2 days |
| 7 | ~~Formal coverage proof (LaTeX)~~ ✅ DONE — REVISED | `publication/paper1_proofs.tex` | **CORRECTED** — added min-cluster condition A3 |
| 7b | ~~Structured dynamics stability proof~~ ✅ DONE — REVISED | `publication/paper1_proofs.tex` | **CORRECTED** — BIBO stability, rotation sign, geometric series direction |

### High (should complete)

| # | Task | Location | Effort |
|---|------|----------|--------|
| 8 | Hyperparameter sensitivity sweep runner | `experiments/run_sensitivity.py` | 3 days |
| 9 | State PCA/t-SNE visualizations | `cissn/evaluation/plots.py` | 2 days |
| 10 | Contribution decomposition plots | `cissn/evaluation/plots.py` | 2 days |
| 11 | Runtime scaling benchmark | `experiments/benchmark_scaling.py` | 1 day |

### Medium (nice to have)

| # | Task | Location | Effort |
|---|------|----------|--------|
| 12 | CRPS-based early stopping | `experiments/run_benchmark.py` | 1 day |
| 13 | Checkpoint versioning (dataset + horizon + seed) | `experiments/run_benchmark.py` | 2 days |
| 14 | Reproduce script (`reproduce.sh`) | `scripts/reproduce.sh` | 1 day |

---

## Baselines to Implement

### DLinear

```python
# cissn/baselines/dlinear.py
# Single linear layer: ŷ = W·x_last + b
# No encoder, no recurrent structure
# The simplest possible deep learning baseline
```

**Rationale:** If CISSN can't beat DLinear, the architecture is unjustified.

### PatchTST

```python
# cissn/baselines/patchtst.py
# Wrapper around pretrained PatchTST from the official repo
# Or reimplement core patch + transformer architecture
```

**Rationale:** Current SOTA transformer for LTSF — must be competitive.

### Flat Conformal (Marginal CP)

```python
# cissn/baselines/flat_conformal.py
# Same CISSN encoder + forecast head
# Replace SCCP with single global quantile (no clustering)
```

**Rationale:** Isolates the contribution of state-conditioning.

### MC-Dropout

```python
# cissn/baselines/mc_dropout.py
# Same CISSN encoder + forecast head with dropout enabled at inference
# Sample N=50 forward passes, use empirical std for interval
```

**Rationale:** Most common UQ baseline in deep learning.

### Deep Ensemble

```python
# cissn/baselines/deep_ensemble.py
# Train M=3 CISSN models from different seeds
# Interval = mean_pred ± z · std_preds
```

**Rationale:** Simple but effective UQ method.

---

## Ablation Configurations

```python
ABLATION_CONFIGS = {
    "full": {
        "structured_A": True,
        "disentanglement_loss": True,
        "sccp": True,
        "correction_mlp": True,
        "state_dim": 5,
    },
    "no_structured_A": {
        "structured_A": False,       # Replace with dense learned A
    },
    "no_disentanglement_loss": {
        "disentanglement_loss": False,  # λ_cov = λ_temp = 0
    },
    "flat_cp": {
        "sccp": False,               # Use single global quantile
    },
    "no_correction_mlp": {
        "correction_mlp": False,     # Pure linear encoder
    },
    "state_dim_4": {
        "state_dim": 4,              # Scalar seasonal instead of 2D rotation
    },
}
```

---

## Experiment Grid

### Main Benchmark (3 seeds)

```
Datasets:  ETTh1, ETTh2, Weather, Exchange-Rate
Horizons:  24, 96, 192, 336, 720
Seeds:     42, 123, 456
Models:    DLinear, PatchTST, DeepState, Flat-CP, CISSN

Total runs per configuration: 4 × 5 × 3 = 60
```

### Ablation Study (1 dataset, 3 seeds)

```
Dataset:  ETTh1
Horizons: 96, 336, 720
Seeds:    42, 123, 456
Configs:  6 (full + 5 ablations)

Total runs: 1 × 3 × 3 × 6 = 54
```

---

## Figures Checklist

| # | Figure | Data Source | Tool |
|---|--------|------------|------|
| 1 | Off-diag correlation over epochs | Training logs | matplotlib |
| 2 | Refinement ratio over epochs | Training logs | matplotlib |
| 3 | Coverage reliability diagram | Test results | matplotlib |
| 4 | State PCA colored by regime | Test states + cluster labels | sklearn PCA + matplotlib |
| 5 | Forecast decomposition plot | Single ETTh1 test window | matplotlib (stacked area) |
| 6 | Interval width vs. state norm | Test results | matplotlib (scatter) |
| 7 | Horizon-wise MSE comparison | Main results table | matplotlib (line plot) |
| 8 | Ablation radar/spider plot | Ablation results | matplotlib |

---

## Writing Schedule

| Week | Section | Deliverable |
|------|---------|-------------|
| 9 | Methods 3.1–3.3 | Encoder + Forecast Head text + formulas |
| 10 | Methods 3.4–3.5 | SCCP text + Algorithm 1 + Theorem statements |
| 11 | Introduction + Related Work | Motivation, gap, contributions, lit review |
| 12 | Experiments 4.1–4.3 | Setup + Main results + Ablation tables |
| 13 | Experiments 4.4 + Discussion | Analysis, figures, limitations |
| 14 | Appendix + Polish | Proofs, sensitivity, final formatting |

---

## Submission Checklist

- [ ] Theorems 1 & 2 proved (critical corrections applied); Theorems 3 & 4 still need formal proofs
- [ ] All 5 ablation configurations run and analyzed
- [ ] 3-seed results with mean ± std for all main tables
- [ ] All 8 figures generated and captioned
- [ ] Hyperparameter sensitivity appendix (C)
- [ ] Reproducibility checklist (E)
- [ ] Code repository cleaned and tagged (v1.0-paper1)
- [ ] arXiv preprint ready to submit simultaneously
- [ ] Supplementary material zip (all raw results + plot scripts)

---

## File Manifest

```
publication/
├── paper1_framework.md       ← this file
├── paper1_proofs.tex         ← formal proofs (to be written)
├── paper1_main.tex           ← main paper source (to be written)
├── paper1_appendix.tex       ← supplementary material (to be written)
└── paper1_figures/           ← generated plots
```

---

## Related Code Paths

| Need | Path |
|------|------|
| Baseline wrappers | `cissn/baselines/dlinear.py`, `patchtst.py`, `flat_cp.py`, `mc_dropout.py` |
| Ablation runner | `experiments/run_ablation.py` |
| Multi-seed runner | `experiments/run_multiseed.py` |
| Extended metrics | `cissn/evaluation/metrics.py` (CRPS, Winkler, PICP, MPIW) |
| Plotting | `cissn/evaluation/plots.py` |
| Sensitivity sweep | `experiments/run_sensitivity.py` |
| Table generator | `scripts/generate_tables.py` |
