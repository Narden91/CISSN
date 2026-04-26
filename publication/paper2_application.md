# Paper 2 — CISSN in Practice: Multi-Domain Interpretable Forecasting with Calibrated Uncertainty

## Metadata

| Field | Value |
|-------|-------|
| **Working Title** | CISSN in Practice: Interpretable and Calibrated Time-Series Forecasting Across Four Real-World Domains |
| **Target Venues** | Engineering Applications of AI (primary) → Expert Systems with Applications → Applied Soft Computing |
| **Deadline** | Rolling submission (no fixed deadline for these journals) |
| **Focus** | Multi-domain application, case studies, practical insights from disentangled states |
| **Target Length** | 12–15 pages journal format |
| **Status** | Planned — domains selected, codebase ready for multi-dataset runs |
| **Prerequisite** | Paper 1 must be submitted (not necessarily accepted) to establish framework |

---

## Research Questions

| # | Question | Domain Test |
|---|---------|-------------|
| RQ1 | Does CISSN's disentangled state decomposition reveal domain-meaningful patterns across different forecasting domains? | All 4 domains |
| RQ2 | How does the adaptive interval width from SCCP translate to operational value in each domain? | Qualitative case studies |
| RQ3 | Does training CISSN on one domain transfer useful representations to another domain? | Cross-domain transfer |
| RQ4 | What is the trade-off between calibration quality and computational cost across domains of different scales? | Runtime + coverage analysis |
| RQ5 | How robust is CISSN to distribution shifts within each domain? | Train/test gap analysis |

---

## Domain Coverage

### Domain 1: Energy — Electricity Load Forecasting

| Aspect | Details |
|--------|---------|
| **Dataset** | ECL (Electricity Consuming Load) — 321 clients, hourly, 26,304 timesteps |
| **Task** | Multivariate → Multivariate: forecast load for all 321 clients |
| **Horizons** | 96, 192, 336, 720 |
| **Why important** | Grid operators need calibrated load forecasts for reliability planning; over-estimation wastes resources, under-estimation causes blackouts |
| **What CISSN reveals** | Seasonal component captures daily + weekly cycles; trend tracks long-term demand growth; residual captures weather-driven spikes |
| **Key visualization** | Interval width widening during demand peaks (hot summer afternoons) |

### Domain 2: Finance — Exchange Rate Prediction

| Aspect | Details |
|--------|---------|
| **Dataset** | Exchange-Rate — 8 currencies, daily, 7,588 timesteps |
| **Task** | Multivariate → Multivariate: forecast 8 FX rates simultaneously |
| **Horizons** | 96, 192, 336, 720 |
| **Why important** | Currency risk management requires calibrated prediction intervals; near-random-walk behavior tests CISSN's ability to produce meaningful uncertainty |
| **What CISSN reveals** | Level component dominates (near-random-walk); trend captures slow macro shifts; seasonal is expected to be weak — a domain-specific insight |
| **Key visualization** | Narrow intervals during stable periods, rapid widening during volatility events |

### Domain 3: Climate — Weather Forecasting

| Aspect | Details |
|--------|---------|
| **Dataset** | Weather — 21 meteorological indicators, 10-min, 52,696 timesteps |
| **Task** | Multivariate → Multivariate: forecast 21 weather variables |
| **Horizons** | 96, 192, 336, 720 |
| **Why important** | Weather forecasting has strong physical seasonality (diurnal, annual); CISSN's explicit 2D rotation should capture this naturally |
| **What CISSN reveals** | Seasonal cos/sin pair captures diurnal cycle across all variables; level component separates long-term climate from short-term weather |
| **Key visualization** | Decomposition plot showing clean separation of day/night cycle from weather fronts |

### Domain 4: Health — Influenza Surveillance

| Aspect | Details |
|--------|---------|
| **Dataset** | ILI (Influenza-Like Illness) — CDC weekly data, 7 features, 966 timesteps |
| **Task** | Multivariate → Multivariate: forecast ILI rates with horizon 24–60 weeks |
| **Horizons** | 24, 36, 48, 60 |
| **Why important** | Short series (only ~20 years) with long horizon; public health decisions depend on calibrated uncertainty — overconfident forecasts cause misallocation |
| **What CISSN reveals** | Seasonal component captures annual flu season cycle; trend tracks multi-year epidemiological shifts; residual captures outbreak anomalies |
| **Key visualization** | Interval calibration during a flu season peak — SCCP widens appropriately |

---

## Paper Outline

### 1. Introduction (2 pages)

- Real-world forecasting requires more than point accuracy
- Three stakeholder needs: accuracy (operations), interpretability (trust), calibrated uncertainty (risk management)
- Most published methods focus on accuracy alone
- CISSN (introduced in Paper 1) addresses all three
- **This paper**: Apply CISSN across four diverse domains, extract domain insights from disentangled states, demonstrate practical value of adaptive uncertainty

### 2. Background: CISSN Recap (1.5 pages)

- Brief summary of the CISSN architecture (citing Paper 1):
  - Disentangled State Encoder (5D state, structured A, small correction MLP)
  - Hybrid Forecast Head (linear + refinement)
  - SCCP for adaptive prediction intervals
- Key metrics: point (MSE, MAE), interval (Coverage, MPIW, Winkler, CRPS), disentanglement (off-diag correlation, refinement ratio)

### 3. Domain Studies (6 pages)

#### 3.1 Energy: Electricity Load Forecasting
- Dataset description and preprocessing
- Experimental setup
- Results table (MSE, MAE, Coverage, MPIW)
- **Insight 1**: Seasonal decomposition analysis — what drives the daily/weekly cycle?
- **Insight 2**: Interval behavior during peak demand hours
- **Case study**: A specific week showing forecast + intervals + decomposition

#### 3.2 Finance: Exchange Rate Prediction
- Dataset description
- Results table
- **Insight 1**: Why is seasonal weak in FX? (domain knowledge confirmation)
- **Insight 2**: Interval adaptation during volatility events (e.g., Brexit week)
- **Case study**: Narrow intervals during calm, widening during shocks

#### 3.3 Climate: Weather Forecasting
- Dataset description
- Results table
- **Insight 1**: 2D seasonal rotation naturally captures diurnal cycle across 21 variables
- **Insight 2**: Decomposition separates climate signal from weather noise
- **Case study**: 7-day forecast decomposition

#### 3.4 Health: Influenza Surveillance
- Dataset description
- Results table
- **Insight 1**: Handling short series — calibration set size sensitivity
- **Insight 2**: Interval calibration during flu season onset
- **Case study**: Retrospective analysis of a flu season

### 4. Cross-Domain Analysis (2 pages)

#### 4.1 Disentanglement Quality Across Domains
- **Table**: Off-diag correlation, refinement ratio, per-dim variance for each domain
- **Finding**: Weather shows strongest seasonal signal (as expected); Exchange shows weakest

#### 4.2 Calibration Quality Across Domains
- **Table**: Coverage vs nominal, MPIW, Winkler score for each domain
- **Finding**: SCCP maintains coverage across all domains; interval width adapts to domain volatility

#### 4.3 Cross-Domain Transfer
- **Experiment**: Train on ETTh1, test on Weather/ECL/Exchange
- **Table**: Transfer MSE vs. native CISSN vs. baselines
- **Finding**: Encoder transfers reasonably across energy/climate domains; finance requires domain-specific training

#### 4.4 Computational Cost
- **Table**: Training time, inference time, calibration time per domain
- **Finding**: SCCP adds <5% overhead at inference

### 5. Discussion (1.5 pages)

- RQ1–RQ5 interpretation
- **What we learned that Paper 1 didn't show**:
  - Domain-specific decomposition patterns
  - Practical value of adaptive intervals in operational settings
  - Cross-domain generalization of disentangled representations
  - Scalability to large multivariate datasets (321 features for ECL)
- **Limitations specific to applications**:
  - ILI: very short series limits calibration set splitting
  - ECL: 321 features pushes memory; batching strategies needed
  - Exchange: near-random-walk limits forecast improvement over naive
- **Practitioner guidance**: When to use CISSN vs. alternatives

### 6. Conclusion (0.5 pages)

- Summary of domain findings
- Practical recommendations for deploying CISSN
- Future work: streaming/online deployment, domain-specific fine-tuning recipes

### Appendix

- A: Full results tables per domain (all horizons × all metrics)
- B: Additional case studies (4 more weeks per domain)
- C: Cross-domain transfer full results
- D: Hyperparameter configurations per domain

---

## Implementation TODOs

### Critical

| # | Task | Location | Effort |
|---|------|----------|--------|
| 1 | Run CISSN on all 4 domains × 5 horizons × 3 seeds | Experiment runner | 1 week compute |
| 2 | Cross-domain transfer experiments | `experiments/run_transfer.py` | 3 days |
| 3 | Domain-specific case study notebooks | `publication/paper2_notebooks/` | 1 week |
| 4 | Extract and visualize per-domain decomposition patterns | `cissn/evaluation/plots.py` | 3 days |

### High

| # | Task | Location | Effort |
|---|------|----------|--------|
| 5 | Computational cost benchmarking per domain | `experiments/benchmark_scaling.py` | 2 days |
| 6 | Calibration set size sensitivity per domain | `experiments/run_sensitivity.py` | 2 days |
| 7 | Domain practitioner interview / expert validation | Qualitative | 1 week |

### Medium

| # | Task | Location | Effort |
|---|------|----------|--------|
| 8 | Baseline comparison per domain (domain-specific SOTA) | `cissn/baselines/` | 1 week |
| 9 | Cross-domain encoder visualization (PCA of all domains) | `cissn/evaluation/plots.py` | 1 day |

---

## Domain-Specific Baselines

| Domain | Baseline | Rationale |
|--------|----------|-----------|
| Energy | N-BEATS, DeepAR | Standard energy forecasting baselines |
| Finance | GARCH, ARIMA-GARCH | Volatility modeling is standard |
| Climate | ConvLSTM, MetNet | Spatio-temporal weather models |
| Health | ETS, Prophet | Classical epidemiological forecasting |

---

## Figures Checklist

| # | Figure | Content |
|---|--------|---------|
| 1 | Domain comparison radar plot | MSE, Coverage, MPIW across 4 domains |
| 2 | Per-domain decomposition | 2×2 grid of stacked area plots |
| 3 | Interval behavior during events | 4 panels: peak demand, volatility event, storm, flu season |
| 4 | Cross-domain transfer heatmap | 4×4 matrix: train→test MSE |
| 5 | Calibration quality across domains | 4 reliability diagrams |
| 6 | Computational cost scaling | Runtime vs. n_features for ECL + Traffic |
| 7 | Seasonal component analysis | Rotation frequency ω learned per domain |
| 8 | Refinement ratio per domain | Bar chart: how much does non-linear path contribute per domain |

---

## Results Tables (Expected)

### Table 1 — Point forecast per domain

| Domain | CISSN MSE | Domain SOTA MSE | Improvement |
|--------|-----------|-----------------|-------------|
| Energy (ECL) | — | — | — |
| Finance (Exchange) | — | — | — |
| Climate (Weather) | — | — | — |
| Health (ILI) | — | — | — |

### Table 2 — Interval quality per domain

| Domain | Coverage | MPIW | Winkler | CRPS |
|--------|---------|------|---------|------|
| Energy | — | — | — | — |
| Finance | — | — | — | — |
| Climate | — | — | — | — |
| Health | — | — | — | — |

### Table 3 — Disentanglement per domain

| Domain | off-diag corr | Refinement ratio | Seasonal ω | Level α_L |
|--------|--------------|-----------------|------------|-----------|
| Energy | — | — | — | — |
| Finance | — | — | — | — |
| Climate | — | — | — | — |
| Health | — | — | — | — |

### Table 4 — Cross-domain transfer

| Train → Test | ETTh1 | ECL | Weather | Exchange |
|-------------|-------|-----|---------|----------|
| ETTh1 | — | — | — | — |
| ECL | — | — | — | — |
| Weather | — | — | — | — |
| Exchange | — | — | — | — |

---

## Writing Schedule

| Week | Section | Deliverable |
|------|---------|-------------|
| 17 | Energy domain study | Section 3.1 + figures |
| 18 | Finance + Climate domain studies | Sections 3.2–3.3 + figures |
| 19 | Health domain study + Cross-domain analysis | Sections 3.4 + 4 |
| 20 | Introduction + Background + Discussion | Sections 1, 2, 5, 6 |
| 21 | Polish + Internal review | Full draft |
| 22 | Submit | Final manuscript |

---

## Submission Checklist

- [ ] All 4 domain experiments complete (3 seeds × 5 horizons)
- [ ] Cross-domain transfer results
- [ ] All 8 figures generated
- [ ] Domain-specific baselines run and compared
- [ ] Domain expert validation (qualitative feedback on decomposition insights)
- [ ] Code repository tagged (v1.1-paper2)
- [ ] Supplementary material (full results tables, additional case studies)

---

## File Manifest

```
publication/
├── paper2_application.md      ← this file
├── paper2_main.tex            ← main paper source (to be written)
├── paper2_appendix.tex        ← supplementary material (to be written)
├── paper2_figures/            ← generated plots
└── paper2_notebooks/          ← case study Jupyter notebooks
    ├── energy_case_study.ipynb
    ├── finance_case_study.ipynb
    ├── climate_case_study.ipynb
    └── health_case_study.ipynb
```

---

## Relationship to Paper 1

| Aspect | Paper 1 | Paper 2 |
|--------|---------|---------|
| **Contribution** | Methodological novelty | Practical validation |
| **Datasets** | 4 curated benchmarks (ETT, Weather, Exchange) | 4 real-world domains |
| **Baselines** | DL competitors (PatchTST, N-HiTS) | Domain-specific SOTA |
| **Novelty** | Structured dynamics theory, SCCP proof | Cross-domain insights, operational value |
| **Venue** | ML/AI conference | Applied AI journal |
| **Audience** | ML researchers | Practitioners + domain experts |
| **Timeline** | Submit Month 4 | Submit Month 6–7 |

**Citation dependency**: Paper 2 cites Paper 1 as the methodological foundation. Paper 1 should be on arXiv (even if under review) before Paper 2 submission.

---

## Related Code Paths

| Need | Path |
|------|------|
| Multi-domain runner | `experiments/run_multidomain.py` |
| Cross-domain transfer | `experiments/run_transfer.py` |
| Domain analysis notebooks | `publication/paper2_notebooks/` |
| Domain baseline wrappers | `cissn/baselines/domain_specific/` |
| Decomposition plots | `cissn/evaluation/plots.py` |
