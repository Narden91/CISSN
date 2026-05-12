# CISSN Manuscript

**Working Title:** Conformal Inference via State-Conditional Disentangled State-Space Networks for Time-Series Forecasting

**Target Venue:** First Q1 journal submission

**Submission Mode:** Rolling journal schedule

**Experiment Source Of Truth:** `../document.md`

---

## Research Hypotheses

1. **H1 – Disentanglement improves coverage.** State-conditional conformal intervals are narrower and better-calibrated when the encoder imposes disentangled latent dynamics than when it does not.
2. **H2 – State conditioning beats marginal conformal.** Conditioning on the latent state at prediction time yields tighter intervals than a flat conformal baseline at equal empirical coverage.
3. **H3 – Structured dynamics remain competitive at long horizons.** CISSN preserves calibration and point accuracy on multivariate long-horizon benchmarks while keeping the latent state interpretable.

---

## Paper Structure

### 1. Introduction

- Motivate reliable uncertainty quantification in time-series forecasting.
- Frame the trilemma: interpretability, accuracy, and uncertainty.
- State the paper contributions clearly and conservatively.

### 2. Related Work

- State-space models for sequence modeling.
- Disentangled representation learning.
- Conformal prediction and adaptive conformal methods.
- Uncertainty baselines in forecasting.

### 3. Methods

- Problem formulation.
- Disentangled state-space encoder.
- Hybrid forecast head.
- State-conditional conformal prediction.
- Training procedure and artifact protocol.

### 4. Experiments

- Consume the executable scope and commands from `../document.md`.
- Report only results backed by saved JSON, CSV, and raw array artifacts.
- Separate main results, UQ comparison, ablations, and analysis figures.

### 5. Discussion

- Interpret strengths, limits, and failure modes carefully.
- Keep exchangeability and calibration assumptions explicit.

### 6. Conclusion

- Summarize the method, the empirical evidence, and the practical implications.

### Appendix

- Proofs.
- Dataset statistics.
- Extended result tables.
- Reproducibility checklist.

---

## Writing Gates

- Methods, related work, and theory drafting can proceed now.
- Results writing starts only after the main CISSN grid, baseline grid, UQ subset, and ablation artifacts exist.
- Claims tied to Theorems 3 and 4 remain empirical unless formal proofs are added later.

## Non-Goals For This File

- Do not duplicate the runnable experiment loops from `../document.md`.
- Do not duplicate proof details from `../publication/paper1_framework.md`.
- Do not maintain a second benchmark dataset checklist here.