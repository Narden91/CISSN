# CISSN Publication Strategy

## 1. Scientific Positioning (The "Hook")

**Problem:**
Time-series forecasting faces a trilemma:
1.  **Interpretability** (SSMs, ARIMA): Good for trust, bad for complex patterns.
2.  **Performance** (Transformers, LSTMs): Great metrics, black-box nature.
3.  **Uncertainty**: Often disregarded or assumed Gaussian (which is false for real-world data).

**Solution (CISSN):**
**CISSN** (Conformally Calibrated Interpretable State-Space Networks) solves this by enforcing *structural interpretability* (Level, Trend, Seasonal) via a physics-based transition matrix, while using deep learning for the *residuals* and *innovations*. Crucially, it uses **State-Conditional Conformal Prediction** to guarantee valid uncertainty intervals for **non-Gaussian, heteroscedastic** data.

**Novelty Claim:**
> "We propose CISSN, a framework that unifies disentangled representation learning with conformal prediction. Unlike standard hybrids (e.g., DeepState), CISSN explicitly models 2D seasonality via rotation matrices and uses the latent state to adaptively calibrate uncertainty intervals, providing distinct 'regimes' of volatility."

---

## 2. Experimental Rigor (The "Proof")

To publish in a top-tier venue (NeurIPS, ICML, ICLR, or specialized TS workshops), we must demonstrate rigorous validation.

### A. Baselines
You must compare against:
1.  **Statistical (Sanity Check)**:
    -   *Naïve Seasonal*: Repeat the last season. (Critical baseline).
    -   *ETS / ARIMA*: Standard stats models.
2.  **Deep Learning (Performance)**:
    -   *N-BEATS / N-HiTS*: Pure DL SOTA.
    -   *PatchTST*: Transformer SOTA.
3.  **Hybrid (Direct Competitor)**:
    -   *DeepState* (Amazon): The closest relative. We must beat or match it while offering better *uncertainty* or *interpretability*.

### B. Ablation Studies (Proving Components)
1.  **Impact of 2D Seasonality**:
    -   Compare `CISSN (2D Rotation)` vs `CISSN (1D Scalar)`.
    -   *Expected Result*: 2D captures oscillating waves; 1D fails to model complex periodicity.
2.  **Impact of Conformal Prediction**:
    -   Compare `CISSN + SCCP` vs `CISSN + Gaussian Likelihood`.
    -   *Metric*: Coverage Error (Target 90%, Actual ?). Gaussian likely under/over-confident.
3.  **Impact of Disentanglement**:
    -   Show that removing the $\mathbf{A}$ structure (making it fully learnable/unconstrained) hurts long-term forecasting or interpretability.

---

## 3. Validation Strategy

### Key Metrics
-   **Point Accuracy**: MSE, MAE.
-   **Probabilistic Accuracy**: CRPS (Continuous Ranked Probability Score).
-   **Calibration**: Coverage at 90% (ACE), Interval Width (MPIW). *Narrower is better, but only if Coverage $\ge$ 90%.*

### Critical Bias Checks
1.  **Look-ahead Bias**: Ensure the Scaler (e.g., StandardScaler) is fit *only* on the training set, not the whole dataset.
2.  **Test Set Leakage**: Ensure no overlap between train/val/test windows.
3.  **Rolling Origin Evaluation**: Do not test on just one slice. Use a rolling window (Cross-Validation) to test on multiple "futures".

---

## 4. Visual Evidence (Qualitative Analysis)

Paper figures are crucial. We need:

1.  **Decomposition Plot**:
    -   Show detailed breakdown: $y = \text{Level} + \text{Trend} + \text{Seasonal} + \text{Residual}$.
    -   *Example*: Show "Level" staying stable while "Seasonal" oscillates. This proves disentanglement works.
2.  **Uncertainty Regimes**:
    -   Show a plot where the interval widens during a "volatile" period (high residual state) and narrows during a "stable" period.
    -   *Caption*: "State-Conditional Conformal Prediction adapts to changing volatility."

---

## 5. Roadmap to Submission

1.  **Code Check**: Verify `encoder.py` implements the exact 2D rotation matrix derived in theory.
2.  **Benchmark Setup**: Create `benchmark_runner.py` to run CISSN vs DeepState on standard datasets (Electricity, Traffic, Weather).
3.  **Drafting**: Use the "Method -> Experiments -> Analysis" structure.
4.  **Pre-print**: Publish on arXiv once results are stable.
