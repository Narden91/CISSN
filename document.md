# CISSN: Intuition and How It Works

## What Problem Does CISSN Solve?

Time-series forecasting has a **trilemma**:

1. **Interpretability** (ARIMA, ETS) → You can explain why a forecast was made, but the models are linear and miss complex patterns.
2. **Accuracy** (Transformers, LSTMs) → Great predictions, but they are black boxes. You cannot decompose a forecast into understandable components.
3. **Uncertainty** (Bayesian methods, ensembles) → Needed for decision-making, but typically assume Gaussian distributions which are wrong for real-world data (heavy tails, regime changes, heteroscedasticity).

**CISSN solves all three simultaneously** by combining the structural discipline of state-space models with the flexibility of neural networks and the distribution-free guarantees of conformal prediction.

## The Core Intuition

### 1. Every time series is built from a few fundamental components

Think about any time series — stock prices, temperature, electricity consumption. It always has:

- A **baseline level** (the average value around which everything moves)
- A **trend** (is it going up or down over time?)
- **Seasonal patterns** (daily, weekly, yearly cycles)
- **Random noise** (unpredictable short-term fluctuations)

CISSN encodes these four components explicitly into a 5-dimensional latent state vector. The seasonal component uses 2 dimensions because it models periodicity as a rotation in a 2D plane (like a clock hand), which naturally captures sine/cosine waves.

### 2. The state should evolve according to physics-like rules

Instead of letting a neural network arbitrarily transform the state at each timestep, CISSN imposes a **structured transition matrix**:

```
New State = A · Old State + Innovation_from_input + Small_Neural_Correction
```

The matrix A is constrained so that:
- The **level** barely changes (α ≈ 0.85–1.0, near unit root)
- The **trend** persists smoothly (α ≈ 0.70–0.95)
- The **seasonal** components rotate at a learned frequency (2D rotation matrix)
- The **residual** decays quickly (α ≈ 0.0–0.4, like noise should)

The "small neural correction" is a tiny MLP (initially scaled down by 100×) that can fix errors in the linear prediction, but the physics rules always dominate. This means even without training, the state dynamics make physical sense.

### 3. Interpretability comes from the linear path

The forecast head has two paths:
- A **linear path**: Each state dimension is multiplied by a weight and summed. This is fully transparent — you can say "the forecast is +2.3 because the trend component contributed +1.8 and the seasonal contributed +0.5."
- A **non-linear path**: A small MLP refines the prediction. The ratio of non-linear to linear contribution is monitored during training.

The `ForecastExplainer` decomposes any prediction into Level, Trend, Seasonal, and Residual contributions.

### 4. Uncertainty should adapt to the current regime

Traditional conformal prediction produces one fixed-width interval for all predictions. But some situations are inherently more uncertain than others — a calm market period vs. a crash, normal weather vs. a storm.

CISSN solves this with **State-Conditional Conformal Prediction (SCCP)**:

1. Encode past data through the encoder to get a latent state
2. Cluster all calibration states using K-Means (each cluster = a "regime")
3. For each cluster, compute how wrong the model was on average — this becomes the interval width for that regime
4. At prediction time: encode the new input, find which regime (cluster) it belongs to, and use that regime's interval width

The result: narrow intervals in predictable regimes, wide intervals in volatile regimes — all with a mathematical guarantee that at least (1-α)% of true values will fall inside.

### 5. Disentanglement is enforced, not just hoped for

A separate disentanglement loss encourages the state dimensions to be independent:

- **Covariance loss**: Penalizes correlation between different state dimensions (they should move independently)
- **Temporal loss**: Penalizes deviation from the expected structured dynamics (level should be smooth, residual should be noisy)

During training, the off-diagonal correlation between state dimensions is logged — it should approach zero as the model learns to separate the components.

## How It All Fits Together

```
Input Sequence (e.g., 96 hours of temperature data)
        │
        ▼
┌──────────────────────────────────────┐
│  Disentangled State Encoder          │
│  ┌────────────────────────────────┐  │
│  │ For each timestep:              │  │
│  │   State = A · PrevState         │  │  ← Structured physics
│  │         + Innovation(Input)     │  │  ← Neural input mapping
│  │         + 0.01 · tanh(MLP(...)) │  │  ← Tiny correction
│  └────────────────────────────────┘  │
│  Output: 5-dim state [L, T, S₀, S₁, R]│
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  Forecast Head                       │
│  ┌────────────────────────────────┐  │
│  │ ŷ = W·s + b                    │  │  ← Interpretable linear path
│  │   + γ · MLP(s)                 │  │  ← Learnable refinement
│  └────────────────────────────────┘  │
│  Output: Multi-horizon forecast      │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  State-Conditional Conformal (SCCP)  │
│  ┌────────────────────────────────┐  │
│  │ 1. KMeans(s) → cluster k       │  │
│  │ 2. Look up quantile qₖ         │  │
│  │ 3. Interval = [ŷ - qₖ, ŷ + qₖ] │  │
│  └────────────────────────────────┘  │
│  Output: Prediction intervals        │
└──────────────────────────────────────┘
```

## Why This Matters

- **Trust**: When CISSN predicts a spike in electricity demand, you can see it's because the seasonal component (time of day) and trend component (growing baseline) both point upward.
- **Safety**: In a medical setting, wider intervals during unstable patient states let clinicians know when to be cautious.
- **Robustness**: The conformal guarantee means you can state "90% of our predictions will contain the true value" and back it up mathematically — no distribution assumptions needed.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 5 state dimensions (not 4) | Seasonal needs 2 dimensions for proper 2D rotation; 1 dimension would force a scalar oscillation that cannot model phase shifts |
| Sigmoid-gated transition parameters | Keeps eigenvalues in physically meaningful ranges without manual tuning |
| Small correction MLP (init scale 0.01) | Ensures the linear physics dominates; non-linearity is a refinement, not the main engine |
| K-Means for state clustering | Simple, fast, interpretable; clusters map directly to human-understandable "regimes" |
| Per-cluster quantile correction ⌈(n+1)(1-α)⌉/n | Standard conformal finite-sample correction that guarantees coverage without distribution assumptions |
| Learnable refinement scale | Model can learn to reduce non-linear contribution if the linear path is sufficient, preserving interpretability |

## Further Reading

- `architecture/architecture_and_flow.md` — Detailed math and data flow
- `docs/cissn_technical_specification.md` — Full theoretical specification with pseudocode
- `manuscript/README.md` — Paper outline, research hypotheses, and experiment plan
