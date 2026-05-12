# CISSN: Intuition and How It Works

This note was previously the root `document.md`. It now lives under `docs/` because the root `document.md` is the canonical Q1 Journal 1 experiment plan.

## What Problem Does CISSN Solve?

Time-series forecasting has a trilemma:

1. Interpretability: classical models are explainable but often too rigid.
2. Accuracy: deep models predict well but are often black boxes.
3. Uncertainty: probabilistic methods help decisions but often rely on brittle distributional assumptions.

CISSN combines structured state-space dynamics, a hybrid forecast head, and conformal prediction so the model can target all three at once.

## The Core Intuition

### 1. Time series decompose into a small set of recurring components

Most real sequences can be described in terms of:

- a baseline level,
- a trend,
- a seasonal pattern,
- and residual noise.

CISSN encodes these components explicitly in a 5-dimensional latent state. Seasonality uses two dimensions because periodic motion is most naturally modeled as a 2D rotation.

### 2. The latent state follows structured dynamics

Instead of letting a neural network freely rewrite the state at every step, CISSN constrains the transition:

```text
new_state = A * old_state + innovation_from_input + small_neural_correction
```

The structure of `A` keeps the state physically meaningful:

- level changes slowly,
- trend persists smoothly,
- seasonal dimensions rotate,
- residual noise decays quickly.

The neural correction is intentionally small, so the structured dynamics dominate.

### 3. Interpretability comes from the linear forecast path

The forecast head has two parts:

- a linear path that maps each state dimension to the forecast directly,
- a small nonlinear refinement path.

This makes it possible to attribute a forecast to level, trend, seasonality, and residual contributions while still allowing modest nonlinear corrections.

### 4. Uncertainty adapts to the current regime

Standard conformal prediction uses one global residual quantile. CISSN instead conditions interval width on the latent state:

1. encode calibration windows,
2. cluster the resulting states,
3. compute a residual quantile inside each cluster,
4. use the cluster assigned to a new forecast to pick its interval width.

This yields narrow intervals in stable regimes and wider intervals in volatile regimes.

### 5. Disentanglement is trained explicitly

The loss includes terms that:

- penalize cross-correlation between state dimensions,
- encourage each dimension to follow the structured dynamics it is meant to represent.

This makes the latent state easier to interpret and more stable across horizons.

## End-To-End Flow

```text
Input sequence
  -> structured encoder
  -> final 5D state
  -> hybrid forecast head
  -> point forecast
  -> state-conditional conformal calibration
  -> prediction interval
```

## Why It Matters

- You can inspect why a forecast moved up or down.
- You can get intervals without committing to a Gaussian residual model.
- You can adapt uncertainty to regimes instead of using one fixed interval width.

## Where To Read Next

- `document.md` for the executable Q1 Journal 1 experiment plan.
- `architecture/architecture_and_flow.md` for the architectural details.
- `docs/cissn_technical_specification.md` for the formal technical specification.
- `publication/paper1_framework.md` for the theory-facing Paper 1 framework.