# CISSN Architecture and Data Flow

## 1. System Overview

**CISSN** (Conformally Calibrated Interpretable State-Space Networks) is a hybrid architecture designed to bridge the gap between physically interpretable **State-Space Models (SSMs)** and high-capacity **Deep Neural Networks**.

It decomposes time-series data into clinically/physically meaningful components—Level, Trend, Seasonality, and Residuals—while guaranteeing valid uncertainty quantification via **State-Conditional Conformal Prediction (SCCP)**.

## 2. Core Architecture

The system consists of three main modules, corresponding to specific Python classes in the codebase:

1.  **Disentangled State Encoder** (`cissn.models.encoder.DisentangledStateEncoder`):
    -   Maps input sequences to a structural latent state space.
    -   Enforces specific dynamics (e.g., conservation of level, rotation of seasonality) via a structured transition matrix.
2.  **Forecast Head** (`cissn.models.forecast_head.ForecastHead`):
    -   Projects latent states into future horizons using a linear path for interpretability and a non-linear path for refinement.
3.  **State-Conditional Conformal Predictor** (`cissn.conformal.StateConditionalConformal`):
    -   Calibrates prediction intervals based on the regime (cluster) of the latent state.

### 2.1 State Space Structure ($s_t$)

The latent state $s_t \in \mathbb{R}^5$ is explicitly structured and disentangled. Unlike standard RNNs where hidden states are abstract, every dimension in CISSN has a physical meaning:

| Index | Component | Symbol | Description | Dynamics |
| :--- | :--- | :--- | :--- | :--- |
| 0 | **Level** | $s_t^{(0)}$ | Baseline value | Slow-varying, near-unit root. |
| 1 | **Trend** | $s_t^{(1)}$ | Rate of change | Smooth, persistent. |
| 2-3 | **Seasonal** | $s_t^{(2)}, s_t^{(3)}$ | Periodic behavior | Rotational (Sine/Cosine). |
| 4 | **Residual** | $s_t^{(4)}$ | Innovation/Noise | Fast-decaying, mean-reverting. |

### 2.2 Transition Dynamics

The state evolves according to a **Residual-Corrected Linear Transition**. This combines the stability of Kalman Filters with the flexibility of RNNs.

$$
s_t = \underbrace{\mathbf{A} s_{t-1}}_{\text{Physics}} + \underbrace{\mathbf{B}(x_t)}_{\text{Innovation}} + \underbrace{\text{Net}(s_{t-1}, x_t)}_{\text{Correction}}
$$

#### A. Structured Transition Matrix ($\mathbf{A}$)
The matrix $\mathbf{A}$ is block-diagonal and strictly constrained to enforce component behavior. The values are learned via sigmoid gating to ensure stability:

$$
\mathbf{A} = \begin{bmatrix}
\alpha_L & 0 & 0 & 0 & 0 \\
0 & \alpha_T & 0 & 0 & 0 \\
0 & 0 & \gamma \cos(\omega) & -\gamma \sin(\omega) & 0 \\
0 & 0 & \gamma \sin(\omega) & \gamma \cos(\omega) & 0 \\
0 & 0 & 0 & 0 & \alpha_R
\end{bmatrix}
$$

**Implementation Constraints** (from `encoder.py`):
-   **Level ($\alpha_L$)**: $\sigma(\theta_L) \cdot 0.15 + 0.85 \in [0.85, 1.0]$. Enforces long memory.
-   **Trend ($\alpha_T$)**: $\sigma(\theta_T) \cdot 0.25 + 0.70 \in [0.70, 0.95]$. Enforces smooth changes.
-   **Seasonal Damping ($\gamma$)**: $\sigma(\theta_\gamma) \cdot 0.20 + 0.80 \in [0.80, 1.0]$. Allows sustained oscillations.
-   **Frequency ($\omega$)**: Learnable parameter controlling the cycle length.
-   **Residual ($\alpha_R$)**: $\sigma(\theta_R) \cdot 0.40 \in [0.0, 0.4]$. Enforces fast decay (noise).

#### B. Innovation & Correction
-   **Innovation ($\mathbf{B}(x_t)$)**: A neural network projects the input $x_t$ directly into increments for each state component.
-   **Correction**: A small Tanh-activated network refines the state update, scaled by a small factor (initially 0.01) to keep the linear physics dominant.

---

## 3. Data Flow

### Step 1: Input Processing
-   **Input**: $X \in \mathbb{R}^{B \times T \times D_{in}}$
-   **Embedding**: Projected to hidden dimension $H$ via `Linear -> LayerNorm -> GELU`.

### Step 2: Recurrent Encoding
For each timestep $t$:
1.  **Innovation**: Compute updates $\Delta s_t$ from input.
2.  **Physics Update**: Apply linear transition $s_{linear} = \mathbf{A} s_{t-1} + \Delta s_t$.
3.  **Refinement**: $s_t = s_{linear} + 0.01 \cdot \text{Tanh}(\text{MLP}(s_{linear}, x_t))$.
4.  **Save**: Store $s_t$ for forecasting and explanation.

### Step 3: Hybrid Forecasting
The final state $s_T$ predicts the horizon $H$:
$$ \hat{Y} = \underbrace{s_T \mathbf{W}_{lin}}_{\text{Interpretable}} + \underbrace{\text{MLP}(s_T)}_{\text{Refinement}} + \text{Bias} $$
-   **Interpretability**: We can extract exact contributions (e.g., "The forecast is high because the **Trend** component is 1.5").

---

## 4. Uncertainty Quantification (SCCP)

We use **State-Conditional Conformal Prediction** to generate rigorous valid intervals.

### Why State-Conditional?
Standard conformal prediction assumes exchangeability (constant variance). Time series are heteroscedastic (variance changes with state). CISSN clusters the state space to adapt intervals to the current regime (e.g., "High Volatility" vs "Stable").

### Algorithm (from `state_conditional.py`)

1.  **Calibration**:
    -   Encode calibration set to get states $S_{cal}$.
    -   Compute absolute residuals $R_{cal} = |Y_{cal} - \hat{Y}_{cal}|$.
    -   **Cluster States**: Use K-Means ($K=5$) on $S_{cal}$ to find regimes.
    -   **Compute Quantiles**: For each cluster $k$, compute the $1-\alpha$ quantile $q_k$ of residuals.
    
    *Finite Sample Correction*:
    $$ q_k = \text{Quantile}\left(R_{k}, \frac{\lceil (n_k+1)(1-\alpha) \rceil}{n_k}\right) $$

2.  **Inference**:
    -   Encode new input to $s_{new}$.
    -   Identify cluster $k = \text{KMeans.predict}(s_{new})$.
    -   **Interval**: $[\hat{y} - q_k, \hat{y} + q_k]$.

---

## 5. Key Differentiators

| Feature | LSTM/Transformer | SSM (Kalman/ETS) | **CISSN (Ours)** |
| :--- | :--- | :--- | :--- |
| **Dynamics** | Black-box | Linear | **Structured + Non-linear correction** |
| **Seasonality** | Learned patterns | Explicit cycles | **Explicit 2D Rotation** |
| **Uncertainty** | Variational (Complex) | Gaussian (Wrong) | **Conformal (Distribution-Free)** |
| **Interpretability** | Low (Attention maps) | High | **High (Decomposed States)** |
