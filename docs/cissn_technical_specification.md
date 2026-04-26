# CISSN: Complete Technical Specification

## Conformally Calibrated Interpretable State-Space Networks

> A step-by-step guide to theory and implementation for publication-worthy time-series forecasting

---

# Part I: Theoretical Foundations

## 1. Problem Statement

### 1.1 The Time-Series Forecasting Problem

Given a multivariate time series:
```
X = {x_1, x_2, ..., x_T} where x_t ∈ ℝ^D
```

We aim to predict future values:
```
Ŷ = {ŷ_{T+1}, ŷ_{T+2}, ..., ŷ_{T+H}} where ŷ_t ∈ ℝ^D (or ℝ for univariate)
```

### 1.2 Limitations of Current Approaches

| Approach | Limitation |
|----------|------------|
| Transformers (Informer, Autoformer) | Black-box, no calibration guarantees |
| State-Space Models (Mamba, S4) | Focus on efficiency, states not interpretable |
| Bayesian Methods | Computationally expensive, approximate inference |
| Conformal Prediction | Treats model as black-box, fixed intervals |
| Classical Decomposition (STL) | Not learnable, limited expressiveness |

### 1.3 Our Goal

Design a framework that provides:
1. **Accurate forecasts** (competitive with SOTA)
2. **Interpretable states** (understand *why* a forecast was made)
3. **Calibrated uncertainty** (finite-sample coverage guarantees)

---

## 2. Disentangled State-Space Models

### 2.1 Classical State-Space Model

A linear state-space model (SSM) is defined by:
```
State transition:    s_t = A·s_{t-1} + B·x_t + w_t     (w_t ~ N(0, Q))
Observation model:   y_t = C·s_t + D·x_t + v_t         (v_t ~ N(0, R))
```

Where:
- `s_t ∈ ℝ^K` is the latent state
- `A` is the state transition matrix
- `B` is the input-to-state matrix
- `C` is the state-to-output matrix
- `D` is the direct input-to-output matrix

**Problem**: States `s_t` have no inherent meaning—they are arbitrary latent dimensions.

### 2.2 Disentangled State Representation

We impose structure on the state space so each dimension captures a specific temporal pattern.

**Definition (Disentangled State)**: A state representation `s_t = [s_t^{(1)}, s_t^{(2)}, ..., s_t^{(K)}]` is *disentangled* if:
1. State dimensions are statistically independent: `I(s^{(i)}; s^{(j)}) ≈ 0` for `i ≠ j`
2. Each dimension captures a distinct generative factor of variation

For time-series, we target K=5 interpretable dimensions (seasonal requires 2 dimensions for 2D rotation):

| Dimension | Notation | Semantic Meaning | Dynamics |
|-----------|----------|------------------|----------|
| 0 | `s^{(level)}` | Current level/mean | Slow-varying |
| 1 | `s^{(trend)}` | Direction of change | Smooth integration |
| 2-3 | `s^{(seasonal_0)}, s^{(seasonal_1)}` | Periodic patterns (cos/sin pair) | 2D rotation |
| 4 | `s^{(residual)}` | Innovation/noise | High-frequency |

### 2.3 Achieving Disentanglement (Unsupervised)

Without supervision, we encourage disentanglement through:

**Objective 1: Independence Constraint**
```
L_independence = ||off_diagonal(Σ_s)||_F²

where Σ_s = (1/T) Σ_t (s_t - μ_s)(s_t - μ_s)ᵀ
```

This penalizes correlation between state dimensions.

**Objective 2: Total Correlation Minimization**
```
TC(s) = KL(p(s) || Π_k p(s^{(k)}))
```

Approximated via:
```
L_TC ≈ Σ_k H(s^{(k)}) - H(s)
```

Where entropy can be estimated via batch statistics.

**Objective 3: Temporal Consistency**
```
L_temporal = Σ_t ||s_t^{(slow)} - s_{t-1}^{(slow)}||² 
           - λ·Var_t(s_t^{(fast)})
```

This encourages level/trend to be slow-varying and seasonal/residual to capture higher frequencies.

### 2.4 Structured Transition Matrix

To further encourage interpretable dynamics, we constrain the transition matrix `A`:

```
A = diag(A^{(level)}, A^{(trend)}, A^{(seasonal)}, A^{(residual)})
```

**Block 1 - Level** (near-unity eigenvalue):
```
A^{(level)} ≈ 1 - ε, where ε << 1
```
Encourages slow decay, capturing the baseline level.

**Block 2 - Trend** (integrator dynamics):
```
A^{(trend)} = [1, 1]
              [0, 1]  (for 2D trend state)
```
Or simply `≈ 1` for 1D, capturing accumulated direction.

**Block 3 - Seasonal** (oscillator):
```
A^{(seasonal)} = [cos(ω), -sin(ω)]
                 [sin(ω),  cos(ω)]
```
Where `ω = 2π/period` is learned or fixed.

**Block 4 - Residual** (fast-decaying):
```
A^{(residual)} = α, where |α| < 0.5
```
Encourages rapid forgetting of innovations.

---

## 3. Neural State-Space Architecture

### 3.1 Hybrid Linear-Nonlinear Design

Pure linear SSMs are limited in expressiveness. We add a nonlinear correction:

```
s_t = A·s_{t-1} + B·x_t + g_φ(s_{t-1}, x_t)
      ⎣________⎦   ⎣___⎦   ⎣______________⎦
      interpretable linear   expressive nonlinear
      dynamics               correction
```

**Key Design Choices**:
1. `g_φ` is a **sparse** MLP (dropout, L1 regularization)
2. `g_φ` outputs a **small correction** (scaled by learned factor `β < 1`)
3. Linear part dominates for interpretability

### 3.2 Encoder Architecture

```python
class DisentangledStateEncoder(nn.Module):
    """
    Encodes input sequence into disentangled state representation.
    
    Parameters:
        input_dim: D (number of input features)
        state_dim: K=5 (disentangled dimensions: level, trend, seasonal_0, seasonal_1, residual)
        hidden_dim: H (MLP hidden size)
    """
    
    def __init__(self, input_dim, state_dim=5, hidden_dim=64):
        # Input projection: x_t → intermediate
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Per-dimension state extractors
        self.level_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),  # Bounded output
            nn.Linear(hidden_dim // 2, 1)
        )
        self.trend_extractor = nn.Sequential(...)
        self.seasonal_extractor = nn.Sequential(...)
        self.residual_extractor = nn.Sequential(...)
        
        # Learnable transition matrices (structured)
        self.A_level = nn.Parameter(torch.tensor(0.99))
        self.A_trend = nn.Parameter(torch.tensor(0.95))
        self.A_seasonal = nn.Parameter(torch.zeros(2, 2))  # Rotation
        self.A_residual = nn.Parameter(torch.tensor(0.3))
        
        # Nonlinear correction
        self.correction_mlp = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, state_dim)
        )
        self.correction_scale = nn.Parameter(torch.tensor(0.1))
```

### 3.3 Forward Pass (Single Step)

```python
def step(self, x_t, s_prev):
    """
    Single state transition step.
    
    Args:
        x_t: Input at time t, shape (batch, input_dim)
        s_prev: Previous state, shape (batch, state_dim)
    
    Returns:
        s_t: New state, shape (batch, state_dim)
    """
    # Project input
    h_t = self.input_proj(x_t)  # (batch, hidden_dim)
    
    # Extract per-dimension innovations from input
    level_inn = self.level_extractor(h_t)
    trend_inn = self.trend_extractor(h_t)
    seasonal_inn = self.seasonal_extractor(h_t)
    residual_inn = self.residual_extractor(h_t)
    B_x = torch.cat([level_inn, trend_inn, seasonal_inn, residual_inn], dim=-1)
    
    # Linear state transition
    s_linear = self.apply_structured_A(s_prev) + B_x
    
    # Nonlinear correction (sparse)
    correction_input = torch.cat([s_prev, h_t], dim=-1)
    correction = self.correction_mlp(correction_input)
    correction = self.correction_scale * torch.tanh(correction)
    
    # Final state
    s_t = s_linear + correction
    
    return s_t

def apply_structured_A(self, s):
    """Apply block-diagonal structured transition."""
    s_level = s[:, 0:1] * torch.sigmoid(self.A_level) * 0.2 + 0.8  # ∈ [0.8, 1.0]
    s_trend = s[:, 1:2] * torch.sigmoid(self.A_trend) * 0.3 + 0.7  # ∈ [0.7, 1.0]
    s_seasonal = self.apply_rotation(s[:, 2:3], self.A_seasonal)
    s_residual = s[:, 3:4] * torch.sigmoid(self.A_residual) * 0.5  # ∈ [0, 0.5]
    return torch.cat([s_level, s_trend, s_seasonal, s_residual], dim=-1)
```

---

## 4. Conformal Prediction for Time-Series

### 4.1 Standard Split Conformal Prediction

**Setup**: 
- Training set: `{(X_i, y_i)}_{i=1}^{n_train}` for model training
- Calibration set: `{(X_i, y_i)}_{i=1}^{n_cal}` for conformal calibration
- Test set: `{X_i}_{i=1}^{n_test}` for prediction

**Algorithm**:
1. Train model f on training set
2. Compute conformity scores on calibration set:
   ```
   R_i = |y_i - f(X_i)|  (absolute residual)
   ```
3. Compute quantile:
   ```
   q_{1-α} = Quantile(R_1, ..., R_{n_cal}, level = ⌈(n_cal + 1)(1 - α)⌉ / n_cal)
   ```
4. Prediction interval for new X:
   ```
   C(X) = [f(X) - q_{1-α}, f(X) + q_{1-α}]
   ```

**Guarantee (Theorem)**: Under exchangeability:
```
P(y_{new} ∈ C(X_{new})) ≥ 1 - α
```

### 4.2 Limitations for Time-Series

1. **Exchangeability violation**: Time-series data is inherently sequential
2. **Fixed interval width**: Same width regardless of prediction difficulty
3. **No state awareness**: Ignores model's internal confidence

### 4.3 State-Conditional Conformal Prediction (Our Contribution)

**Key Insight**: The disentangled state `s_T` contains information about forecast difficulty. We should use different calibration quantiles for different states.

**Algorithm (SCCP - State-Conditional Conformal Prediction)**:

1. **Train model** and extract states for calibration set:
   ```
   For each (X_i, y_i) in calibration set:
       s_i = StateEncoder(X_i)  # Final state
       ŷ_i = ForecastHead(s_i)
       R_i = |y_i - ŷ_i|
   ```

2. **Cluster states** into M groups:
   ```
   clusters = KMeans(n_clusters=M).fit([s_1, ..., s_{n_cal}])
   ```

3. **Compute per-cluster quantiles**:
   ```
   For each cluster m = 1, ..., M:
       R^{(m)} = {R_i : cluster(s_i) = m}
       q^{(m)}_{1-α} = Quantile(R^{(m)}, level = ⌈(|R^{(m)}| + 1)(1 - α)⌉ / |R^{(m)}|)
   ```

4. **Prediction with adaptive intervals**:
   ```
   For new input X:
       s = StateEncoder(X)
       m = cluster(s)  # Assign to cluster
       ŷ = ForecastHead(s)
       C(X) = [ŷ - q^{(m)}_{1-α}, ŷ + q^{(m)}_{1-α}]
   ```

### 4.4 Theoretical Guarantee

**Theorem (State-Conditional Coverage)**:

Let `{(X_i, y_i)}_{i=1}^{n+1}` be exchangeable within each state cluster. Then:
```
P(y_{n+1} ∈ C(X_{n+1}) | cluster(s_{n+1}) = m) ≥ 1 - α
```

**Proof Sketch**:
- Within each cluster, samples are exchangeable by assumption
- Standard conformal guarantee applies within cluster
- Marginalizing over clusters preserves coverage

**Practical Relaxation**: 
For time-series, exact exchangeability doesn't hold. We use:
1. Sliding window calibration (recent data)
2. Adaptive recalibration (periodic updates)
3. Empirical coverage monitoring

### 4.5 Choosing Number of Clusters M

Trade-off:
- **M too small**: Intervals don't adapt to state (back to standard conformal)
- **M too large**: Insufficient samples per cluster (wide intervals)

**Heuristic**: M = ⌊√n_cal⌋ or use cross-validation

---

## 5. Decision-Level Explanations

### 5.1 Why This Forecast?

For each prediction, we provide a structured explanation:

```python
class ForecastExplanation:
    # State decomposition
    level_contribution: float      # How much level contributed
    trend_contribution: float      # How much trend contributed  
    seasonal_contribution: float   # How much seasonality contributed
    residual_contribution: float   # How much recent innovations contributed
    
    # Confidence assessment
    state_cluster: int             # Which regime we're in
    historical_error_in_cluster: float  # Typical error for this regime
    interval_width: float          # Calibrated uncertainty
    
    # Temporal trace
    state_history: List[StateVector]  # How we got here
```

### 5.2 Computing Contributions

**Linear Decomposition**:

Since the forecast is:
```
ŷ = W_level · s^{(level)} + W_trend · s^{(trend)} + W_seasonal · s^{(seasonal)} + W_residual · s^{(residual)} + bias
```

Each contribution is simply:
```
contribution_k = W_k · s^{(k)} / |ŷ - bias|
```

**Gradient-Based Attribution** (for nonlinear head):
```
contribution_k = s^{(k)} · ∂ŷ/∂s^{(k)}
```

### 5.3 Counterfactual Explanations

"What if the trend were different?"

```python
def counterfactual(self, x, s_original, dimension, new_value):
    """
    Generate counterfactual forecast.
    
    Args:
        x: Input sequence
        s_original: Original state
        dimension: Which state dimension to modify (0-3)
        new_value: New value for that dimension
    """
    s_counterfactual = s_original.clone()
    s_counterfactual[:, dimension] = new_value
    
    y_original = self.forecast_head(s_original)
    y_counterfactual = self.forecast_head(s_counterfactual)
    
    return {
        'original_forecast': y_original,
        'counterfactual_forecast': y_counterfactual,
        'difference': y_counterfactual - y_original,
        'dimension_modified': ['level', 'trend', 'seasonal', 'residual'][dimension]
    }
```

---

# Part II: Implementation Guide

## 6. Project Structure

```
cissn/
├── cissn/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py              # DisentangledStateEncoder
│   │   ├── transition.py           # Structured state transitions
│   │   ├── forecast_head.py        # Multi-horizon prediction
│   │   ├── cissn.py                # Main CISSN model
│   │   └── baselines/              # Baseline implementations
│   ├── conformal/
│   │   ├── __init__.py
│   │   ├── base.py                 # Standard conformal
│   │   ├── state_conditional.py    # SCCP implementation
│   │   └── calibration.py          # Calibration utilities
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── forecast_loss.py        # MSE, MAE, quantile
│   │   ├── disentangle_loss.py     # Independence, TC
│   │   └── combined.py             # Multi-objective loss
│   ├── explanations/
│   │   ├── __init__.py
│   │   ├── state_attribution.py    # Contribution analysis
│   │   ├── counterfactual.py       # Counterfactual generation
│   │   └── visualization.py        # Plotting utilities
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py             # ETT, Weather, Traffic loaders
│   │   ├── preprocessing.py        # Normalization, windowing
│   │   └── dataloaders.py          # PyTorch DataLoaders
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Main training loop
│   │   ├── scheduler.py            # LR scheduling
│   │   └── early_stopping.py       # Early stopping
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # MSE, MAE, coverage, etc.
│   │   └── calibration_metrics.py  # ECE, PICP, MPIW
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       └── logging.py              # Experiment logging
├── configs/
│   ├── default.yaml
│   ├── etth1.yaml
│   ├── weather.yaml
│   └── ...
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── experiments/
│   └── run_all_benchmarks.py
├── tests/
│   ├── test_encoder.py
│   ├── test_conformal.py
│   └── ...
├── requirements.txt
└── README.md
```

---

## 7. Step-by-Step Implementation

### Step 1: Core State Encoder

```python
# cissn/models/encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DisentangledStateEncoder(nn.Module):
    """
    Encodes input time series into K=5 disentangled state dimensions.
    
    State dimensions:
        0: Level - slow-varying baseline
        1: Trend - direction of change
        2: Seasonal (cosine) - periodic component
        3: Seasonal (sine) - periodic component (forms 2D rotation with dim 2)
        4: Residual - fast-changing innovations
    """
    
    def __init__(
        self,
        input_dim: int,
        state_dim: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Per-dimension innovation extractors
        self.extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(state_dim)
        ])
        
        # Structured transition parameters
        # Level: should stay close to 1 (slow decay)
        self.A_level = nn.Parameter(torch.tensor(2.0))  # sigmoid → ~0.88
        
        # Trend: moderate persistence
        self.A_trend = nn.Parameter(torch.tensor(1.0))  # sigmoid → ~0.73
        
        # Seasonal: rotation matrix parameters (learnable frequency)
        self.omega = nn.Parameter(torch.tensor(0.1))  # Initial frequency
        
        # Residual: fast decay
        self.A_residual = nn.Parameter(torch.tensor(-1.0))  # sigmoid → ~0.27
        
        # Nonlinear correction network
        self.correction_net = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 2),  # Higher dropout for sparsity
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh()  # Bounded output
        )
        self.correction_scale = nn.Parameter(torch.tensor(0.1))
        
        # Initial state (learnable)
        self.initial_state = nn.Parameter(torch.zeros(1, state_dim))
        
    def get_structured_A(self) -> torch.Tensor:
        """Construct the diagonal transition matrix."""
        A_level = torch.sigmoid(self.A_level) * 0.15 + 0.85  # [0.85, 1.0]
        A_trend = torch.sigmoid(self.A_trend) * 0.25 + 0.70  # [0.70, 0.95]
        A_seasonal = torch.cos(self.omega)  # Rotation (simplified 1D)
        A_residual = torch.sigmoid(self.A_residual) * 0.40   # [0.0, 0.4]
        
        return torch.stack([A_level, A_trend, A_seasonal, A_residual])
    
    def step(self, x_t: torch.Tensor, s_prev: torch.Tensor) -> torch.Tensor:
        """
        Single state transition step.
        
        Args:
            x_t: (batch, input_dim) - input at time t
            s_prev: (batch, state_dim) - previous state
            
        Returns:
            s_t: (batch, state_dim) - new state
        """
        batch_size = x_t.size(0)
        
        # Project input
        h_t = self.input_proj(x_t)  # (batch, hidden_dim)
        
        # Extract innovations for each state dimension
        innovations = []
        for extractor in self.extractors:
            inn = extractor(h_t)  # (batch, 1)
            innovations.append(inn)
        B_x = torch.cat(innovations, dim=-1)  # (batch, state_dim)
        
        # Apply structured transition
        A_diag = self.get_structured_A()  # (state_dim,)
        s_linear = s_prev * A_diag.unsqueeze(0) + B_x
        
        # Nonlinear correction (sparse)
        correction_input = torch.cat([s_prev, h_t], dim=-1)
        correction = self.correction_net(correction_input)
        correction = self.correction_scale * correction
        
        s_t = s_linear + correction
        
        return s_t
    
    def forward(
        self, 
        x: torch.Tensor,
        return_all_states: bool = False
    ) -> torch.Tensor:
        """
        Encode full sequence into states.
        
        Args:
            x: (batch, seq_len, input_dim) - input sequence
            return_all_states: if True, return all intermediate states
            
        Returns:
            If return_all_states:
                states: (batch, seq_len, state_dim)
            Else:
                final_state: (batch, state_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize state
        s_t = self.initial_state.expand(batch_size, -1)
        
        states = []
        for t in range(seq_len):
            s_t = self.step(x[:, t, :], s_t)
            if return_all_states:
                states.append(s_t.unsqueeze(1))
        
        if return_all_states:
            return torch.cat(states, dim=1)  # (batch, seq_len, state_dim)
        return s_t  # (batch, state_dim)
```

### Step 2: Forecast Head

```python
# cissn/models/forecast_head.py

import torch
import torch.nn as nn

class ForecastHead(nn.Module):
    """
    Generates forecasts from disentangled state representation.
    
    Supports multi-horizon forecasting with interpretable per-dimension contributions.
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        output_dim: int = 1,
        horizon: int = 1,
        hidden_dim: int = 32
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.horizon = horizon
        
        # Linear contributions (interpretable)
        self.linear_weights = nn.Parameter(torch.randn(state_dim, output_dim * horizon) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_dim * horizon))
        
        # Small nonlinear refinement
        self.refinement = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim * horizon),
            nn.Tanh()
        )
        self.refinement_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Generate forecast from state.
        
        Args:
            state: (batch, state_dim)
            
        Returns:
            forecast: (batch, horizon, output_dim)
        """
        # Linear contribution (interpretable)
        linear_out = torch.matmul(state, self.linear_weights) + self.bias
        
        # Nonlinear refinement (small)
        nonlinear_out = self.refinement_scale * self.refinement(state)
        
        # Combine
        out = linear_out + nonlinear_out
        
        # Reshape to (batch, horizon, output_dim)
        batch_size = state.size(0)
        return out.view(batch_size, self.horizon, self.output_dim)
    
    def get_contributions(self, state: torch.Tensor) -> dict:
        """
        Get per-dimension contributions to forecast.
        
        Returns interpretable breakdown of forecast.
        """
        contributions = {}
        dimension_names = ['level', 'trend', 'seasonal_0', 'seasonal_1', 'residual']
        
        for i, name in enumerate(dimension_names):
            # Contribution of dimension i
            contrib = state[:, i:i+1] * self.linear_weights[i:i+1, :]
            contributions[name] = contrib.sum(dim=-1)  # Sum over output dims
            
        contributions['bias'] = self.bias.sum()
        contributions['total_linear'] = torch.matmul(state, self.linear_weights).sum(dim=-1)
        
        return contributions
```

### Step 3: Disentanglement Loss

```python
# cissn/losses/disentangle_loss.py

import torch
import torch.nn as nn

class DisentanglementLoss(nn.Module):
    """
    Unsupervised loss to encourage disentangled state representations.
    
    Components:
    1. Covariance regularization (independence)
    2. Temporal consistency (slow vs fast dynamics)
    """
    
    def __init__(
        self,
        lambda_cov: float = 1.0,
        lambda_temporal: float = 0.5
    ):
        super().__init__()
        self.lambda_cov = lambda_cov
        self.lambda_temporal = lambda_temporal
        
    def covariance_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Penalize off-diagonal covariance (encourage independence).
        
        Args:
            states: (batch, seq_len, state_dim)
        """
        # Flatten batch and time
        batch_size, seq_len, state_dim = states.shape
        flat_states = states.reshape(-1, state_dim)  # (batch*seq_len, state_dim)
        
        # Center the states
        mean = flat_states.mean(dim=0, keepdim=True)
        centered = flat_states - mean
        
        # Compute covariance matrix
        cov = torch.mm(centered.t(), centered) / (flat_states.size(0) - 1)
        
        # Penalize off-diagonal elements
        # Create mask for off-diagonal
        eye = torch.eye(state_dim, device=states.device)
        off_diag_cov = cov * (1 - eye)
        
        return torch.norm(off_diag_cov, p='fro') ** 2
    
    def temporal_consistency_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Encourage appropriate temporal dynamics per dimension.
        
        - Level (dim 0): Very slow change
        - Trend (dim 1): Slow change
        - Seasonal (dim 2): Medium variation
        - Residual (dim 3): Fast change (penalize smoothness)
        """
        # states: (batch, seq_len, state_dim)
        
        # Compute temporal differences
        diffs = states[:, 1:, :] - states[:, :-1, :]  # (batch, seq_len-1, state_dim)
        diff_var = diffs.var(dim=(0, 1))  # (state_dim,)
        
        # Target variances (relative)
        # Level should have smallest variance, residual should have largest
        target_order = torch.tensor([0.1, 0.3, 0.6, 1.0], device=states.device)
        
        # Normalize actual variances
        normalized_var = diff_var / (diff_var.sum() + 1e-8)
        
        # Loss: encourage ordering
        loss = nn.functional.mse_loss(normalized_var, target_order / target_order.sum())
        
        return loss
    
    def forward(self, states: torch.Tensor) -> dict:
        """
        Compute total disentanglement loss.
        
        Args:
            states: (batch, seq_len, state_dim)
            
        Returns:
            Dictionary with 'total' and individual components
        """
        cov_loss = self.covariance_loss(states)
        temporal_loss = self.temporal_consistency_loss(states)
        
        total = self.lambda_cov * cov_loss + self.lambda_temporal * temporal_loss
        
        return {
            'total': total,
            'covariance': cov_loss,
            'temporal': temporal_loss
        }
```

### Step 4: State-Conditional Conformal Prediction

```python
# cissn/conformal/state_conditional.py

import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, Optional

class StateConditionalConformal:
    """
    State-Conditional Conformal Prediction (SCCP).
    
    Uses disentangled states to compute adaptive prediction intervals.
    """
    
    def __init__(
        self,
        n_clusters: int = 4,
        alpha: float = 0.1  # 90% coverage
    ):
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        self.clusterer = None
        self.cluster_quantiles = {}
        self.is_calibrated = False
        
    def calibrate(
        self,
        states: np.ndarray,      # (n_cal, state_dim)
        predictions: np.ndarray,  # (n_cal,) or (n_cal, horizon)
        targets: np.ndarray       # (n_cal,) or (n_cal, horizon)
    ):
        """
        Calibrate conformal predictor using calibration set.
        
        Args:
            states: State representations from encoder
            predictions: Model predictions
            targets: True values
        """
        # Compute conformity scores (absolute residuals)
        residuals = np.abs(predictions - targets)
        if residuals.ndim > 1:
            residuals = residuals.mean(axis=1)  # Average over horizon
            
        # Cluster states
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.clusterer.fit_predict(states)
        
        # Compute quantile for each cluster
        for m in range(self.n_clusters):
            cluster_mask = cluster_labels == m
            cluster_residuals = residuals[cluster_mask]
            
            if len(cluster_residuals) > 0:
                # Compute conformal quantile
                n_m = len(cluster_residuals)
                q_level = np.ceil((n_m + 1) * (1 - self.alpha)) / n_m
                q_level = min(q_level, 1.0)  # Cap at 1
                
                self.cluster_quantiles[m] = np.quantile(cluster_residuals, q_level)
            else:
                # Fallback: use global quantile
                self.cluster_quantiles[m] = np.quantile(residuals, 1 - self.alpha)
                
        self.is_calibrated = True
        
        # Store calibration stats
        self.calibration_stats = {
            'n_samples': len(states),
            'n_clusters': self.n_clusters,
            'cluster_sizes': [np.sum(cluster_labels == m) for m in range(self.n_clusters)],
            'cluster_quantiles': dict(self.cluster_quantiles),
            'global_quantile': np.quantile(residuals, 1 - self.alpha)
        }
        
    def predict(
        self,
        state: np.ndarray,       # (1, state_dim) or (batch, state_dim)
        point_prediction: np.ndarray  # (1,) or (batch,) or with horizon
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate prediction intervals.
        
        Returns:
            lower: Lower bound of interval
            upper: Upper bound of interval
            cluster: Assigned cluster for each sample
        """
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() before predict()")
            
        # Ensure 2D
        if state.ndim == 1:
            state = state.reshape(1, -1)
            
        # Assign to clusters
        clusters = self.clusterer.predict(state)
        
        # Get quantiles for each sample
        quantiles = np.array([self.cluster_quantiles[c] for c in clusters])
        
        # Handle multi-dimensional predictions
        if point_prediction.ndim == 1:
            lower = point_prediction - quantiles
            upper = point_prediction + quantiles
        else:
            quantiles = quantiles.reshape(-1, 1)
            lower = point_prediction - quantiles
            upper = point_prediction + quantiles
            
        return lower, upper, clusters
    
    def evaluate_coverage(
        self,
        states: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> dict:
        """Evaluate empirical coverage on a dataset."""
        lower, upper, clusters = self.predict(states, predictions)
        
        # Check coverage
        covered = (targets >= lower) & (targets <= upper)
        if covered.ndim > 1:
            covered = covered.all(axis=1)  # All horizons must be covered
            
        overall_coverage = covered.mean()
        
        # Per-cluster coverage
        cluster_coverage = {}
        for m in range(self.n_clusters):
            mask = clusters == m
            if mask.sum() > 0:
                cluster_coverage[m] = covered[mask].mean()
                
        # Interval width statistics
        widths = upper - lower
        if widths.ndim > 1:
            widths = widths.mean(axis=1)
            
        return {
            'overall_coverage': overall_coverage,
            'target_coverage': 1 - self.alpha,
            'cluster_coverage': cluster_coverage,
            'mean_interval_width': widths.mean(),
            'std_interval_width': widths.std()
        }
```

### Step 5: Main CISSN Model

```python
# cissn/models/cissn.py

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .encoder import DisentangledStateEncoder
from .forecast_head import ForecastHead

class CISSN(nn.Module):
    """
    Conformally Calibrated Interpretable State-Space Network.
    
    Main model class combining:
    - Disentangled state encoder
    - Multi-horizon forecast head
    - Explanation generation
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        state_dim: int = 5,
        hidden_dim: int = 64,
        horizon: int = 96,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.horizon = horizon
        
        # Core components
        self.encoder = DisentangledStateEncoder(
            input_dim=input_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.forecast_head = ForecastHead(
            state_dim=state_dim,
            output_dim=output_dim,
            horizon=horizon,
            hidden_dim=hidden_dim // 2
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim) - input sequence
            return_states: whether to return all intermediate states
            
        Returns:
            Dictionary containing:
                - 'forecast': (batch, horizon, output_dim)
                - 'final_state': (batch, state_dim)
                - 'all_states': (batch, seq_len, state_dim) if return_states
                - 'contributions': per-dimension contributions
        """
        # Encode sequence
        if return_states:
            all_states = self.encoder(x, return_all_states=True)
            final_state = all_states[:, -1, :]
        else:
            final_state = self.encoder(x, return_all_states=False)
            all_states = None
            
        # Generate forecast
        forecast = self.forecast_head(final_state)
        
        # Get contributions
        contributions = self.forecast_head.get_contributions(final_state)
        
        outputs = {
            'forecast': forecast,
            'final_state': final_state,
            'contributions': contributions
        }
        
        if return_states:
            outputs['all_states'] = all_states
            
        return outputs
    
    def explain(self, x: torch.Tensor) -> Dict:
        """
        Generate detailed explanation for a forecast.
        
        Returns human-readable explanation dictionary.
        """
        with torch.no_grad():
            outputs = self.forward(x, return_states=True)
            
        state = outputs['final_state']
        contributions = outputs['contributions']
        
        # Normalize contributions to percentages
        total = sum(outputs['contributions'][k].abs().mean().item() 
                   for k in ['level', 'trend', 'seasonal', 'residual'])
        
        explanation = {
            'forecast': outputs['forecast'].squeeze().tolist(),
            'state_values': {
                'level': state[:, 0].mean().item(),
                'trend': state[:, 1].mean().item(),
                'seasonal': state[:, 2].mean().item(),
                'residual': state[:, 3].mean().item()
            },
            'contribution_percentages': {
                'level': contributions['level'].abs().mean().item() / total * 100,
                'trend': contributions['trend'].abs().mean().item() / total * 100,
                'seasonal': contributions['seasonal'].abs().mean().item() / total * 100,
                'residual': contributions['residual'].abs().mean().item() / total * 100
            }
        }
        
        return explanation
```

### Step 6: Training Loop

```python
# cissn/training/trainer.py

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional
import numpy as np

from ..models.cissn import CISSN
from ..losses.disentangle_loss import DisentanglementLoss
from ..conformal.state_conditional import StateConditionalConformal

class CISSNTrainer:
    """
    Trainer for CISSN model.
    
    Handles:
    - Multi-objective training (forecast + disentanglement)
    - Conformal calibration after training
    - Evaluation with calibration metrics
    """
    
    def __init__(
        self,
        model: CISSN,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_disentangle: float = 0.1,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        
        # Losses
        self.forecast_loss = nn.MSELoss()
        self.disentangle_loss = DisentanglementLoss()
        self.lambda_disentangle = lambda_disentangle
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Conformal predictor (calibrated after training)
        self.conformal = StateConditionalConformal(n_clusters=4, alpha=0.1)
        
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_forecast_loss = 0
        total_disentangle_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            x, y = batch  # x: (batch, seq_len, dim), y: (batch, horizon, dim)
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with states for disentanglement loss
            outputs = self.model(x, return_states=True)
            forecast = outputs['forecast']
            all_states = outputs['all_states']
            
            # Forecast loss
            loss_forecast = self.forecast_loss(forecast, y)
            
            # Disentanglement loss
            loss_disentangle_dict = self.disentangle_loss(all_states)
            loss_disentangle = loss_disentangle_dict['total']
            
            # Combined loss
            loss = loss_forecast + self.lambda_disentangle * loss_disentangle
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_forecast_loss += loss_forecast.item()
            total_disentangle_loss += loss_disentangle.item()
            n_batches += 1
            
        return {
            'loss': total_loss / n_batches,
            'forecast_loss': total_forecast_loss / n_batches,
            'disentangle_loss': total_disentangle_loss / n_batches
        }
    
    def calibrate(self, calibration_loader):
        """Calibrate conformal predictor after training."""
        self.model.eval()
        
        all_states = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in calibration_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x)
                
                all_states.append(outputs['final_state'].cpu().numpy())
                all_predictions.append(outputs['forecast'].cpu().numpy())
                all_targets.append(y.cpu().numpy())
                
        states = np.concatenate(all_states, axis=0)
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Flatten if needed
        if predictions.ndim == 3:
            predictions = predictions.reshape(predictions.shape[0], -1)
            targets = targets.reshape(targets.shape[0], -1)
            
        self.conformal.calibrate(states, predictions, targets)
        
        return self.conformal.calibration_stats
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """Evaluate model with calibration metrics."""
        self.model.eval()
        
        all_states = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x)
                
                all_states.append(outputs['final_state'].cpu().numpy())
                all_predictions.append(outputs['forecast'].cpu().numpy())
                all_targets.append(y.cpu().numpy())
                
        states = np.concatenate(all_states, axis=0)
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Flatten
        predictions_flat = predictions.reshape(predictions.shape[0], -1)
        targets_flat = targets.reshape(targets.shape[0], -1)
        
        # Forecast metrics
        mse = ((predictions - targets) ** 2).mean()
        mae = np.abs(predictions - targets).mean()
        
        # Calibration metrics
        if self.conformal.is_calibrated:
            coverage_stats = self.conformal.evaluate_coverage(
                states, predictions_flat, targets_flat
            )
        else:
            coverage_stats = {}
            
        return {
            'mse': mse,
            'mae': mae,
            **coverage_stats
        }
```

---

## 8. Experiments Configuration

```yaml
# configs/etth1.yaml

model:
  input_dim: 7        # ETTh1 has 7 features
  output_dim: 1       # Predict OT (oil temperature)
  state_dim: 5        # Disentangled dimensions (level, trend, seasonal_0, seasonal_1, residual)
  hidden_dim: 64
  horizon: 96         # Predict 96 steps ahead
  dropout: 0.1

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  lambda_disentangle: 0.1
  early_stopping_patience: 10

data:
  dataset: etth1
  seq_len: 336        # Look-back window
  pred_len: 96        # Prediction horizon
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2

conformal:
  n_clusters: 4
  alpha: 0.1          # 90% coverage target
  calibration_ratio: 0.3  # 30% of validation for calibration

evaluation:
  metrics:
    - mse
    - mae
    - coverage
    - interval_width
    - ece  # Expected Calibration Error
```

---

# Part III: Experimental Roadmap

## 9. Week-by-Week Plan

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1-2 | Core implementation | Encoder, ForecastHead, basic training |
| 3 | Disentanglement | Loss functions, validation of state separation |
| 4 | Conformal layer | SCCP implementation, calibration code |
| 5-6 | Data pipelines | ETT, Weather, Traffic loaders |
| 7-8 | Baseline comparison | Informer, Autoformer, PatchTST |
| 9-10 | Ablation studies | Remove each component, measure impact |
| 11-12 | Paper writing | Draft introduction, methods, experiments |
| 13-14 | Refinement | Polish results, camera-ready |

---

## 10. Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| MSE | Within 5% of SOTA | Competitive accuracy |
| Coverage@90% | ≥ 88% | Near-valid coverage |
| Interval Width | 20-30% tighter than baseline conformal | Benefit of state-conditioning |
| Disentanglement | Correlation < 0.2 between state dims | Meaningful separation |

---

This document provides the complete theoretical and practical foundation for implementing CISSN. Ready to begin implementation?
