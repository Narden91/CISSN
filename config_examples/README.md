# Configurations for CISSN

This directory contains example configuration files for CISSN benchmark experiments.

## Usage

```bash
python experiments/run_benchmark.py --data ETTh1 --features M --pred_len 96
```

Run `python experiments/run_benchmark.py --help` for all available parameters. CISSN uses CLI arguments rather than YAML config files for benchmark experiments — see the argument parser in `experiments/run_benchmark.py` for the full list.

## Key Parameters (CLI)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data` | str | ETTh1 | Dataset name (ETTh1, ETTh2, ETTm1, ETTm2, weather, exchange_rate, ECL, traffic, ILI, solar) |
| `--features` | str | M | Forecasting task: M (multivariate), S (univariate), MS (multivariate→univariate) |
| `--seq_len` | int | 96 | Input sequence length |
| `--pred_len` | int | 96 | Prediction horizon |
| `--state_dim` | int | 5 | Latent state dimension (fixed at 5 for structured encoder) |
| `--d_model` | int | 64 | Hidden dimension for MLPs |
| `--lambda_cov` | float | 1.0 | Covariance loss weight (disentanglement) |
| `--lambda_temp` | float | 0.5 | Temporal consistency loss weight (disentanglement) |
| `--train_epochs` | int | 10 | Number of training epochs |
| `--batch_size` | int | 32 | Batch size |
| `--learning_rate` | float | 0.001 | Learning rate |
| `--patience` | int | 3 | Early stopping patience |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--use_wandb` | flag | — | Enable Weights & Biases logging |

## Structuring Your Own Experiments

For custom scripts that use the CISSN package directly, import from `cissn`:

```python
from cissn.models import DisentangledStateEncoder, ForecastHead
from cissn.conformal import StateConditionalConformal
from cissn.losses import DisentanglementLoss
from cissn.explanations import ForecastExplainer
from cissn.data import get_data_loader
```
