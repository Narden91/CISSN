# Benchmark Datasets for LTSF

Benchmark datasets used for Long-Term Time Series Forecasting (LTSF) experiments.
All datasets are automatically downloadable via `scripts/download_datasets.py`.

## Dataset Overview

| Dataset | Domain | Frequency | Features | Source |
| :--- | :--- | :--- | :--- | :--- |
| **ETTh1, ETTh2** | Electricity transformer temperature | Hourly | 7 | GitHub — zhouhaoyi/ETDataset |
| **ETTm1, ETTm2** | Electricity transformer temperature | 15-min | 7 | GitHub — zhouhaoyi/ETDataset |
| **Weather** | Meteorological indicators | 10-min | 21 | Autoformer Google Drive |
| **Exchange-Rate** | Daily FX rates (8 countries) | Daily | 8 | Autoformer Google Drive |
| **ECL (Electricity)** | Hourly electricity consumption | Hourly | 321 | Autoformer Google Drive |
| **Traffic** | Road occupancy (862 sensors) | Hourly | 862 | Autoformer Google Drive |
| **ILI** | Influenza-like illness | Weekly | 7 | Autoformer Google Drive |
| **Solar-Energy** | Solar power (137 Alabama stations) | 10-min | 137 | GitHub — laiguokun/multivariate-time-series-data |

## Download

The automated script handles all datasets including gzip decompression for Solar-Energy:

```bash
# Download all datasets
uv run scripts/download_datasets.py

# Check which datasets are already present
uv run scripts/download_datasets.py --status

# Download specific datasets only
uv run scripts/download_datasets.py --datasets ETTh1,ETTh2,weather
```

ETT and Solar-Energy download directly from GitHub (no extra dependencies).
Weather, Exchange-Rate, ECL, Traffic, and ILI come from the Autoformer Google Drive folder
and require `gdown` — the script will offer to install it automatically.

## Expected Directory Layout

```
data/
├── ETT/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── weather.csv
├── exchange_rate.csv
├── electricity.csv
├── traffic.csv
├── national_illness.csv
└── solar_AL.txt
```

## Dataset Registry (data_loader.py)

The `get_data_loader` function maps dataset names to classes and default frequencies:

| Key | Class | Default freq |
|-----|-------|-------------|
| `ETTh1`, `ETTh2` | `Dataset_ETT_hour` | `h` |
| `ETTm1`, `ETTm2` | `Dataset_ETT_minute` | `t` |
| `weather` | `Dataset_Custom` | `t` (10-min) |
| `exchange_rate` | `Dataset_Custom` | `d` (daily) |
| `ECL` | `Dataset_Custom` | `h` (hourly) |
| `traffic` | `Dataset_Custom` | `h` (hourly) |
| `ILI` | `Dataset_Custom` | `w` (weekly) |
| `solar` | `Dataset_Custom` | `t` (10-min) |

All ETT datasets use a 12/4/4-month train/val/test split.
All Custom datasets use a 70%/10%/20% proportional split.
