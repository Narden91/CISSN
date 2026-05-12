"""Canonical dataset metadata for CISSN experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "ETTh1": {
        "root_path": "./data/ETT/",
        "data_path": "ETTh1.csv",
        "freq": "h",
        "enc_in": 7,
        "c_out": 7,
        "target": "OT",
        "horizons": [24, 96, 192, 336, 720],
    },
    "ETTh2": {
        "root_path": "./data/ETT/",
        "data_path": "ETTh2.csv",
        "freq": "h",
        "enc_in": 7,
        "c_out": 7,
        "target": "OT",
        "horizons": [24, 96, 192, 336, 720],
    },
    "ETTm1": {
        "root_path": "./data/ETT/",
        "data_path": "ETTm1.csv",
        "freq": "t",
        "enc_in": 7,
        "c_out": 7,
        "target": "OT",
        "horizons": [24, 96, 192, 336, 720],
    },
    "ETTm2": {
        "root_path": "./data/ETT/",
        "data_path": "ETTm2.csv",
        "freq": "t",
        "enc_in": 7,
        "c_out": 7,
        "target": "OT",
        "horizons": [24, 96, 192, 336, 720],
    },
    "weather": {
        "root_path": "./data/",
        "data_path": "weather.csv",
        "freq": "t",
        "enc_in": 21,
        "c_out": 21,
        "target": "OT",
        "horizons": [96, 192, 336, 720],
    },
    "exchange_rate": {
        "root_path": "./data/",
        "data_path": "exchange_rate.csv",
        "freq": "d",
        "enc_in": 8,
        "c_out": 8,
        "target": "OT",
        "horizons": [96, 192, 336, 720],
    },
    "ECL": {
        "root_path": "./data/",
        "data_path": "electricity.csv",
        "freq": "h",
        "enc_in": 321,
        "c_out": 321,
        "target": "OT",
        "horizons": [96, 192, 336, 720],
    },
    "traffic": {
        "root_path": "./data/",
        "data_path": "traffic.csv",
        "freq": "h",
        "enc_in": 862,
        "c_out": 862,
        "target": "OT",
        "horizons": [96, 192, 336, 720],
    },
    "ILI": {
        "root_path": "./data/",
        "data_path": "national_illness.csv",
        "freq": "w",
        "enc_in": 7,
        "c_out": 7,
        "target": "OT",
        "horizons": [24, 36, 48, 60],
    },
    "solar": {
        "root_path": "./data/",
        "data_path": "solar_AL.txt",
        "freq": "t",
        "enc_in": 137,
        "c_out": 137,
        "target": "OT",
        "horizons": [96, 192, 336, 720],
    },
}


def get_dataset_spec(name: str) -> dict[str, Any]:
    """Return a copy of the canonical metadata for a supported dataset."""
    if name not in DATASET_REGISTRY:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset {name!r}. Supported datasets: {supported}.")
    return deepcopy(DATASET_REGISTRY[name])


def supported_datasets() -> list[str]:
    return sorted(DATASET_REGISTRY)
