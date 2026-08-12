"""Canonical dataset metadata for CISSN experiments."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "ETTh1": {
        "root_path": "./data/ETT/",
        "data_path": "ETTh1.csv",
        "freq": "h",
        "enc_in": 7,
        "c_out": 7,
        "target": "OT",
        "horizons": [24, 96, 192, 336, 720],
        "integrity": {
            "n_rows": 17420,
            "date_start": "2016-07-01 00:00:00",
            "date_end": "2018-06-26 19:00:00",
            "sha256": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
            "min_col_mean_abs": 0.5,
            "min_lag1_autocorr": 0.80,
        },
    },
    "ETTh2": {
        "root_path": "./data/ETT/",
        "data_path": "ETTh2.csv",
        "freq": "h",
        "enc_in": 7,
        "c_out": 7,
        "target": "OT",
        "horizons": [24, 96, 192, 336, 720],
        "integrity": {
            "n_rows": 17420,
            "date_start": "2016-07-01 00:00:00",
            "date_end": "2018-06-26 19:00:00",
            "sha256": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
            "min_col_mean_abs": 0.5,
            "min_lag1_autocorr": 0.80,
        },
    },
    "ETTm1": {
        "root_path": "./data/ETT/",
        "data_path": "ETTm1.csv",
        "freq": "t",
        "enc_in": 7,
        "c_out": 7,
        "target": "OT",
        "horizons": [24, 96, 192, 336, 720],
        "integrity": {
            "n_rows": 69680,
            "date_start": "2016-07-01 00:00:00",
            "date_end": "2018-06-26 19:45:00",
            "sha256": "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
            "min_col_mean_abs": 0.5,
            "min_lag1_autocorr": 0.80,
        },
    },
    "ETTm2": {
        "root_path": "./data/ETT/",
        "data_path": "ETTm2.csv",
        "freq": "t",
        "enc_in": 7,
        "c_out": 7,
        "target": "OT",
        "horizons": [24, 96, 192, 336, 720],
        "integrity": {
            "n_rows": 69680,
            "date_start": "2016-07-01 00:00:00",
            "date_end": "2018-06-26 19:45:00",
            "sha256": "db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd",
            "min_col_mean_abs": 0.5,
            "min_lag1_autocorr": 0.80,
        },
    },
    "weather": {
        "root_path": "./data/",
        "data_path": "weather.csv",
        "freq": "t",
        "enc_in": 21,
        "c_out": 21,
        "target": "OT",
        "horizons": [96, 192, 336, 720],
        "integrity": {
            "n_rows": 52696,
            "date_start": "2020-01-01 00:10:00",
            "date_end": "2021-01-01 00:00:00",
            "sha256": "34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63",
            "min_col_mean_abs": 0.5,
            "min_lag1_autocorr": 0.60,
        },
    },
    "exchange_rate": {
        "root_path": "./data/",
        "data_path": "exchange_rate.csv",
        "freq": "d",
        "enc_in": 8,
        "c_out": 8,
        "target": "OT",
        "horizons": [96, 192, 336, 720],
        "integrity": {
            "n_rows": 7588,
            "date_start": "1990/1/1 0:00",
            "date_end": "2010/10/10 0:00",
            "sha256": "48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842",
            "min_col_mean_abs": 0.1,
            "min_lag1_autocorr": 0.80,
        },
    },
    "ECL": {
        "root_path": "./data/",
        "data_path": "electricity.csv",
        "freq": "h",
        "enc_in": 321,
        "c_out": 321,
        "target": "OT",
        "horizons": [96, 192, 336, 720],
        "integrity": {
            "n_rows": 26304,
            "date_start": "2016-07-01 02:00:00",
            "date_end": "2019-07-02 01:00:00",
            "sha256": "7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad",
            "min_col_mean_abs": 0.5,
            "min_lag1_autocorr": 0.60,
        },
    },
    "traffic": {
        "root_path": "./data/",
        "data_path": "traffic.csv",
        "freq": "h",
        "enc_in": 862,
        "c_out": 862,
        "target": "OT",
        "horizons": [96, 192, 336, 720],
        "integrity": {
            "n_rows": 17544,
            "date_start": "2016-07-01 02:00:00",
            "date_end": "2018-07-02 01:00:00",
            "sha256": "cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16",
            "min_col_mean_abs": 0.01,
            "min_lag1_autocorr": 0.60,
        },
    },
    "ILI": {
        "root_path": "./data/",
        "data_path": "national_illness.csv",
        "freq": "w",
        "enc_in": 7,
        "c_out": 7,
        "target": "OT",
        "horizons": [24, 36, 48, 60],
        "integrity": {
            "n_rows": 966,
            "date_start": "2002-01-01 00:00:00",
            "date_end": "2020-06-30 00:00:00",
            "sha256": "93601f64d2566dc796ca4305adad8b8560c2db1a1ff04543c3bd813a7263570a",
            "min_col_mean_abs": 1.0,
            "min_lag1_autocorr": 0.60,
        },
    },
    "solar": {
        "root_path": "./data/",
        "data_path": "solar_AL.txt",
        "freq": "t",
        "enc_in": 137,
        "c_out": 137,
        "target": "OT",
        "horizons": [96, 192, 336, 720],
        # solar_AL.txt has no header/date column; structural checks only.
        "integrity": None,
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


class DatasetIntegrityError(RuntimeError):
    """Raised when a dataset CSV on disk fails a structural integrity check."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(name: str, data_root: str | Path | None = None, strict: bool = True) -> dict[str, Any]:
    """Check a dataset CSV on disk against its registered integrity fingerprint.

    Verifies row count, date bounds, checksum (if known), and two structural
    invariants that catch the two failure modes that matter for time series:
    a pre-standardized file (near-zero column means) and a noise substitution
    (near-zero lag-1 autocorrelation). Both are cheap and need no external
    reference beyond values captured once from a known-good download.

    Returns a report dict with 'passed' and 'failures'/'warnings' lists.
    Raises DatasetIntegrityError when strict and any check fails.
    """
    spec = get_dataset_spec(name)
    integrity = spec.get("integrity")
    root = Path(data_root) if data_root is not None else Path(spec["root_path"])
    path = root / spec["data_path"]

    report: dict[str, Any] = {
        "dataset": name, "path": str(path), "passed": True, "failures": [], "warnings": [],
        "expected_sha256": integrity.get("sha256") if integrity else None,
        "actual_sha256": None,
    }

    if not path.exists():
        report["passed"] = False
        report["failures"].append(f"file not found: {path}")
        if strict:
            raise DatasetIntegrityError(report["failures"][0])
        return report

    if integrity is None:
        report["actual_sha256"] = _sha256(path)
        report["warnings"].append("no integrity fingerprint registered for this dataset")
        return report

    df = pd.read_csv(path)
    n_rows = len(df)
    if n_rows != integrity["n_rows"]:
        report["failures"].append(f"row count {n_rows} != expected {integrity['n_rows']}")

    date_col = df.columns[0]
    date_start, date_end = str(df[date_col].iloc[0]), str(df[date_col].iloc[-1])
    if date_start != integrity["date_start"] or date_end != integrity["date_end"]:
        report["failures"].append(
            f"date range [{date_start}, {date_end}] != expected "
            f"[{integrity['date_start']}, {integrity['date_end']}]"
        )

    values = df.select_dtypes("number").values
    if values.size:
        col_mean_abs = float(np.abs(values).mean())
        if col_mean_abs < integrity["min_col_mean_abs"]:
            report["failures"].append(
                f"mean(|value|)={col_mean_abs:.4f} below {integrity['min_col_mean_abs']} "
                "— file looks pre-standardized"
            )

        lag1 = [
            float(np.corrcoef(values[:-1, i], values[1:, i])[0, 1])
            for i in range(values.shape[1])
            if np.std(values[:-1, i]) > 0 and np.std(values[1:, i]) > 0
        ]
        mean_lag1 = float(np.mean(lag1)) if lag1 else 0.0
        if mean_lag1 < integrity["min_lag1_autocorr"]:
            report["failures"].append(
                f"mean lag-1 autocorrelation={mean_lag1:.4f} below {integrity['min_lag1_autocorr']} "
                "— file looks like noise, not a real time series"
            )

    actual_sha256 = _sha256(path)
    report["actual_sha256"] = actual_sha256
    if actual_sha256 != integrity["sha256"]:
        report["warnings"].append(
            f"sha256 mismatch (got {actual_sha256[:12]}…, expected {integrity['sha256'][:12]}…); "
            "file differs byte-for-byte from the known-good download but passed structural checks"
        )

    report["passed"] = not report["failures"]
    if not report["passed"] and strict:
        raise DatasetIntegrityError(f"{name}: " + "; ".join(report["failures"]))
    return report
