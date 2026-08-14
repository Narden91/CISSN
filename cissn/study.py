"""Fail-closed study-manifest validation for publication artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from cissn.utils.artifacts import verify_completion_manifest


REQUIRED_RUN_KEYS = ("model", "dataset", "pred_len", "seed", "design_hash")
EVIDENCE_ROLES = {"selection", "confirmation"}


def load_study_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Study manifest must be a mapping.")
    if not isinstance(payload.get("study_id"), str) or not payload["study_id"]:
        raise ValueError("Study manifest requires a non-empty study_id.")
    if payload.get("evidence_role") not in EVIDENCE_ROLES:
        raise ValueError("Study manifest evidence_role must be 'selection' or 'confirmation'.")
    runs = payload.get("expected_runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Study manifest requires a non-empty expected_runs list.")
    signatures = [study_run_signature(run) for run in runs]
    if len(signatures) != len(set(signatures)):
        raise ValueError("Study manifest has duplicate expected_runs entries.")
    return payload


def study_run_signature(run: dict[str, Any]) -> tuple[str, str, int, int, str]:
    missing = [key for key in REQUIRED_RUN_KEYS if key not in run]
    if missing:
        raise ValueError(f"Study run is missing: {', '.join(missing)}")
    return (
        str(run["model"]),
        str(run["dataset"]),
        int(run["pred_len"]),
        int(run["seed"]),
        str(run["design_hash"]),
    )


def validate_study_results(results_root: str | Path, study_manifest: dict[str, Any]) -> set[Path]:
    root = Path(results_root)
    expected = {study_run_signature(run) for run in study_manifest["expected_runs"]}
    observed: dict[tuple[str, str, int, int, str], Path] = {}
    metrics_paths = list(root.glob("**/metrics.json"))
    if not metrics_paths:
        raise RuntimeError(f"No metrics artifacts found under {root}.")

    for metrics_path in metrics_paths:
        run_dir = metrics_path.parent
        completion = verify_completion_manifest(run_dir)
        protocol_path = run_dir / "protocol.json"
        config_path = run_dir / "config.json"
        if not protocol_path.is_file() or not config_path.is_file():
            raise RuntimeError(f"Run is missing protocol/config: {run_dir}")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if protocol.get("design_hash") != completion["design_hash"]:
            raise RuntimeError(f"Protocol/design hash mismatch: {run_dir}")
        if protocol.get("config", {}).get("evidence_role") != study_manifest["evidence_role"]:
            raise RuntimeError(f"Evidence role mismatch: {run_dir}")
        signature = study_run_signature(
            {
                "model": metrics.get("model", config.get("model", "cissn")),
                "dataset": config.get("data"),
                "pred_len": config.get("pred_len"),
                "seed": config.get("seed"),
                "design_hash": completion["design_hash"],
            }
        )
        if signature not in expected:
            raise RuntimeError(f"Unexpected study run: {run_dir}")
        if signature in observed:
            raise RuntimeError(f"Duplicate study run: {run_dir} and {observed[signature]}")
        _require_finite_metrics(metrics, run_dir)
        observed[signature] = run_dir

    missing = expected - set(observed)
    if missing:
        raise RuntimeError(f"Study is incomplete; missing {len(missing)} expected run(s).")
    return set(observed.values())


def _require_finite_metrics(metrics: dict[str, Any], run_dir: Path) -> None:
    point = metrics.get("point", {})
    for key in ("mse", "mae", "rmse"):
        value = point.get(key)
        if not isinstance(value, (int, float)) or not float("-inf") < float(value) < float("inf"):
            raise RuntimeError(f"Run has non-finite point metric {key}: {run_dir}")
