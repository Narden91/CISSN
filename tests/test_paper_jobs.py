"""The cluster checker must reject missing or mismatched frozen resources."""
import json
from pathlib import Path

import pandas as pd
import pytest

from slurm import paper_jobs


@pytest.fixture
def resources(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("date,OT\n2020-01-01,1\n")
    run = tmp_path / "run"
    run.mkdir()
    for name in ("test_origin_indices.npy", "per_origin_interval_metrics.npy", "per_origin_interval_flat_cp_metrics.npy"):
        (run / name).write_bytes(b"evidence")
    weights = tmp_path / "weights" / run.name
    weights.mkdir(parents=True)
    for name in ("checkpoint.pth", "checkpoint_head.pth", "checkpoint_revin.pth"):
        (weights / name).write_bytes(b"frozen")
    config = {"data": "ETTh2", "pred_len": 96, "seed": 42, "multivariate_strategy": "per_feature",
              "protocol": {"dataset": {"actual_sha256": "recorded"}}}
    spec = {"root_path": str(tmp_path), "data_path": dataset.name, "c_out": 7,
            "integrity": {"n_rows": 17420, "sha256": "recorded"}}
    settings = {"datasets": ["ETTh2"], "horizons": [96], "seeds": [42], "models": ["cissn"],
                "checkpoints_root": str(tmp_path / "weights"), "output_root": str(tmp_path / "output"),
                "test_strides": {"ETTh2": 1}}
    monkeypatch.setattr(paper_jobs, "inventory", lambda _: [(run, config, "cissn")])
    monkeypatch.setattr(paper_jobs, "sha256", lambda _: "recorded")
    monkeypatch.setattr("cissn.data.registry.get_dataset_spec", lambda _: spec)
    return settings, weights, run


def test_preflight_accepts_complete_resources(resources):
    settings, _, _ = resources
    assert paper_jobs.check(settings) == 0


def test_preflight_rejects_missing_head(resources, capsys):
    settings, weights, _ = resources
    (weights / "checkpoint_head.pth").unlink()
    assert paper_jobs.check(settings) == 1
    assert "Missing or empty weight" in capsys.readouterr().out


def test_preflight_rejects_empty_revin_weights(resources):
    settings, weights, _ = resources
    (weights / "checkpoint_revin.pth").write_bytes(b"")
    assert paper_jobs.check(settings) == 1


def test_preflight_rejects_dataset_mismatch(resources, monkeypatch, capsys):
    settings, _, _ = resources
    monkeypatch.setattr(paper_jobs, "sha256", lambda _: "changed")
    assert paper_jobs.check(settings) == 1
    assert "dataset SHA-256 disagrees" in capsys.readouterr().out


def test_preflight_rejects_duplicate_configs(resources, monkeypatch):
    settings, _, run = resources
    config = {"data": "ETTh2", "pred_len": 96, "seed": 42}
    monkeypatch.setattr(paper_jobs, "inventory", lambda _: [(run, config, "cissn"), (run, config, "cissn")])
    assert paper_jobs.check(settings) == 1


def test_prepare_keeps_outputs_separate_and_solar_model_scope_explicit(tmp_path):
    settings = {"datasets": ["ECL", "solar"], "models": ["cissn", "dlinear", "patchtst"], "output_root": str(tmp_path)}
    folder = tmp_path / "configs"
    paper_jobs.prepare(settings, folder)
    import yaml
    ecl = yaml.safe_load((folder / "ECL.yaml").read_text())
    solar = yaml.safe_load((folder / "solar.yaml").read_text())
    assert ecl["output_root"] != solar["output_root"]
    assert solar["models"] == ["cissn"]
    assert ecl["models"] == settings["models"]


def test_merge_rejects_incomplete_job(tmp_path):
    folder = tmp_path / "ECL"
    folder.mkdir()
    (folder / "inventory.json").write_text(json.dumps([{"status": "blocked"}]))
    settings = {"datasets": ["ECL"], "horizons": [96], "seeds": [42], "models": ["cissn"], "output_root": str(tmp_path)}
    with pytest.raises(ValueError, match="incomplete jobs"):
        paper_jobs.merge(settings)


def test_merge_keeps_all_dataset_rows(tmp_path):
    for dataset in ("ECL", "solar"):
        folder = tmp_path / dataset
        folder.mkdir()
        (folder / "inventory.json").write_text(json.dumps([{"status": "completed", "dataset": dataset}]))
        for name in ("runs.csv", "summary.csv", "uncertainty.csv", "phase_effects.csv"):
            pd.DataFrame([{"dataset": dataset, "value": 1}]).to_csv(folder / name, index=False)
    settings = {"datasets": ["ECL", "solar"], "horizons": [96], "seeds": [42], "models": ["cissn"], "output_root": str(tmp_path)}
    paper_jobs.merge(settings)
    assert pd.read_csv(tmp_path / "summary.csv").dataset.tolist() == ["ECL", "solar"]
