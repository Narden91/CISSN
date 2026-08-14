import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from cissn.study import load_study_manifest, validate_study_results
from cissn.utils.artifacts import (
    canonical_hash,
    create_temporary_result_root,
    finalize_result_directory,
    require_new_run,
    verify_completion_manifest,
    write_completion_manifest,
)


class TestArtifactContracts(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.run_dir = Path(self.tmpdir.name) / "run"
        self.run_dir.mkdir()
        (self.run_dir / "metrics.json").write_text(json.dumps({"mse": 1.0}), encoding="utf-8")
        np.save(self.run_dir / "pred.npy", np.arange(6, dtype=np.float32).reshape(2, 3))
        self.protocol = {
            "design_hash": "design", "protocol_hash": "protocol",
            "config": {"evidence_role": "confirmation"},
        }

    def test_completion_manifest_verifies_hashes_and_array_metadata(self):
        write_completion_manifest(self.run_dir, ["metrics.json", "pred.npy"], self.protocol)

        manifest = verify_completion_manifest(self.run_dir)

        self.assertEqual(manifest["design_hash"], "design")
        self.assertEqual(manifest["files"][1]["shape"], [2, 3])

    def test_completion_manifest_rejects_mutated_artifact(self):
        write_completion_manifest(self.run_dir, ["metrics.json", "pred.npy"], self.protocol)
        (self.run_dir / "metrics.json").write_text(json.dumps({"mse": 2.0}), encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            verify_completion_manifest(self.run_dir)

    def test_new_run_rejects_existing_checkpoint_or_result(self):
        checkpoint = Path(self.tmpdir.name) / "checkpoint"
        result = Path(self.tmpdir.name) / "result"
        checkpoint.mkdir()

        with self.assertRaisesRegex(FileExistsError, "Immutable run already exists"):
            require_new_run(checkpoint, result)

    def test_finalize_moves_only_completed_temporary_result(self):
        final_root = Path(self.tmpdir.name) / "results"
        temporary_root = create_temporary_result_root(final_root)
        temporary_run = temporary_root / "setting"
        temporary_run.mkdir(parents=True)
        (temporary_run / "completion.json").write_text("{}", encoding="utf-8")

        finalized = finalize_result_directory(temporary_root, final_root, "setting")

        self.assertEqual(finalized, final_root / "setting")
        self.assertTrue((finalized / "completion.json").is_file())
        self.assertFalse(temporary_run.exists())

    def test_hash_changes_when_scientific_config_changes(self):
        first = canonical_hash({"dropout": 0.05, "n_clusters": 5})
        second = canonical_hash({"dropout": 0.40, "n_clusters": 5})

        self.assertNotEqual(first, second)

    def test_study_validator_requires_the_exact_completed_run_set(self):
        metrics = {
            "model": "dlinear",
            "point": {"mse": 1.0, "mae": 0.5, "rmse": 1.0},
        }
        config = {"data": "ETTh2", "pred_len": 96, "seed": 42}
        protocol = {
            "design_hash": "design", "protocol_hash": "protocol",
            "config": {"evidence_role": "confirmation"},
        }
        for name, payload in (("metrics.json", metrics), ("config.json", config), ("protocol.json", protocol)):
            (self.run_dir / name).write_text(json.dumps(payload), encoding="utf-8")
        write_completion_manifest(
            self.run_dir, ["metrics.json", "config.json", "protocol.json", "pred.npy"], protocol
        )
        manifest_path = Path(self.tmpdir.name) / "study.yaml"
        manifest_path.write_text(yaml.safe_dump({
            "study_id": "unit", "evidence_role": "confirmation",
            "expected_runs": [{
                "model": "dlinear", "dataset": "ETTh2", "pred_len": 96,
                "seed": 42, "design_hash": "design",
            }],
        }), encoding="utf-8")

        approved = validate_study_results(self.tmpdir.name, load_study_manifest(manifest_path))

        self.assertEqual(approved, {self.run_dir})

    def test_study_validator_rejects_incomplete_artifact(self):
        manifest_path = Path(self.tmpdir.name) / "study.yaml"
        manifest_path.write_text(yaml.safe_dump({
            "study_id": "unit", "evidence_role": "confirmation",
            "expected_runs": [{
                "model": "dlinear", "dataset": "ETTh2", "pred_len": 96,
                "seed": 42, "design_hash": "design",
            }],
        }), encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "Missing completion manifest"):
            validate_study_results(self.tmpdir.name, load_study_manifest(manifest_path))


if __name__ == "__main__":
    unittest.main()
