"""Immutable result-artifact helpers for locked experiment studies."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_record(path: Path, root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r")
        record["shape"] = list(array.shape)
        record["dtype"] = str(array.dtype)
    return record


def write_completion_manifest(
    run_dir: str | Path,
    required_files: Iterable[str],
    protocol: dict[str, Any],
) -> Path:
    root = Path(run_dir)
    required = sorted(set(required_files))
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"Cannot finalize incomplete run {root}: missing {', '.join(missing)}")
    files = [build_file_record(root / name, root) for name in required]
    payload = {
        "design_hash": protocol.get("design_hash"),
        "protocol_hash": protocol.get("protocol_hash"),
        "required_files": required,
        "files": files,
    }
    path = root / "completion.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def verify_completion_manifest(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    path = root / "completion.json"
    if not path.is_file():
        raise RuntimeError(f"Missing completion manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("design_hash") or not manifest.get("protocol_hash"):
        raise RuntimeError(f"Completion manifest has no provenance hash: {path}")
    for record in manifest.get("files", []):
        artifact = root / record["path"]
        if not artifact.is_file():
            raise RuntimeError(f"Completion artifact is missing: {artifact}")
        if sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"Completion artifact hash mismatch: {artifact}")
        if artifact.suffix == ".npy":
            array = np.load(artifact, mmap_mode="r")
            if list(array.shape) != record.get("shape") or str(array.dtype) != record.get("dtype"):
                raise RuntimeError(f"Completion artifact array metadata mismatch: {artifact}")
    return manifest


def require_new_run(checkpoint_dir: str | Path, result_dir: str | Path) -> None:
    existing = [str(path) for path in (Path(checkpoint_dir), Path(result_dir)) if path.exists()]
    if existing:
        raise FileExistsError("Immutable run already exists: " + ", ".join(existing))


def create_temporary_result_root(final_root: str | Path) -> Path:
    final = Path(final_root)
    return final.parent / f".{final.name}.tmp-{uuid4().hex}"


def finalize_result_directory(temp_root: str | Path, final_root: str | Path, setting: str) -> Path:
    temporary = Path(temp_root) / setting
    final = Path(final_root) / setting
    if not temporary.is_dir():
        raise RuntimeError(f"Temporary result directory does not exist: {temporary}")
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite completed result directory: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, final)
    return final
