#!/usr/bin/env python
"""Generate a reproducibility appendix markdown from available artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _collect_run_dirs(results_root: Path) -> list[Path]:
    return sorted({p.parent for p in results_root.glob("**/metrics.json")})


def _summarize_environments(run_dirs: list[Path]) -> dict[str, Any]:
    envs = []
    for run in run_dirs:
        payload = _safe_read_json(run / "environment.json")
        if payload:
            envs.append(payload)
    if not envs:
        return {"count": 0}
    device_counter = Counter(str(e.get("device")) for e in envs)
    torch_counter = Counter(str(e.get("torch")) for e in envs)
    python_counter = Counter(str(e.get("python", "")).split(" ")[0] for e in envs)
    commits = Counter(str(e.get("git_commit")) for e in envs if e.get("git_commit"))
    return {
        "count": len(envs),
        "devices": dict(device_counter),
        "torch_versions": dict(torch_counter),
        "python_versions": dict(python_counter),
        "git_commits": dict(commits),
    }


def _artifact_coverage(run_dirs: list[Path]) -> dict[str, int]:
    required = [
        "metrics.json",
        "config.json",
        "environment.json",
        "runtime.json",
        "history.json",
        "sanity.json",
        "protocol.json",
        "pred.npy",
        "true.npy",
    ]
    interval = ["lower.npy", "upper.npy"]
    counts = {k: 0 for k in required + interval}
    for run in run_dirs:
        for name in counts:
            if (run / name).exists():
                counts[name] += 1
    counts["runs"] = len(run_dirs)
    return counts


def _write_appendix(
    output_path: Path,
    results_root: Path,
    run_dirs: list[Path],
    env_summary: dict[str, Any],
    artifact_summary: dict[str, int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_runs = [str(p) for p in run_dirs[:15]]

    lines = [
        "# Reproducibility Appendix (Auto-Generated)",
        "",
        "This appendix is generated from experiment artifacts present in the repository.",
        "",
        "## Scope",
        "",
        f"- Artifact root: `{results_root}`",
        f"- Runs discovered (via `metrics.json`): **{len(run_dirs)}**",
        "",
        "## Environment Summary",
        "",
        f"- Environment snapshots discovered: **{env_summary.get('count', 0)}**",
        f"- Devices: `{env_summary.get('devices', {})}`",
        f"- Torch versions: `{env_summary.get('torch_versions', {})}`",
        f"- Python versions: `{env_summary.get('python_versions', {})}`",
        f"- Git commits in artifacts: `{env_summary.get('git_commits', {})}`",
        "",
        "## Artifact Contract Coverage",
        "",
        f"- Runs inspected: **{artifact_summary.get('runs', 0)}**",
        f"- `metrics.json`: {artifact_summary.get('metrics.json', 0)}",
        f"- `config.json`: {artifact_summary.get('config.json', 0)}",
        f"- `environment.json`: {artifact_summary.get('environment.json', 0)}",
        f"- `runtime.json`: {artifact_summary.get('runtime.json', 0)}",
        f"- `history.json`: {artifact_summary.get('history.json', 0)}",
        f"- `sanity.json`: {artifact_summary.get('sanity.json', 0)}",
        f"- `protocol.json`: {artifact_summary.get('protocol.json', 0)}",
        f"- `pred.npy`: {artifact_summary.get('pred.npy', 0)}",
        f"- `true.npy`: {artifact_summary.get('true.npy', 0)}",
        f"- `lower.npy`: {artifact_summary.get('lower.npy', 0)}",
        f"- `upper.npy`: {artifact_summary.get('upper.npy', 0)}",
        "",
        "## Execution Commands",
        "",
        "Canonical commands are in `RUNBOOK.md`:",
        "",
        "- Gates 0, 1, and 2",
        "- Main CISSN, baseline, and ablation grids",
        "",
        "## Sample Run Paths",
        "",
    ]
    if sample_runs:
        lines.extend([f"- `{p}`" for p in sample_runs])
    else:
        lines.append("- No run paths detected yet.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This appendix captures currently present artifacts only.",
            "- Re-run this generator after each major experiment phase to keep counts up to date.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducibility appendix markdown.")
    parser.add_argument("--results_root", type=Path, default=Path("./results"))
    parser.add_argument("--output", type=Path, default=Path("./manuscript/reproducibility_appendix.md"))
    args = parser.parse_args()

    run_dirs = _collect_run_dirs(args.results_root)
    env_summary = _summarize_environments(run_dirs)
    artifact_summary = _artifact_coverage(run_dirs)

    _write_appendix(
        output_path=args.output,
        results_root=args.results_root,
        run_dirs=run_dirs,
        env_summary=env_summary,
        artifact_summary=artifact_summary,
    )

    summary = {
        "output": str(args.output),
        "runs_detected": len(run_dirs),
        "env_snapshots": env_summary.get("count", 0),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

