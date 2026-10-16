"""Check resources, write dataset job configs, and merge follow-up summaries."""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from experiments.run_paper_suite import inventory, sha256, write_json


def selected_settings(config, datasets, output):
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    selected = datasets.split(",") if datasets else settings["datasets"]
    unknown = set(selected) - set(settings["datasets"])
    if unknown or len(set(selected)) != len(selected) or not selected:
        raise ValueError(f"Invalid dataset selection: {selected}.")
    settings["datasets"] = selected
    settings["output_root"] = str(output)
    return settings


def check(settings):
    from cissn.data.registry import get_dataset_spec
    import torch

    failures = []
    runs = inventory(settings)
    index = defaultdict(list)
    for run, config, model in runs:
        index[(config["data"], int(config["pred_len"]), int(config["seed"]), model)].append((run, config))
    required_files = set()
    total_bytes = 0
    for dataset in settings["datasets"]:
        # Solar has no point-baseline publication runs. This absence is explicit.
        models = ["cissn"] if dataset == "solar" else settings["models"]
        spec = get_dataset_spec(dataset)
        for horizon in settings["horizons"]:
            for seed in settings["seeds"]:
                for model in models:
                    key = (dataset, horizon, seed, model)
                    matches = index[key]
                    if len(matches) != 1:
                        failures.append(f"{key}: expected one result config, found {len(matches)}")
                        continue
                    run, config = matches[0]
                    for name in (("checkpoint.pth", "checkpoint_head.pth", "checkpoint_revin.pth") if model == "cissn" else ("checkpoint.pth", "checkpoint_revin.pth")):
                        file = Path(settings["checkpoints_root"]) / run.name / name
                        required_files.add(file)
                        if not file.is_file() or file.stat().st_size == 0:
                            failures.append(f"Missing or empty weight: {file}")
                    for name in ("test_origin_indices.npy", "per_origin_interval_metrics.npy", "per_origin_interval_flat_cp_metrics.npy") if model == "cissn" else ():
                        if not (run / name).is_file():
                            failures.append(f"Missing original reproduction evidence: {run / name}")
                    if config.get("multivariate_strategy") != "per_feature":
                        failures.append(f"{run.name}: requires cellwise per_feature intervals")
                    if config.get("calibration_stride", 1) < 1:
                        failures.append(f"{run.name}: invalid calibration stride")
                    n = spec.get("integrity", {}).get("n_rows") if spec.get("integrity") else 52560
                    nominal_train = 12 * 30 * 24 if dataset.startswith("ETTh") else int(n * 0.7)
                    n_cal = round(config.get("cal_fraction", 0.2) * nominal_train) - horizon + 1
                    nominal_test = 4 * 30 * 24 if dataset.startswith("ETTh") else n - int(n * 0.7) - int(n * 0.1)
                    n_test = (nominal_test - horizon) // settings["test_strides"][dataset] + 1
                    total_bytes += 4 * (n_cal * (horizon * spec["c_out"] + 5) + n_test * (3 * horizon * spec["c_out"] + 5))
        data_path = Path(spec["root_path"]) / spec["data_path"]
        if not data_path.is_file():
            failures.append(f"Missing dataset: {data_path}")
            continue
        observed = sha256(data_path)
        expected = {c.get("protocol", {}).get("dataset", {}).get("actual_sha256") for _, c, _ in runs if c["data"] == dataset}
        expected.discard(None)
        if not expected and spec.get("integrity"):
            expected.add(spec["integrity"]["sha256"])
        if not expected:
            failures.append(f"{dataset}: no recorded dataset checksum in frozen run configs")
        elif expected != {observed}:
            failures.append(f"{dataset}: dataset SHA-256 disagrees with frozen run(s): {observed}")
        else:
            print(f"OK dataset {dataset}: {observed}")
    source_files = ["experiments/run_paper_suite.py", "experiments/paper_suite.yaml", "cissn/conformal/phase.py",
                    "slurm/paper_jobs.py", "slurm/paper_preflight.sh", "slurm/launch_paper.sh", "slurm/run_paper.slurm",
                    "manuscript/analysis/build_strengthening.py"]
    for name in source_files:
        file = Path(name)
        if not file.is_file():
            failures.append(f"Missing source: {name}")
        elif name.endswith((".sh", ".slurm")) and b"\r\n" in file.read_bytes():
            failures.append(f"Shell source must use LF line endings: {name}")
        elif name.endswith(".py"):
            ast.parse(file.read_text(encoding="utf-8"), filename=name)
    if torch.version.cuda is None:
        failures.append("Project torch is a CPU build; a CUDA build is required on compute nodes")
    output = Path(settings["output_root"])
    parent = output
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    required = int(total_bytes * 1.5) + (2 << 30)
    print(f"INFO cache estimate {total_bytes / 2**30:.1f} GiB; required free with headroom {required / 2**30:.1f} GiB; free {free / 2**30:.1f} GiB")
    if free < required:
        failures.append("Not enough disk space for prediction caches and temporary quantile arrays")
    print(f"INFO checked {len(required_files)} expected weight paths; torch {torch.__version__}; login-node GPU visibility is not required")
    if "solar" in settings["datasets"]:
        print("INFO Solar jobs use CISSN only: no Solar DLinear/PatchTST publication configurations exist")
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        return 1
    print("OK all selected frozen-run resources are present")
    return 0


def prepare(settings, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for dataset in settings["datasets"]:
        child = {**settings, "datasets": [dataset], "models": ["cissn"] if dataset == "solar" else settings["models"],
                 "output_root": str(Path(settings["output_root"]) / dataset)}
        (directory / f"{dataset}.yaml").write_text(yaml.safe_dump(child, sort_keys=False), encoding="utf-8")
    (directory / "datasets.txt").write_text("\n".join(settings["datasets"]) + "\n", encoding="utf-8")


def merge(settings):
    output = Path(settings["output_root"])
    inventory_rows = []
    tables = defaultdict(list)
    for dataset in settings["datasets"]:
        child = output / dataset
        records = json.loads((child / "inventory.json").read_text(encoding="utf-8"))
        n_models = 1 if dataset == "solar" else len(settings["models"])
        expected = len(settings["horizons"]) * len(settings["seeds"]) * n_models
        if len(records) != expected or any(r["status"] != "completed" for r in records):
            raise ValueError(f"{dataset}: incomplete jobs; do not build a final report")
        inventory_rows.extend(records)
        for name in ("runs.csv", "summary.csv", "uncertainty.csv", "phase_effects.csv"):
            tables[name].append(pd.read_csv(child / name))
    for name, frames in tables.items():
        pd.concat(frames, ignore_index=True).to_csv(output / name, index=False)
    write_json(output / "inventory.json", inventory_rows)
    print(f"Merged {len(inventory_rows)} completed frozen runs: {dict(Counter(r['dataset'] for r in inventory_rows))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "prepare", "merge"))
    parser.add_argument("--config", type=Path, default=Path("experiments/paper_suite.yaml"))
    parser.add_argument("--datasets", default="ECL,solar")
    parser.add_argument("--output", type=Path, default=Path("results/paper_cluster"))
    parser.add_argument("--job-configs", type=Path, default=Path("results/paper_cluster/job_configs"))
    args = parser.parse_args()
    settings = selected_settings(args.config, args.datasets, args.output)
    if args.action == "check":
        raise SystemExit(check(settings))
    if args.action == "prepare":
        prepare(settings, args.job_configs)
    else:
        merge(settings)


if __name__ == "__main__":
    main()
