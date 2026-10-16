#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
set -euo pipefail
ROOT="${ROOT:?Submit with bash slurm/launch_paper.sh}"
PAPER_CONFIG="${PAPER_CONFIG:?Missing dataset job config}"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
export PYTHONUNBUFFERED=1
export CISSN_REQUIRE_GPU=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "Start UTC: $(date -u +%FT%TZ); job ${SLURM_JOB_ID:-manual}; source $(git rev-parse HEAD)"
nvidia-smi -L
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is not visible on compute node"'
"$PY" -m experiments.run_paper_suite --config "$PAPER_CONFIG"
"$PY" - "$PAPER_CONFIG" <<'PY'
import json, sys
from pathlib import Path
import yaml
settings = yaml.safe_load(Path(sys.argv[1]).read_text())
records = json.loads((Path(settings["output_root"]) / "inventory.json").read_text())
expected = len(settings["horizons"]) * len(settings["seeds"]) * len(settings["models"])
assert len(records) == expected and all(r["status"] == "completed" for r in records), "Incomplete frozen-run suite; read inventory.json"
PY
echo "End UTC: $(date -u +%FT%TZ)"
