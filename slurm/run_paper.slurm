#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Edit account, partition, and resources in the untracked .env; submit with launch_paper.sh.
# Empty GRES means a CPU job: do not require CUDA. A GPU request still fails without a device.
set -euo pipefail
ROOT="${ROOT:?Submit with bash slurm/launch_paper.sh}"
PAPER_CONFIG="${PAPER_CONFIG:?Missing dataset job config}"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "Start UTC: $(date -u +%FT%TZ); job ${SLURM_JOB_ID:-manual}; source $(git rev-parse HEAD)"
if [ -n "${GRES:-}" ]; then
    export CISSN_REQUIRE_GPU=1
    nvidia-smi -L
    "$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is not visible on compute node"'
else
    unset CISSN_REQUIRE_GPU
    export CUDA_VISIBLE_DEVICES=""
    echo "CPU job: GRES is empty, so CUDA is not required"
    "$PY" -c 'import torch; print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())'
fi
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
