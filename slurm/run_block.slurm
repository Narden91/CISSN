#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# One array task of one block (see slurm/cells.py). launch.sh sets the job name, account,
# partition, GPU, time, array range, dependency, output and the exports ROOT, BLOCK, NTASKS.
# Submit through "bash slurm/launch.sh", never with sbatch directly.
set -euo pipefail

ROOT="${ROOT:?ROOT is not set; submit with slurm/launch.sh}"
BLOCK="${BLOCK:?BLOCK is not set; submit with slurm/launch.sh}"
NTASKS="${NTASKS:?NTASKS is not set; submit with slurm/launch.sh}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
PY="$ROOT/.venv/bin/python"

cd "$ROOT"
export PYTHONUNBUFFERED=1
export CISSN_REQUIRE_GPU=1                       # a silent CPU fallback would run for days
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "================================================"
echo "Block:       $BLOCK  task $TASK of $NTASKS"
echo "Job ID:      ${SLURM_JOB_ID:-manual}  array job: ${SLURM_ARRAY_JOB_ID:-none}"
echo "Node:        ${SLURM_NODELIST:-localhost}"
echo "Commit:      $(git rev-parse HEAD)  ($(git rev-parse --abbrev-ref HEAD))"
echo "Python:      $PY"
nvidia-smi -L || true
echo "Start (UTC): $(date -u +%FT%TZ)"
echo "================================================"

# Fail fast: every cell would fail on these, and each failure costs a data load.
[ -x "$PY" ] || { echo "ERROR: $PY missing; run slurm/setup_cluster.sh on the login node" >&2; exit 4; }
DIRTY=$(git status --porcelain)
[ -z "$DIRTY" ] || { echo "ERROR: git tree not clean (--require_clean_git would reject every run):" >&2; echo "$DIRTY" >&2; exit 3; }
"$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || { echo "ERROR: torch cannot see a GPU on this node" >&2; exit 5; }

CODE=0
"$PY" slurm/cells.py run "$BLOCK" --task "$TASK" --ntasks "$NTASKS" || CODE=$?
echo "End (UTC): $(date -u +%FT%TZ)  exit $CODE"
exit "$CODE"
