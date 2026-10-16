#!/usr/bin/env bash
# Frozen-checkpoint follow-up. No training, Git changes, or package installation.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$ROOT"
DATASETS=ECL,solar
CONFIG=experiments/paper_suite.yaml
OUTPUT=results/paper_cluster
NO_SLURM=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --datasets|--config|--output)
            [ "$#" -ge 2 ] || { echo "FAIL $1 requires a value" >&2; exit 2; }
            case "$1" in --datasets) DATASETS="$2";; --config) CONFIG="$2";; --output) OUTPUT="$2";; esac
            shift 2 ;;
        --no-slurm) NO_SLURM=1; shift ;;
        -h|--help) echo "Usage: bash slurm/paper_preflight.sh [--datasets ECL,solar] [--config FILE] [--output DIR] [--no-slurm]"; exit 0 ;;
        *) echo "FAIL unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -x "$PY" ] || { echo "FAIL project environment missing: $PY" >&2; exit 1; }
"$PY" -c 'import torch, numpy, pandas, sklearn, scipy, yaml, matplotlib, cissn'
for file in slurm/paper_preflight.sh slurm/launch_paper.sh slurm/run_paper.slurm; do
    bash -n "$file"
done
echo "INFO checkout $ROOT; source $(git rev-parse --short HEAD); follow-up accepts uncommitted code and records hashes"
"$PY" slurm/paper_jobs.py check --config "$CONFIG" --datasets "$DATASETS" --output "$OUTPUT"
if [ "$NO_SLURM" -eq 0 ]; then
    command -v sbatch >/dev/null || { echo "FAIL sbatch unavailable; use --no-slurm only for a workstation check" >&2; exit 1; }
    sbatch --test-only --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" --gres="$GRES" \
        --nodes=1 --cpus-per-task="$CPUS" --mem="$MEM" --time="${PAPER_TIME:-06:00:00}" --wrap=true
fi
echo "Paper preflight OK"
