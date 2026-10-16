#!/usr/bin/env bash
# One GPU job per dataset. Submit only after the frozen-resource check succeeds.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$ROOT"
DATASETS=ECL,solar
CONFIG=experiments/paper_suite.yaml
OUTPUT=results/paper_cluster
DRY_RUN=0
NO_SLURM=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --datasets|--config|--output)
            [ "$#" -ge 2 ] || { echo "ERROR $1 requires a value" >&2; exit 2; }
            case "$1" in --datasets) DATASETS="$2";; --config) CONFIG="$2";; --output) OUTPUT="$2";; esac
            shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --no-slurm) NO_SLURM=1; shift ;;
        -h|--help) echo "Usage: bash slurm/launch_paper.sh [--datasets ECL,solar] [--config FILE] [--output DIR] [--dry-run [--no-slurm]]"; exit 0 ;;
        *) echo "ERROR unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [ "$NO_SLURM" -eq 1 ] && [ "$DRY_RUN" -ne 1 ]; then
    echo "ERROR --no-slurm is allowed only with --dry-run" >&2; exit 2
fi
PREFLIGHT_ARGS=(--config "$CONFIG" --datasets "$DATASETS" --output "$OUTPUT")
[ "$NO_SLURM" -eq 1 ] && PREFLIGHT_ARGS+=(--no-slurm)
bash slurm/paper_preflight.sh "${PREFLIGHT_ARGS[@]}"
IFS=, read -r -a NAMES <<< "$DATASETS"
if [ "$DRY_RUN" -eq 0 ]; then
    active=$(squeue -h -u "$(id -un)" -o '%j')
    for dataset in "${NAMES[@]}"; do
        if printf '%s\n' "$active" | grep -Fxq "cissn_paper_$dataset"; then
            echo "ERROR cissn_paper_$dataset is already queued or running; wait before reusing its output" >&2
            exit 1
        fi
    done
    mkdir -p "$LOGS"
    "$PY" slurm/paper_jobs.py prepare --config "$CONFIG" --datasets "$DATASETS" --output "$OUTPUT" --job-configs "$OUTPUT/job_configs"
fi
for dataset in "${NAMES[@]}"; do
    cmd=(sbatch --parsable --job-name="cissn_paper_$dataset" --account="$ACCOUNT" --partition="$PARTITION"
        --qos="$QOS" --gres="$GRES" --cpus-per-task="$CPUS" --mem="$MEM" --time="${PAPER_TIME:-06:00:00}"
        --export="ALL,ROOT=$ROOT,PAPER_CONFIG=$OUTPUT/job_configs/$dataset.yaml"
        --output="$LOGS/%x_%j.out" slurm/run_paper.slurm)
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%q ' "${cmd[@]}"; printf '\n'
    else
        id=$("${cmd[@]}")
        echo "$dataset submitted: $id"
    fi
done
echo "After all jobs succeed: $PY slurm/paper_jobs.py merge --config $CONFIG --datasets $DATASETS --output $OUTPUT"
echo "Then: $PY manuscript/analysis/build_strengthening.py --results $OUTPUT --prefix cluster"
