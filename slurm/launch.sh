#!/usr/bin/env bash
# Submits the cluster runs R1-R6 (experiments.md) after preflight passes. Re-run it to resume:
# finished cells are skipped, and blocks with nothing left are not submitted.
#
#   bash slurm/launch.sh                  # preflight, smoke job, then all blocks
#   bash slurm/launch.sh --dry-run        # preflight and print the sbatch commands
#   bash slurm/launch.sh --smoke-only     # preflight and the 1-epoch GPU smoke job only
#   bash slurm/launch.sh --blocks r4,r4b  # a subset
#
# Order: smoke -> (r1, r2, r4) -> r3/r5 after r1 -> r4b after r4 -> r6 after r2-r5. The smoke job
# gates everything with afterok; between blocks the order uses afterany, so one failed cell does not
# strand the rest. R3 and R5 depend on the R1 outcome: only one of them has cells.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$ROOT"

DRY_RUN=0
SMOKE_ONLY=0
BLOCKS=(r1 r2 r3 r4 r4b r5 r6)

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --smoke-only) SMOKE_ONLY=1 ;;
        --blocks) [ $# -ge 2 ] || { echo "ERROR: --blocks needs a value" >&2; exit 2; }; IFS=, read -r -a BLOCKS <<< "$2"; shift ;;
        --throttle) [ $# -ge 2 ] || { echo "ERROR: --throttle needs a value" >&2; exit 2; }; THROTTLE="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
for b in "${BLOCKS[@]}"; do
    case "$b" in r1|r2|r3|r4|r4b|r5|r6) ;; *) echo "ERROR: unknown block '$b' (r1 r2 r3 r4 r4b r5 r6)" >&2; exit 2 ;; esac
done

PREFLIGHT_ARGS=()
if [ "$DRY_RUN" -eq 1 ] && ! command -v sbatch >/dev/null 2>&1; then PREFLIGHT_ARGS=(--no-slurm); fi
bash slurm/preflight.sh ${PREFLIGHT_ARGS[@]+"${PREFLIGHT_ARGS[@]}"} || { echo "ERROR: preflight failed; nothing submitted" >&2; exit 1; }
echo

mkdir -p "$LOGS"

upper() { echo "$1" | tr '[:lower:]' '[:upper:]'; }

# submit BLOCK ARRAY_SIZE DEPENDENCY -> prints the job id
submit() {
    local block="$1" size="$2" dep="$3" time_var
    time_var="TIME_$(upper "$block")"
    local cmd=(sbatch --parsable --job-name="cissn_$block" --account="$ACCOUNT" --partition="$PARTITION"
        --qos="$QOS" --gres="$GRES" --cpus-per-task="$CPUS" --mem="$MEM" --time="${!time_var}"
        --array="0-$((size - 1))%$THROTTLE" --mail-type=FAIL --mail-user="$MAIL_USER"
        --export="ALL,ROOT=$ROOT,BLOCK=$block,NTASKS=$size"
        --output="$LOGS/%x_%A_%a.out")
    [ -n "$dep" ] && cmd+=(--dependency="$dep")
    cmd+=(slurm/run_block.slurm)
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%q ' "${cmd[@]}" >&2; printf '\n' >&2
        echo "DRYRUN_${block}"
    else
        "${cmd[@]}"
    fi
}

# Tasks for a block: the configured shards, capped by the cells left. Prints 0 (skip) or a count.
# r3/r5 print "undecided" while R1 is unfinished; they then get the configured shards.
tasks_for() {
    local block="$1" todo shards_var shards
    shards_var="SHARDS_$(upper "$block")"; shards="${!shards_var}"
    todo=$("$PY" slurm/cells.py todo "$block" 2>/dev/null | tail -1)
    if [ "$todo" = "undecided" ]; then echo "$shards"; return; fi
    if [ "$todo" -lt "$shards" ]; then echo "$todo"; else echo "$shards"; fi
}

declare -A IDS
SMOKE_ID=$(submit smoke 1 "")
echo "smoke   job $SMOKE_ID  (1 epoch each: ECL selection x2, ETTh1 benchmark, dlinear, deep_ensemble)"
GATE="afterok:$SMOKE_ID"
if [ "$SMOKE_ONLY" -eq 1 ]; then
    echo "Smoke only. Watch: tail -f $LOGS/cissn_smoke_${SMOKE_ID}_0.out"
    exit 0
fi

for block in "${BLOCKS[@]}"; do
    n=$(tasks_for "$block")
    if [ "$n" -eq 0 ]; then echo "$block: skipped (all cells done, or excluded by the R1 outcome)"; continue; fi
    dep="$GATE"
    case "$block" in
        r3|r5) [ -n "${IDS[r1]:-}" ] && dep="$GATE,afterany:${IDS[r1]}" ;;
        r4b)   [ -n "${IDS[r4]:-}" ] && dep="$GATE,afterany:${IDS[r4]}" ;;
        r6)    after=""
               for b in r2 r3 r4 r5; do [ -n "${IDS[$b]:-}" ] && after="$after:${IDS[$b]}"; done
               [ -n "$after" ] && dep="$GATE,afterany$after" ;;
    esac
    IDS[$block]=$(submit "$block" "$n" "$dep")
    printf '%-7s job %s  (%s tasks, dependency %s)\n' "$block" "${IDS[$block]}" "$n" "$dep"
done

if [ "$DRY_RUN" -eq 1 ]; then echo; echo "Dry run: nothing was submitted."; exit 0; fi

cat <<EOF

Submitted. If the smoke job fails, cancel the rest:
  squeue -h -u \$(id -un) -o '%i %j' | awk '\$2 ~ /^cissn_/ {print \$1}' | xargs scancel
  queue:        squeue -u \$(id -un)
  progress:     $PY slurm/cells.py status
  logs:         $LOGS/   (failed cells: $LOGS/failed_<block>.txt)
  resume:       bash slurm/launch.sh   (skips finished cells)
EOF
