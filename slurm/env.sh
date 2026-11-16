# Defaults for every launcher and preflight script. Site account, partition, QOS,
# and email belong in the untracked repo-root .env (gitignored), which this file loads.
# Do not put those values here: this file is tracked.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"

ACCOUNT=""                              # required: your project account
PARTITION=""                            # required: partition for the requested resources
QOS=""                                  # optional: blank uses the scheduler default
MAIL_USER=""                            # optional: blank disables email notifications
MAIL_TYPE="${MAIL_TYPE:-FAIL}"         # used only when MAIL_USER is set
PAPER_TIME="${PAPER_TIME:-24:00:00}"   # paper job limit; dcgp_fua_prod max is 24h
GRES="${GRES:-}"                        # blank omits --gres and runs on CPU; set gpu:1 for a GPU job
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
THROTTLE="${THROTTLE:-8}"                   # array tasks of one block that run at the same time
MIN_FREE_GB="${MIN_FREE_GB:-40}"            # R4 (ECL) + R5 (216 cells) write about 25 GB

# Array tasks per block and time limit per task. A task runs its share of cells one after the
# other, so the limit must cover the whole share. Per-cell times in experiments.md (RTX 5080) are
# 3-4 min (R5), 10 min (R1, R3), 18 min (R4), 5 min per ablation (R6); the limits keep a 2x margin.
SHARDS_SMOKE=1; TIME_SMOKE=01:00:00
SHARDS_R1=4;    TIME_R1=02:00:00
SHARDS_R2=2;    TIME_R2=01:00:00
SHARDS_R3=4;    TIME_R3=03:00:00
SHARDS_R4=9;    TIME_R4=03:00:00
SHARDS_R4B=1;   TIME_R4B=01:00:00
SHARDS_R5=12;   TIME_R5=06:00:00
SHARDS_R6=3;    TIME_R6=04:00:00
# Step 7 has no timed runs yet. Estimates: an ECL baseline at most 3x the 18-min R4 CISSN cell
# (the ensemble trains 3 members), so 2 cells per P1 task; a solar cell (137 channels, twice the
# training windows of ECL) about 40 min; an ACI pass reloads 6 runs and recalibrates, minutes per run.
SHARDS_P1=12;   TIME_P1=06:00:00
SHARDS_P2=1;    TIME_P2=02:00:00
SHARDS_P3S=1;   TIME_P3S=01:00:00
SHARDS_P3=6;    TIME_P3=04:00:00
SHARDS_P3B=1;   TIME_P3B=02:00:00
# P5 estimate: ECL DLinear 146-173 s and PatchTST-lite 368-426 s at H=336, times 1.4 for solar
# (CISSN solar/ECL ratio), so at most about 10 min per cell and 20 min per 2-cell task.
SHARDS_P5=6;    TIME_P5=02:00:00

# uv state lives next to the checkout, not in $HOME (small quota; the torch wheel is about 3 GB).
UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "$ROOT")/.uv-cache}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$(dirname "$ROOT")/.uv-python}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_LINK_MODE

# Compute nodes have no internet and must not call "uv run": use the venv python directly.
if [ -x "$ROOT/.venv/bin/python" ]; then
    PY="$ROOT/.venv/bin/python"
elif [ -x "$ROOT/.venv/Scripts/python.exe" ]; then  # workstation dry run only
    PY="$ROOT/.venv/Scripts/python.exe"
else
    PY="$ROOT/.venv/bin/python"
fi
LOGS="$ROOT/logs/slurm"                    # logs/ is gitignored; --require_clean_git needs a clean tree

if [ -f "$ROOT/.env" ]; then
    # shellcheck disable=SC1091
    source "$ROOT/.env"
fi

# An empty GRES must not become --gres=, which the scheduler rejects.
append_gres() {
    local -n _dest=$1
    if [ -n "${GRES:-}" ]; then
        _dest+=(--gres="$GRES")
    fi
}

require_cluster_settings() {
    if [ -z "$ACCOUNT" ] || [ -z "$PARTITION" ]; then
        echo "ERROR: fill ACCOUNT and PARTITION in slurm/env.sh before using Slurm." >&2
        echo "ERROR: put site values in the untracked .env instead of this tracked file." >&2
        return 2
    fi
}
