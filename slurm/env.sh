# Cluster settings, sourced by launch.sh, preflight.sh and setup_cluster.sh.
# Override any value by exporting it before the call, e.g.  ACCOUNT=OTHER bash slurm/launch.sh
# Defaults follow the HQD-GP example scripts (GPU account and Booster-style partition). Confirm
# them with preflight.sh, which asks the scheduler with "sbatch --test-only".

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"

ACCOUNT="${ACCOUNT:-FUPB2_MEATIC}"
PARTITION="${PARTITION:-boost_fua_prod}"
QOS="${QOS:-normal}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
THROTTLE="${THROTTLE:-8}"                   # array tasks of one block that run at the same time
MAIL_USER="${MAIL_USER:-emanuele.nardone@unicas.it}"
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
# (the ensemble trains 3 members), so 2 cells per P1 task; a traffic cell (862 channels, 2.7x
# ECL) about 1 h; an ACI pass reloads 6 runs and recalibrates, minutes per run.
SHARDS_P1=12;   TIME_P1=06:00:00
SHARDS_P2=1;    TIME_P2=02:00:00
SHARDS_P3S=1;   TIME_P3S=01:00:00
SHARDS_P3=6;    TIME_P3=04:00:00
SHARDS_P3B=1;   TIME_P3B=02:00:00

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
