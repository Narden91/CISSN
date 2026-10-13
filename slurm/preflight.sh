#!/usr/bin/env bash
# Checks the cluster checkout before anything is submitted. Runs every check, prints each
# result, and exits non-zero if any check fails. launch.sh calls it; you can also run it alone
# on the login node.  --no-slurm skips the scheduler checks (workstation dry run).
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$ROOT"

NO_SLURM=0
for arg in "$@"; do
    case "$arg" in
        --no-slurm) NO_SLURM=1 ;;
        -h|--help) echo "Usage: $0 [--no-slurm]"; exit 0 ;;
        *) echo "ERROR: unknown argument '$arg'" >&2; exit 2 ;;
    esac
done

FAILURES=0
ok()   { echo "  OK    $1"; }
bad()  { echo "  FAIL  $1" >&2; FAILURES=$((FAILURES + 1)); }
info() { echo "  info  $1"; }
check() {
    local description="$1"
    shift
    if "$@" >/dev/null 2>&1; then ok "$description"; else bad "$description"; fi
}

SLURM_FILES=(cells.py env.sh launch.sh preflight.sh run_block.slurm setup_cluster.sh)

echo "== Source ($(date -u +%FT%TZ))"
info "checkout $ROOT, branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null), commit $(git rev-parse --short HEAD 2>/dev/null)"
check "HEAD contains docs/prespecification.md (R4 needs it in the run commit)" git cat-file -e HEAD:docs/prespecification.md
for f in "${SLURM_FILES[@]}"; do
    check "slurm/$f is committed in HEAD" git cat-file -e "HEAD:slurm/$f"
done
if grep -lU $'\r' "${SLURM_FILES[@]/#/slurm/}" >/dev/null 2>&1; then
    bad "CRLF line endings in slurm/: $(grep -lU $'\r' "${SLURM_FILES[@]/#/slurm/}" | tr '\n' ' ')"
else
    ok "slurm/ files use LF line endings"
fi
for f in "${SLURM_FILES[@]}"; do
    case "$f" in *.py) continue ;; esac
    check "slurm/$f bash syntax" bash -n "slurm/$f"
done

echo "== Environment"
info "machine $(uname -m), $(uname -s)"
if [ -x "$PY" ]; then
    check "python >= 3.12 ($("$PY" -V 2>&1))" "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 12))'
    check "project imports (torch numpy pandas sklearn scipy yaml cissn)" "$PY" -c \
        'import torch, numpy, pandas, sklearn, scipy, yaml, tqdm, matplotlib, cissn'
    info "$("$PY" -c 'import torch; print("torch", torch.__version__, "built for CUDA", torch.version.cuda)' 2>&1 | tail -1)"
    "$PY" -c 'import torch; assert "+cu" in torch.__version__' >/dev/null 2>&1 \
        && ok "torch is a CUDA build" || bad "torch is not a CUDA build (run slurm/setup_cluster.sh; uv.lock pins +cu128)"
else
    bad "virtual environment missing: $PY (run bash slurm/setup_cluster.sh on the login node)"
fi

echo "== Data (exact checksums: --immutable_artifacts requires byte-identical files)"
if [ -x "$PY" ]; then
    DATA_OUT=$("$PY" - <<'EOF' 2>&1
from cissn.data.registry import verify_dataset
failed = []
for name in ("ETTh1", "ETTh2", "weather", "exchange_rate", "ECL"):
    try:
        verify_dataset(name, strict=True, require_exact_checksum=True)
        print(f"OK {name}")
    except Exception as exc:
        print(f"FAIL {name}: {str(exc).splitlines()[0][:160]}")
        failed.append(name)
raise SystemExit(1 if failed else 0)
EOF
)
    while IFS= read -r line; do
        case "$line" in OK*) ok "dataset ${line#OK }" ;; FAIL*) bad "dataset ${line#FAIL }" ;; *) info "$line" ;; esac
    done <<< "$DATA_OUT"
else
    bad "datasets not checked (no python)"
fi

echo "== Cells"
if [ -x "$PY" ]; then
    CELLS_OUT=$("$PY" slurm/cells.py check 2>&1); CELLS_CODE=$?
    echo "$CELLS_OUT" | grep -E '^(OK|FAIL) ' | sed 's/^/  /'
    if [ "$CELLS_CODE" -eq 0 ]; then ok "every cell parses with its runner and the counts match"
    else bad "cell manifest check failed"; echo "$CELLS_OUT" | tail -5 >&2; fi
    "$PY" slurm/cells.py status 2>&1 | sed 's/^/  info  /'
else
    bad "cells not checked (no python)"
fi

echo "== Disk and logs"
if mkdir -p "$LOGS"; then ok "log folder $LOGS is writable"; else bad "cannot create $LOGS"; fi
FREE_GB=$(( $(df -Pk "$ROOT" | awk 'NR == 2 {print $4}') / 1024 / 1024 ))
if [ "$FREE_GB" -ge "$MIN_FREE_GB" ]; then ok "free disk ${FREE_GB} GB >= ${MIN_FREE_GB} GB"; else bad "free disk ${FREE_GB} GB < ${MIN_FREE_GB} GB"; fi

echo "== Scheduler"
if [ "$NO_SLURM" -eq 1 ]; then
    info "skipped (--no-slurm)"
elif ! command -v sbatch >/dev/null 2>&1; then
    bad "sbatch not found; run on a login node"
else
    OUT=$(sbatch --test-only -A "$ACCOUNT" -p "$PARTITION" --qos="$QOS" --gres="$GRES" -N 1 -c "$CPUS" \
        --mem="$MEM" -t 01:00:00 --wrap=true 2>&1)
    if [[ "$OUT" == *"to start at"* ]]; then
        ok "scheduler accepts the job shape ($ACCOUNT, $PARTITION, qos $QOS, $GRES, $CPUS cpus, $MEM)"
    else
        bad "scheduler rejects the job shape: $(echo "$OUT" | tail -1)"
    fi
    ACTIVE=$(squeue -h -u "$(id -un)" -o "%i %j" 2>/dev/null | awk '$2 ~ /^cissn_/')
    if [ -z "$ACTIVE" ]; then ok "no queued or running cissn_* job"
    else bad "cissn_* jobs already active (scancel them or wait):"; echo "$ACTIVE" >&2; fi
fi

# Last: venv, logs, and the checks above must leave the tree clean for --require_clean_git.
echo "== Git tree"
DIRTY=$(git status --porcelain 2>&1)
if [ -z "$DIRTY" ]; then ok "git tree clean at $(git rev-parse --short HEAD)"
else bad "git tree not clean (commit, or add the file to .gitignore):"; echo "$DIRTY" | head -20 >&2; fi

echo
if [ "$FAILURES" -eq 0 ]; then
    echo "Preflight OK"
else
    echo "Preflight FAILED: $FAILURES check(s)" >&2
    exit 1
fi
