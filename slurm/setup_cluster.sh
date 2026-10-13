#!/usr/bin/env bash
# One-time setup on the cluster login node (the only place with internet).
# Builds .venv from uv.lock at its final path. "--frozen" keeps uv.lock unchanged, which
# --require_clean_git needs. Call it with bash; do not chmod the scripts (a mode change dirties git).
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found. Install it, then run this script again:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
    exit 1
fi

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$LOGS"
echo "uv $(uv --version)  cache $UV_CACHE_DIR"
uv sync --frozen --python 3.12      # 3.12 is the version of the workstation venv

"$ROOT/.venv/bin/python" -c "import torch; print('torch', torch.__version__, 'built for CUDA', torch.version.cuda)"
echo "Done. Next: bash slurm/preflight.sh"
