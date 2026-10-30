#!/bin/bash
# (Re)create the Python environment on the login node. Idempotent; safe to re-run after editing pyproject.toml.
# Installs uv + a uv-managed Python under $PRETRAIN_ROOT/.tools (nothing goes into $HOME), then syncs
# nanochat/.venv: torch 2.9.1+cu130 (aarch64, sm_103-capable) + flash-attn-4 (CuTe DSL, JIT) + deps.
# Login node and GPU nodes are both aarch64 (Grace), and /nvidia is shared, so the venv works everywhere.
set -euo pipefail
source "$(dirname "$0")/env.sh"
if ! command -v uv >/dev/null; then
    mkdir -p "$PRETRAIN_ROOT/.tools/bin"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PRETRAIN_ROOT/.tools/bin" UV_NO_MODIFY_PATH=1 sh
fi
cd "$NANOCHAT_DIR"
uv sync --extra gpu --group dev
.venv/bin/python -c "import torch, flash_attn.cute; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| arch list', torch.cuda.get_arch_list() or 'n/a on login node')"
"$TRITON_PTXAS_PATH" --version
"$TRITON_PTXAS_PATH" --help | grep sm_103a >/dev/null
