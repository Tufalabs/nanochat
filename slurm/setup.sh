#!/bin/bash
# Create/update the shared Python environment on an allocated compute node.
# Submit slurm/jobs/setup.sbatch, or run this script through srun in an allocation.
# Wait for jobs using this checkout's venv to finish before changing dependencies.
# Installs uv + a uv-managed Python under $PRETRAIN_ROOT/.tools (nothing goes into $HOME), then syncs
# nanochat/.venv: torch 2.9.1+cu130 (aarch64, sm_103-capable) + flash-attn-4 (CuTe DSL, JIT) + deps.
# /nvidia is shared; every compute node uses the same environment.
set -euo pipefail
source "$(dirname "$0")/env.sh"

# salloc may set job variables while leaving the shell on the login node.
# Require both an allocation and execution on one of its hosts before installing.
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo "Setup requires an allocated compute node. Use: sbatch slurm/jobs/setup.sbatch" >&2
    exit 1
fi
ALLOCATED_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
SETUP_HOST=$(hostname -s)
ON_ALLOCATED_HOST=false
while IFS= read -r node; do
    if [[ "${node%%.*}" == "$SETUP_HOST" ]]; then
        ON_ALLOCATED_HOST=true
        break
    fi
done <<< "$ALLOCATED_HOSTS"
if [[ "$ON_ALLOCATED_HOST" != true ]]; then
    echo "Setup must run on an allocated compute node, not $SETUP_HOST. Use srun or sbatch slurm/jobs/setup.sbatch." >&2
    exit 1
fi

if ! command -v uv >/dev/null; then
    mkdir -p "$PRETRAIN_ROOT/.tools/bin"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PRETRAIN_ROOT/.tools/bin" UV_NO_MODIFY_PATH=1 sh
fi
cd "$NANOCHAT_DIR"
uv sync --locked --extra gpu --group dev
.venv/bin/python -c "import torch, flash_attn.cute; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| arch list', torch.cuda.get_arch_list())"
"$TRITON_PTXAS_PATH" --version
"$TRITON_PTXAS_PATH" --help | grep sm_103a >/dev/null
