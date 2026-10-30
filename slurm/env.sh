# Shared environment for every nanochat job on the thur cluster. Sourced by all slurm/* scripts.
# Layout (everything lives under PRETRAIN_ROOT, nothing in $HOME):
#   $PRETRAIN_ROOT/nanochat   code checkout + shared .venv
#   $PRETRAIN_ROOT/base       shared data shards, tokenizer and eval bundle
#   $NANOCHAT_DIR/ladders     <ladder>/<run-tag>/ contains all run outputs; no source copies
#   $PRETRAIN_ROOT/logs       shared data preparation / setup logs only
#   $PRETRAIN_ROOT/.tools     uv, uv-managed python, uv cache
#   $PRETRAIN_ROOT/cache      JIT caches (FlashAttention-4)
export PRETRAIN_ROOT="${PRETRAIN_ROOT:-/nvidia/store/matteos/pre-train}"
export NANOCHAT_DIR="${NANOCHAT_DIR:-$PRETRAIN_ROOT/nanochat}"   # checkout that owns the .venv
export CODE_DIR="$NANOCHAT_DIR"                                 # runs use the live checkout
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$PRETRAIN_ROOT/base}"
export LADDERS_DIR="${LADDERS_DIR:-$NANOCHAT_DIR/ladders}"
export LOGS_DIR="${LOGS_DIR:-$PRETRAIN_ROOT/logs}"

# Slurm defaults (override per submission, e.g. PARTITION=short)
export SLURM_ACCOUNT_NAME="${SLURM_ACCOUNT_NAME:-llm-team}"
export GPUS_PER_NODE=4   # GB300 nodes: four GPUs per node; node count is selected at submission
export PATH="/cm/local/apps/slurm/current/bin:$PATH"
if [ -z "${SLURM_CONF:-}" ] && [ -r /cm/shared/apps/slurm/etc/slurm/slurm.conf ]; then
    export SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf
fi

# Toolchain
export PATH="$PRETRAIN_ROOT/.tools/bin:$PATH"
export UV_CACHE_DIR="$PRETRAIN_ROOT/.tools/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PRETRAIN_ROOT/.tools/python"

# Runtime
export OMP_NUM_THREADS=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1                      # persist FA4 JIT kernels across jobs
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR="$PRETRAIN_ROOT/cache/fa4"
export PYTHONWARNINGS="ignore::DeprecationWarning"                   # cutlass-dsl is very chatty

mkdir -p "$NANOCHAT_BASE_DIR" "$LADDERS_DIR" "$LOGS_DIR" "$FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"
if [ -f "$NANOCHAT_DIR/.venv/bin/activate" ]; then
    source "$NANOCHAT_DIR/.venv/bin/activate"
fi

# Keep Triton's compiler cache and the CUDA 13 assembler inside this setup.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRETRAIN_ROOT/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PRETRAIN_ROOT/cache/inductor}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-$NANOCHAT_DIR/.venv/lib/python3.10/site-packages/nvidia/cu13/bin/ptxas}"

# Logging is opt-in. jobs/train.sbatch / configure_run_dir put all W&B files inside the run.
