#!/bin/bash
# Two nodes by default, four GB300s per node; sbatch --nodes=N overrides the count.
# Prefer slurm/submit_run.py or ladder.py: they prepare descriptive run folders
# before submission, including the path for Slurm startup/error output.
#SBATCH --job-name=nanochat-train
#SBATCH --account=llm-team
#SBATCH --partition=dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:b300:4
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
set -euo pipefail
source "${NANOCHAT_DIR:-/nvidia/store/matteos/pre-train/nanochat}/slurm/env.sh"

# Submission tools set NANOCHAT_RUN_DIR before sbatch so scheduler output also lands here.
TAG="${TAG:-run-${SLURM_JOB_ID}}"
export NANOCHAT_RUN_DIR="${NANOCHAT_RUN_DIR:-$LADDERS_DIR/${LADDER_NAME:-standalone}/$TAG}"
mkdir -p "$NANOCHAT_RUN_DIR"/{tmp,checkpoints,wandb,nccl,torchrun}
NANOCHAT_RUN_DIR=$(cd "$NANOCHAT_RUN_DIR" && pwd)
export NANOCHAT_RUN_DIR
export TMPDIR="$NANOCHAT_RUN_DIR/tmp"
export WANDB_DIR="$NANOCHAT_RUN_DIR/wandb"
export WANDB_CACHE_DIR="$WANDB_DIR/cache"
export WANDB_CONFIG_DIR="$WANDB_DIR/config"
export WANDB_DATA_DIR="$WANDB_DIR/data"
export NCCL_DEBUG_FILE="$NANOCHAT_RUN_DIR/nccl/%h.%p.log"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$(basename "$(dirname "$NANOCHAT_RUN_DIR")")}"
export CODE_DIR="$NANOCHAT_DIR"
ARGS=("$@" "--run-dir=$NANOCHAT_RUN_DIR")
LOG="$NANOCHAT_RUN_DIR/train.log"
DONE="$NANOCHAT_RUN_DIR/done"
# Also supports a direct sbatch invocation (whose default scheduler sink is /dev/null).
exec >> "$NANOCHAT_RUN_DIR/slurm-${SLURM_JOB_ID}.out" 2>&1

if [ -f "$DONE" ]; then
    echo "$TAG already finished ($DONE exists), nothing to do"
    exit 0
fi

NNODES=${SLURM_JOB_NUM_NODES:-1}
export NPROC_PER_NODE=${NANOCHAT_NPROC_PER_NODE:-$GPUS_PER_NODE}
[[ "$NPROC_PER_NODE" =~ ^[1-4]$ ]] || { echo "workers per node must be 1-4" >&2; exit 2; }
echo "[$(date '+%F %T')] tag=$TAG host=$(hostname) job=${SLURM_JOB_ID} nodes=$NNODES world_size=$((NNODES * NPROC_PER_NODE)) code=$CODE_DIR"
echo "args: ${ARGS[*]}"
echo "log:  $LOG"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

cd "$CODE_DIR"
python - "$NANOCHAT_RUN_DIR" "$CODE_DIR" "${ARGS[@]}" <<'PY'
import datetime, json, os, subprocess, sys
from pathlib import Path
directory, checkout = Path(sys.argv[1]), sys.argv[2]
git = lambda *args: subprocess.run(['git', '-C', checkout, *args], capture_output=True, text=True).stdout.strip()
record = dict(job_id=os.environ['SLURM_JOB_ID'], tag=os.environ.get('TAG', directory.name),
              workers_per_node=int(os.environ['NPROC_PER_NODE']),
              args=sys.argv[3:], source_checkout=checkout, source_commit=git('rev-parse', 'HEAD'),
              source_dirty=bool(git('status', '--porcelain')),
              started=datetime.datetime.now().isoformat(timespec='seconds'))
(directory / f"execution-{record['job_id']}.json").write_text(json.dumps(record, indent=2) + '\n')
if not (directory / 'run.json').exists():
    (directory / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
PY
START=$(date +%s)
# 'set -o pipefail' makes a crash in torchrun fail the job even though output is piped through tee
if (( NNODES == 1 )); then
    torchrun --log-dir="$NANOCHAT_RUN_DIR/torchrun" --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- "${ARGS[@]}" 2>&1 | tee "$LOG"
else
    # One torchrun agent per node, each spawning four workers. Static rendezvous uses
    # the first allocated host; node-exclusive allocations make port 29500 safe.
    mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    [[ ${#HOSTS[@]} -eq $NNODES ]] || { echo "Cannot resolve allocated hosts" >&2; exit 1; }
    export MASTER_ADDR=${HOSTS[0]}
    export MASTER_PORT=${MASTER_PORT:-29500}
    export CODE_DIR
    srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
        --cpu-bind=none --mpi=none --kill-on-bad-exit=1 \
        bash -c 'set -euo pipefail
            cd "$CODE_DIR"
            # Avoid cross-host Inductor/Triton cache races while retaining reuse on each host.
            export TRITON_CACHE_DIR="$TRITON_CACHE_DIR/nodes/$(hostname)"
            export TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR/nodes/$(hostname)"
            exec torchrun --log-dir="$NANOCHAT_RUN_DIR/torchrun" \
            --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node="$NPROC_PER_NODE" \
            --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
            --max_restarts=0 -m scripts.base_train -- "$@"' bash "${ARGS[@]}" 2>&1 | tee "$LOG"
fi
END=$(date +%s)

echo "wall_time_sec=$((END - START)) job=${SLURM_JOB_ID} nodes=$NNODES world_size=$((NNODES * NPROC_PER_NODE)) host=$(hostname) finished=$(date '+%F %T')" > "$DONE"
echo "[$(date '+%F %T')] $TAG finished in $((END - START))s"
