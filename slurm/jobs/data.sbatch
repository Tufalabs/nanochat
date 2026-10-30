#!/bin/bash
# One-off data preparation: ClimbMix shards + BPE tokenizer + CORE eval bundle.
# Idempotent: already-downloaded shards / existing tokenizer are skipped, so it is safe to re-run
# with a larger NUM_SHARDS later. Submit from the nanochat dir:
#   sbatch slurm/jobs/data.sbatch                 # 1000 shards (~100GB, enough for the default ladders)
#   NUM_SHARDS=2000 sbatch slurm/jobs/data.sbatch  # more data for bigger ladders (max 6542)
# Data preparation is not distributed, so it only allocates one node.
#SBATCH --job-name=nanochat-data
#SBATCH --account=llm-team
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:b300:4
#SBATCH --time=04:00:00
#SBATCH --output=/nvidia/store/matteos/pre-train/logs/%x-%j.out
set -euo pipefail
source "${NANOCHAT_DIR:-/nvidia/store/matteos/pre-train/nanochat}/slurm/env.sh"
cd "$NANOCHAT_DIR"

NUM_SHARDS="${NUM_SHARDS:-1000}"   # each shard ~100MB parquet, ~250M chars; shard 6542 is always the val shard
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
echo "[$(date)] host=$(hostname) NUM_SHARDS=$NUM_SHARDS VOCAB_SIZE=$VOCAB_SIZE NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"

# 1) first 8 train shards (+ val shard): ~2B chars, enough to train the tokenizer
python -m nanochat.dataset -n 8 -w 16
# 2) the rest in the background while the tokenizer trains
python -m nanochat.dataset -n "$NUM_SHARDS" -w 32 > "$LOGS_DIR/data-download-${SLURM_JOB_ID:-local}.log" 2>&1 &
DOWNLOAD_PID=$!

# 3) tokenizer (GPT-4 style BPE, trained on the first ~2B chars), skipped if it already exists
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ]; then
    python -m scripts.tok_train --vocab-size="$VOCAB_SIZE"
    python -m scripts.tok_eval
else
    echo "Tokenizer already exists at $NANOCHAT_BASE_DIR/tokenizer, skipping"
fi

# 4) CORE eval bundle, downloaded once here so concurrent ladder jobs don't race on it
python - <<'PY'
import os
from nanochat.common import get_base_dir, download_file_with_lock
from scripts.base_eval import EVAL_BUNDLE_URL, place_eval_bundle
if not os.path.exists(os.path.join(get_base_dir(), "eval_bundle")):
    download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
print("eval bundle ready")
PY

wait $DOWNLOAD_PID
tail -1 "$LOGS_DIR/data-download-${SLURM_JOB_ID:-local}.log"
echo "[$(date)] done: $(ls "$NANOCHAT_BASE_DIR/base_data_climbmix" | grep -c 'parquet$') parquet files, $(du -sh "$NANOCHAT_BASE_DIR/base_data_climbmix" | cut -f1)"
