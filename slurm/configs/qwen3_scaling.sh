# Editable ladder preset, sourced by ../run_scaling_ladder.sh.
# Copy this file and pass the new path as the runner's second argument for another experiment.
# -------------------------- Ladder configuration -----------------------------
# Use a new name for each experiment. An existing name is never overwritten.
# Optional shell override: LADDER_NAME=my-experiment bash .../run_scaling_ladder.sh plan
NAME="${LADDER_NAME:-qwen3-scaling-v1}"
MODE=ratio                         # isoflop: sizes x FLOP budgets; ratio: sizes at fixed D/N
DEPTHS=(8 12 16 20 24)              # five sizes, ~42M-730M params; width also scales with depth
FLOPS=(1e18 3e18)                   # used only in isoflop mode; ignored by this ratio ladder
RATIO=20                           # training tokens per scaling parameter

# Each run occupies NODES nodes with four GB300 GPUs each.
NODES=2                           # eight GPUs/run; supported counts: 1, 2, 4
PARTITION=dev
MAX_PARALLEL=5                    # run concurrency cap; Slurm's user quota may limit actual concurrency
TIME_MIN=""                        # empty = estimated by ladder.py; otherwise minutes/run
ENABLE_WANDB=true                  # credentials/project come from the environment

# fixed preserves optimization across node counts. per_node multiplies the global
# token batch by NODES to keep the current 16 sequences/GPU at 8K without shrinking.
# The trainer also scales effective learning rates with sqrt(global batch).
BATCH_POLICY=fixed                 # fixed | per_node
GLOBAL_BATCH_TOKENS=524288          # fixed batch, or one-node reference for per_node
case "$BATCH_POLICY" in
    fixed) ;;
    per_node) GLOBAL_BATCH_TOKENS=$((GLOBAL_BATCH_TOKENS * NODES)) ;;
    *) echo "Unknown BATCH_POLICY: $BATCH_POLICY" >&2; exit 2 ;;
esac

# These arguments apply to every run. Keep the validation protocol fixed across sizes.
TRAIN_ARGS=(
    --max-seq-len=8192
    --aspect-ratio=64
    --head-dim=128
    --kv-group-size=2
    --mlp-ratio=3
    --rope-theta=1000000
    --window-pattern=L

    # Global batch in TOKENS/update across all sizes. This is a starting choice;
    # compare optimization quality against 262,144 tokens before the full ladder.
    --total-batch-size="$GLOBAL_BATCH_TOKENS"
    # Upper bound in SEQUENCES/GPU/microbatch. At one node: 16 * 8192 * 4 = 524,288 tokens.
    # With BATCH_POLICY=fixed, the trainer reduces this to 8 at two nodes, 4 at four.
    --device-batch-size=16

    # Base learning rates, before batch/width scaling.
    --matrix-lr=0.005
    --unembedding-lr=0.002
    --scalar-lr=0.1                # RMSNorm base LR = 0.01 * this value
    --weight-decay=0.28
    --warmup-steps=40
    --warmdown-ratio=0.65
    --final-lr-frac=0.05

    --eval-every=250
    --eval-tokens=52428800
    --core-metric-every=999999    # effectively final-only for these budgets
    --core-metric-max-per-task=-1 # full CORE bundle
    --sample-every=-1
    --save-every=-1              # final checkpoint only
    --log-every=10
    --optimizer-timing-every=10  # synchronized timing of optimizer+communication
)
