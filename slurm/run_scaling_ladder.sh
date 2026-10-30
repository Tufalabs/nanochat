#!/usr/bin/env bash
# Run from a shell, not sbatch. Only the explicit `submit` action launches jobs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
usage() {
    echo "Usage: $0 [plan|submit] [CONFIG]"
    echo "  plan    Preview only; also the default action."
    echo "  submit  Submit the configured Slurm jobs."
    echo "  CONFIG  Shell preset; defaults to slurm/configs/qwen3_scaling.sh."
}
if (( $# > 2 )); then
    usage >&2
    exit 2
fi
ACTION="${1:-plan}"
case "$ACTION" in
    plan|submit) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
CONFIG_FILE="${2:-$SCRIPT_DIR/configs/qwen3_scaling.sh}"
if [[ ! -r "$CONFIG_FILE" ]]; then
    echo "Cannot read ladder preset: $CONFIG_FILE" >&2
    exit 2
fi
# Resolve before changing directories; paths containing spaces are supported.
CONFIG_FILE="$(cd -- "$(dirname -- "$CONFIG_FILE")" && pwd)/$(basename -- "$CONFIG_FILE")"
export NANOCHAT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$CONFIG_FILE"
cd "$NANOCHAT_DIR"

LADDER_ARGS=(
    "$ACTION" "$NAME"
    --mode "$MODE"
    --depths "${DEPTHS[@]}"
    --flops "${FLOPS[@]}"
    --ratio "$RATIO"
    --partition "$PARTITION"
    --nodes "$NODES"
    --max-parallel "$MAX_PARALLEL"
)
if [[ -n "$TIME_MIN" ]]; then
    LADDER_ARGS+=(--time "$TIME_MIN")
fi
case "$ENABLE_WANDB" in
    true) LADDER_ARGS+=(--wandb) ;;
    false) ;;
    *) echo "ENABLE_WANDB must be true or false" >&2; exit 2 ;;
esac
exec python "$SCRIPT_DIR/ladder.py" "${LADDER_ARGS[@]}" -- "${TRAIN_ARGS[@]}"
