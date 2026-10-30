#!/bin/bash
# One-off dependency installation on a compute node, into the shared checkout's venv.
# Submit only after jobs using that environment have finished:
#   source slurm/env.sh
#   sbatch slurm/jobs/setup.sbatch
#SBATCH --job-name=nanochat-setup
#SBATCH --account=llm-team
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b300:4
#SBATCH --time=01:00:00
#SBATCH --output=/nvidia/store/matteos/pre-train/logs/%x-%j.out
set -euo pipefail
source "${NANOCHAT_DIR:-/nvidia/store/matteos/pre-train/nanochat}/slurm/env.sh"
exec bash "$NANOCHAT_DIR/slurm/setup.sh"
