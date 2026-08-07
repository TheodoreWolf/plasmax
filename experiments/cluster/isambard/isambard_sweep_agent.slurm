#!/bin/bash
#SBATCH --job-name=plasmax-sweep
#SBATCH --partition=workq
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.out

# One wandb sweep agent per Slurm task, for the PPO/BgB random sweep
# (sweeps/ppo_iter_hybrid_flattop_bgb.yaml). Each task grabs COUNT configs
# (default 1), each of which vmaps 3 seeds. Submit as an array:
#   SWEEP_ID=flair/plasmax/<id> sbatch --array=1-16 experiments/cluster/isambard/isambard_sweep_agent.slurm

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$HOME/plasmax}"
cd "$REPO"
mkdir -p outputs/logs

: "${SWEEP_ID:?set SWEEP_ID=flair/plasmax/<id> (from 'wandb sweep ...')}"
COUNT="${COUNT:-1}"  # sweep configs (each = one 3-seed run) per Slurm task

module purge 2>/dev/null || true
source "${VENV:-.venv}/bin/activate"

export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax-plasmax"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
# num_seeds=3 vmaps 3 runs; disable command-buffer dispatch (see isambard_train.slurm).
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True --xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  task=${SLURM_ARRAY_TASK_ID:-none}  sweep=$SWEEP_ID"
nvidia-smi -L || true

srun wandb agent --count "$COUNT" "$SWEEP_ID"
