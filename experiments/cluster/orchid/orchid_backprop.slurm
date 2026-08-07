#!/bin/bash
#SBATCH --job-name=plasmax-backprop
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.out

# Gradient-based open-loop trajectory optimization (experiments/studies/optimize_open_loop.py)
# on JASMIN ORCHID (A100). Companion to orchid_train.slurm's PPO path, for
# PPO-vs-backprop-control head-to-heads. Submit from the repo root, override
# vars on the sbatch line:
#   ENV_SETUP=iter/hybrid/flattop \
#   DISRUPTION_PENALTY=-5 NUM_STEPS=4400 KNOTS=40 SEED=0 \
#   WANDB_GROUP=ppo_vs_backprop_iter_hybrid_flattop \
#       sbatch experiments/cluster/orchid/orchid_backprop.slurm

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$HOME/plasmax}"
cd "$REPO"
mkdir -p outputs/logs

ENV_SETUP="${ENV_SETUP:-iter/hybrid/flattop}"
BACKEND="${BACKEND:-cgm}"
REWARD="${REWARD:-}"
DISRUPTION_PENALTY="${DISRUPTION_PENALTY:-}"
NUM_STEPS="${NUM_STEPS:-4400}"
KNOTS="${KNOTS:-40}"
ITERS="${ITERS:-50}"
LR="${LR:-0.03}"
SEED="${SEED:-0}"
WANDB_GROUP="${WANDB_GROUP:-orchid}"
WANDB_ENTITY="${WANDB_ENTITY:-flair}"
REWARD_ARG=()
[ -n "$REWARD" ] && REWARD_ARG=(--reward "$REWARD")
PENALTY_ARG=()
[ -n "$DISRUPTION_PENALTY" ] && PENALTY_ARG=(--disruption-penalty "$DISRUPTION_PENALTY")

module purge 2>/dev/null || true
source "${VENV:-.venv}/bin/activate"

# Job-unique cache dir: concurrent seeds trace byte-identical HLO and
# collide writing to a shared cache dir over NFS, stalling all of them
# indefinitely (observed 2026-07-07 -- 5 concurrent seeds all hung before
# even env.reset() returned, each of which is instant when run solo).
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax-plasmax-backprop-${SLURM_JOB_ID:-solo}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True ${XLA_EXTRA:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}"
nvidia-smi -L || true
uv run --no-sync python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"

srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" uv run --no-sync python experiments/studies/optimize_open_loop.py \
    --env "$ENV_SETUP" \
    --backend "$BACKEND" \
    "${REWARD_ARG[@]}" \
    "${PENALTY_ARG[@]}" \
    --num-steps "$NUM_STEPS" \
    --knots "$KNOTS" \
    --iters "$ITERS" \
    --lr "$LR" \
    --seed "$SEED" \
    --out "outputs/open_loop_${WANDB_GROUP}_seed${SEED}.npz" \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    --wandb-entity "$WANDB_ENTITY"
