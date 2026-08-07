#!/bin/bash
#SBATCH --job-name=plasmax
#SBATCH --partition=workq
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.out

# PPO training on Isambard-AI Phase 2 (ARM64 + GH200), adapting the ORCHID
# script for Isambard's architecture and Slurm configuration. Submit with:
#   sbatch experiments/cluster/isambard/isambard_train.slurm                                    # single-seed train
#   ENV_SETUP=iter/hybrid/flattop \
#       TOTAL_TIMESTEPS=100000 sbatch experiments/cluster/isambard/isambard_train.slurm         # test run

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$HOME/plasmax}"
cd "$REPO"
mkdir -p outputs/logs

# --- scenario (override via env vars on the sbatch line) --------------------
ENV_SETUP="${ENV_SETUP:-iter/hybrid/flattop}"
BACKEND="${BACKEND:-cgm}"
TRANSFER_BACKEND="${TRANSFER_BACKEND:-}"
REWARD="${REWARD:-}"
VARIANT="${VARIANT:-realistic}"
ABLATE="${ABLATE:-none}"
SEED="${SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-1}"
DISRUPTION_PENALTY="${DISRUPTION_PENALTY:-}"
DT="${DT:-0.1}"
MINIMAL_CALLBACK="${MINIMAL_CALLBACK:-0}"
MIN_CB_ARG=""; [ "$MINIMAL_CALLBACK" = "1" ] && MIN_CB_ARG="--env.force-minimal-callback"
QUANTIZE_BINS="${QUANTIZE_BINS:-}"   # bins/actuator for discrete actions (realistic only)
QUANTIZE_ARG=""; [ -n "$QUANTIZE_BINS" ] && QUANTIZE_ARG="--env.quantize-bins $QUANTIZE_BINS"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
NUM_ENVS="${NUM_ENVS:-256}"
# Eval collects EVAL_N_ENVS rollouts per checkpoint for the return estimate.
# The eval trajectory is "lean" (heavy per-step sim-state stripped, ~7 KB/step/
# env), so 64 seeds is ~2 GB. num_envs does NOT affect this buffer.
EVAL_N_ENVS="${EVAL_N_ENVS:-64}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-isambard}"
WANDB_ENTITY="${WANDB_ENTITY:-flair}"

TRANSFER_ARG=""
[ -n "$TRANSFER_BACKEND" ] && TRANSFER_ARG="--env.transfer_backend $TRANSFER_BACKEND"
REWARD_ARG=()
[ -n "$REWARD" ] && REWARD_ARG=(--env.reward "$REWARD")
PENALTY_ARG=()
[ -n "$DISRUPTION_PENALTY" ] && PENALTY_ARG=(--env.disruption-penalty "$DISRUPTION_PENALTY")

# --- environment (Isambard-AI aarch64 + CUDA 12) --------------------------
module purge 2>/dev/null || true
source "${VENV:-.venv}/bin/activate"

# JAX compilation cache and GPU tuning (same as ORCHID)
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax-plasmax"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
if [ -n "${XLA_FLAGS_OVERRIDE:-}" ]; then
  export XLA_FLAGS="$XLA_FLAGS_OVERRIDE"
else
  export XLA_FLAGS="--xla_gpu_triton_gemm_any=True ${XLA_EXTRA:-}"
  if [ "$NUM_SEEDS" -gt 1 ]; then
    export XLA_FLAGS="$XLA_FLAGS --xla_gpu_enable_command_buffer="
  fi
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  arch=$(uname -m)"
nvidia-smi -L || true
uv run --no-sync python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"

srun uv run --no-sync python scripts/train_ppo.py \
    --env.env_setup "$ENV_SETUP" \
    --env.backend   "$BACKEND" \
    $TRANSFER_ARG \
    "${REWARD_ARG[@]}" \
    --env.variant   "$VARIANT" \
    --env.ablate    "$ABLATE" \
    "${PENALTY_ARG[@]}" \
    --env.dt "$DT" \
    $MIN_CB_ARG \
    $QUANTIZE_ARG \
    --seed "$SEED" \
    --num-seeds "$NUM_SEEDS" \
    --ppo.total_timesteps "$TOTAL_TIMESTEPS" \
    --ppo.num_envs "$NUM_ENVS" \
    --env.eval_n_envs "$EVAL_N_ENVS" \
    --wandb.mode "$WANDB_MODE" \
    --wandb.group "$WANDB_GROUP" \
    --wandb.entity "$WANDB_ENTITY"
