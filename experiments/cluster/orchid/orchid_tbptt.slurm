#!/bin/bash
#SBATCH --job-name=plasmax-tbptt
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.out

# Direct policy optimization through TORAX with online truncated BPTT.
# Metrics use aggregate differentiated simulator transitions as the wandb step,
# matching PPO's global-step axis. One job trains one reproducible seed.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$HOME/plasmax}"
cd "$REPO"
mkdir -p outputs/logs

ENV_SETUP="${ENV_SETUP:-iter/hybrid/flattop}"
BACKEND="${BACKEND:-bohm_gyrobohm}"
REWARD="${REWARD:-}"
VARIANT="${VARIANT:-realistic}"
DISRUPTION_PENALTY="${DISRUPTION_PENALTY:-}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-4400}"
TRUNCATION_STEPS="${TRUNCATION_STEPS:-100}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-16}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-64}"
PASSES="${PASSES:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SEED="${SEED:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-tbptt-vs-ppo-iter-hybrid-flattop-bgb}"
WANDB_ENTITY="${WANDB_ENTITY:-flair}"
REWARD_ARG=()
[ -n "$REWARD" ] && REWARD_ARG=(--reward "$REWARD")
PENALTY_ARG=()
[ -n "$DISRUPTION_PENALTY" ] && PENALTY_ARG=(--disruption-penalty "$DISRUPTION_PENALTY")

module purge 2>/dev/null || true
source "${VENV:-.venv}/bin/activate"

# Concurrent seeds compile the same graph, so keep cache writes job-local.
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax-plasmax-tbptt-${SLURM_JOB_ID:-solo}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True ${XLA_EXTRA:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  seed=$SEED"
nvidia-smi -L || true
uv run --no-sync python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"

srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    uv run --no-sync python experiments/studies/train_backprop_policy.py \
    --env "$ENV_SETUP" \
    --backend "$BACKEND" \
    "${REWARD_ARG[@]}" \
    --variant "$VARIANT" \
    "${PENALTY_ARG[@]}" \
    --rollout-steps "$ROLLOUT_STEPS" \
    --truncation-steps "$TRUNCATION_STEPS" \
    --num-rollouts "$NUM_ROLLOUTS" \
    --eval-rollouts "$EVAL_ROLLOUTS" \
    --iters "$PASSES" \
    --learning-rate "$LEARNING_RATE" \
    --grad-clip "$GRAD_CLIP" \
    --seed "$SEED" \
    --out "outputs/tbptt_${WANDB_GROUP}_seed${SEED}.npz" \
    --wandb \
    --wandb-mode "$WANDB_MODE" \
    --wandb-group "$WANDB_GROUP" \
    --wandb-entity "$WANDB_ENTITY"
