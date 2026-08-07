#!/bin/bash
#SBATCH --partition=orchid --account=orchid --qos=orchid
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=plasmax-mpc
#SBATCH --output=outputs/logs/%x-%j.out

set -euo pipefail

cd "$HOME/plasmax"
mkdir -p outputs/logs

export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_triton_gemm_any=True --xla_gpu_enable_command_buffer=}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

ENV_SETUP="${ENV_SETUP:-iter/hybrid/flattop}"
BACKEND="${BACKEND:-qlknn}"
REWARD="${REWARD:-}"
VARIANT="${VARIANT:-realistic}"
SEED="${SEED:-0}"
EVAL_N_ENVS="${EVAL_N_ENVS:-16}"
DISRUPTION_PENALTY="${DISRUPTION_PENALTY:-}"
ABLATE="${ABLATE:-none}"

MPC_REWARD_SCALAR="${MPC_REWARD_SCALAR:-P_fusion}"
MPC_HORIZON="${MPC_HORIZON:-5}"
MPC_NUM_SAMPLES="${MPC_NUM_SAMPLES:-64}"
MPC_TOTAL_STEPS="${MPC_TOTAL_STEPS:-6000}"
MPC_LOG_EVERY="${MPC_LOG_EVERY:-100}"
MPC_TRAIN_BATCH_SIZE="${MPC_TRAIN_BATCH_SIZE:-256}"
MPC_BUFFER_SIZE="${MPC_BUFFER_SIZE:-20000}"
MPC_HIDDEN="${MPC_HIDDEN:-128}"
MPC_LR="${MPC_LR:-1e-3}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-baselines}"
WANDB_ENTITY="${WANDB_ENTITY:-flair}"
WANDB_PROJECT="${WANDB_PROJECT:-plasmax}"
REWARD_ARG=()
[ -n "$REWARD" ] && REWARD_ARG=(--env.reward "$REWARD")
PENALTY_ARG=()
[ -n "$DISRUPTION_PENALTY" ] && PENALTY_ARG=(--env.disruption-penalty "$DISRUPTION_PENALTY")

uv run python scripts/evaluate_baselines.py \
  --seed "$SEED" \
  --env.env-setup "$ENV_SETUP" \
  --env.backend "$BACKEND" \
  "${REWARD_ARG[@]}" \
  --env.variant "$VARIANT" \
  --env.eval-n-envs "$EVAL_N_ENVS" \
  "${PENALTY_ARG[@]}" \
  --env.ablate "$ABLATE" \
  --mpc.reward-scalar "$MPC_REWARD_SCALAR" \
  --mpc.horizon "$MPC_HORIZON" \
  --mpc.num-samples "$MPC_NUM_SAMPLES" \
  --mpc.total-steps "$MPC_TOTAL_STEPS" \
  --mpc.log-every "$MPC_LOG_EVERY" \
  --mpc.train-batch-size "$MPC_TRAIN_BATCH_SIZE" \
  --mpc.buffer-size "$MPC_BUFFER_SIZE" \
  --mpc.hidden "$MPC_HIDDEN" \
  --mpc.lr "$MPC_LR" \
  --wandb.mode "$WANDB_MODE" \
  --wandb.group "$WANDB_GROUP" \
  --wandb.entity "$WANDB_ENTITY" \
  --wandb.project "$WANDB_PROJECT"
