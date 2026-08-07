#!/bin/bash
#SBATCH --job-name=plasmax
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.out

# PPO training on JASMIN ORCHID (A100), covering all of: single-seed,
# multi-seed (jax.vmap), zero-shot transfer eval, and multi-seed transfer —
# selected purely by which env vars you set (see scripts/train_ppo.py, which
# this wraps 1:1 and itself consolidates the four scripts that used to back
# these four combinations). Submit from the repo root, override --time/
# --job-name on the sbatch line as needed (16h default covers the slowest
# combo, multi-seed transfer):
#   sbatch experiments/cluster/orchid/orchid_train.slurm                                    # single-seed train
#   NUM_SEEDS=3 sbatch experiments/cluster/orchid/orchid_train.slurm                        # multi-seed train
#   TRANSFER_BACKEND=qlknn_linear \
#       sbatch --time=12:00:00 experiments/cluster/orchid/orchid_train.slurm                # single-seed + transfer
#   NUM_SEEDS=3 TRANSFER_BACKEND=qlknn_linear \
#       sbatch experiments/cluster/orchid/orchid_train.slurm                                # multi-seed + transfer
#   ENV_SETUP=kstar BACKEND=fusion_lstm \
#       sbatch --time=04:00:00 experiments/cluster/orchid/orchid_train.slurm                # KSTAR world-model
#
# The venv is the uv-managed .venv in the repo (home is shared with the compute
# nodes). wandb runs online by default (compute nodes have verified outbound
# net); pass WANDB_MODE=offline for a no-net run, then sync afterwards with
# `wandb sync outputs/wandb/offline-run-*`.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$HOME/plasmax}"
cd "$REPO"
mkdir -p outputs/logs

# --- scenario (override via env vars on the sbatch line) --------------------
ENV_SETUP="${ENV_SETUP:-iter/hybrid/flattop}"
BACKEND="${BACKEND:-cgm}"   # cgm: ~5x faster than qlknn_linear
# If set, zero-shot evaluate the trained policy on this second, typically
# higher-fidelity backend after training (e.g. qlknn_linear).
TRANSFER_BACKEND="${TRANSFER_BACKEND:-}"
REWARD="${REWARD:-}"   # empty inherits task YAML metadata
VARIANT="${VARIANT:-realistic}"
ABLATE="${ABLATE:-none}"   # realistic leave-one-out: none|noise|resolution|filter|delay
SEED="${SEED:-0}"          # base PRNG seed (vary for seed sweeps)
NUM_SEEDS="${NUM_SEEDS:-1}"   # >1 vmaps that many independent seeds in one job
DISRUPTION_PENALTY="${DISRUPTION_PENALTY:-}"  # empty inherits task YAML metadata
DT="${DT:-0.1}"   # must match the env YAML numerics.fixed_dt (e.g. 2.0 for step)
MINIMAL_CALLBACK="${MINIMAL_CALLBACK:-0}"  # 1 = returns-only eval (skip physics logging)
MIN_CB_ARG=""; [ "$MINIMAL_CALLBACK" = "1" ] && MIN_CB_ARG="--env.force-minimal-callback"
TIME_AWARE="${TIME_AWARE:-0}"   # 1 = append elapsed episode time as an obs scalar
TIME_AWARE_ARG=""; [ "$TIME_AWARE" = "1" ] && TIME_AWARE_ARG="--env.time-aware"
QUANTIZE_BINS="${QUANTIZE_BINS:-}"   # bins/actuator for discrete actions (realistic only)
QUANTIZE_ARG=""; [ -n "$QUANTIZE_BINS" ] && QUANTIZE_ARG="--env.quantize-bins $QUANTIZE_BINS"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
NUM_ENVS="${NUM_ENVS:-256}"   # x NUM_SEEDS envs run concurrently when vmapped
# Eval collects EVAL_N_ENVS rollouts per checkpoint for the return estimate.
# The eval trajectory is "lean" (heavy per-step sim-state stripped, ~7 KB/step/
# env), so 64 seeds is ~2 GB and fits alongside training on a 40 GB A100.
# num_envs does NOT affect this buffer.
EVAL_N_ENVS="${EVAL_N_ENVS:-64}"
WANDB_MODE="${WANDB_MODE:-online}"   # compute nodes have outbound net; offline for no-net
WANDB_GROUP="${WANDB_GROUP:-orchid}"
WANDB_ENTITY="${WANDB_ENTITY:-flair}"   # override with your own wandb entity

TRANSFER_ARG=""
[ -n "$TRANSFER_BACKEND" ] && TRANSFER_ARG="--env.transfer_backend $TRANSFER_BACKEND"
REWARD_ARG=()
[ -n "$REWARD" ] && REWARD_ARG=(--env.reward "$REWARD")
PENALTY_ARG=()
[ -n "$DISRUPTION_PENALTY" ] && PENALTY_ARG=(--env.disruption-penalty "$DISRUPTION_PENALTY")

# --- environment ------------------------------------------------------------
# VENV overrides the venv to source (default the repo .venv). Useful if a
# scenario ever needs a different dependency pin without disturbing the shared
# .venv that other running jobs depend on (verified 2026-07-02: STEP does not
# need a different jax pin — 0.10.2 works fine for reset/step/jit+scan).
module purge 2>/dev/null || true
source "${VENV:-.venv}/bin/activate"

# Persist the JAX compilation cache across runs (first compile is slow).
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax-plasmax"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
# GPU matmul codegen + let JAX preallocate most of the A100. XLA_EXTRA appends
# extra flags; XLA_FLAGS_OVERRIDE replaces the whole set (e.g. drop the triton
# GEMM flag, whose autotuning compiles kernels on worker threads and trips the
# XLA reentrant-compilation assertion on some graphs, e.g. the STEP env).
if [ -n "${XLA_FLAGS_OVERRIDE:-}" ]; then
  export XLA_FLAGS="$XLA_FLAGS_OVERRIDE"
else
  export XLA_FLAGS="--xla_gpu_triton_gemm_any=True ${XLA_EXTRA:-}"
  # Vmapping NUM_SEEDS independent runs multiplies the number of "alive" CUDA
  # graphs XLA keeps around for command-buffer dispatch; on a full-discharge
  # scenario (thousands of steps, many distinct kernels) this exhausts the
  # small memory headroom left after XLA_PYTHON_CLIENT_MEM_FRACTION and OOMs
  # at graph-instantiation time (not at the main allocator). Only disable
  # command buffers when actually vmapping — the single-seed path doesn't
  # need it and command buffers help there.
  if [ "$NUM_SEEDS" -gt 1 ]; then
    export XLA_FLAGS="$XLA_FLAGS --xla_gpu_enable_command_buffer="
  fi
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}"
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
    $TIME_AWARE_ARG \
    $QUANTIZE_ARG \
    --seed "$SEED" \
    --num-seeds "$NUM_SEEDS" \
    --ppo.total_timesteps "$TOTAL_TIMESTEPS" \
    --ppo.num_envs "$NUM_ENVS" \
    --env.eval_n_envs "$EVAL_N_ENVS" \
    --wandb.mode "$WANDB_MODE" \
    --wandb.group "$WANDB_GROUP" \
    --wandb.entity "$WANDB_ENTITY"
