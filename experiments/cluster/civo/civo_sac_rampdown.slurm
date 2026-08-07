#!/bin/bash -l
#SBATCH --account=ori_aail
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --job-name=plasmax-sac-rampdown
#SBATCH --output=outputs/logs/%x-%j.out

set -euo pipefail

REPO="${PLASMAX_REPO:-$HOME/plasmax}"
RUN_ROOT="${RUN_ROOT:-$REPO/outputs/civo-sac-iter-hybrid-rampdown}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_GROUP="${WANDB_GROUP:-civo-sac-rampdown}"
RUN_DIR="$RUN_ROOT/${SLURM_JOB_ID}-sac-iter_hybrid_rampdown-realistic-10seeds"

cd "$REPO"
mkdir -p outputs/logs
module load CUDA/12.9.1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [[ ! -x "$REPO/.venv/bin/python" ]]; then
  echo "Missing $REPO/.venv; run experiments/cluster/civo/civo_prepare.slurm first." >&2
  exit 1
fi
: "${TMPDIR:?CIVO did not initialize TMPDIR}"

mkdir -p "$RUN_DIR/wandb" "$TMPDIR/jax-cache"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DIR="$RUN_DIR/wandb"
export JAX_COMPILATION_CACHE_DIR="$TMPDIR/jax-cache"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_triton_gemm_any=True --xla_gpu_enable_command_buffer=}"

echo "host=$(hostname) job=$SLURM_JOB_ID run_dir=$RUN_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run --frozen --no-sync python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"

uv run --frozen --no-sync python scripts/train_sac.py \
  --env.env-setup iter/hybrid/rampdown \
  --env.backend bohm_gyrobohm \
  --env.reward rampdown \
  --env.variant realistic \
  --env.eval-n-envs 16 \
  --sac.total-timesteps 10000000 \
  --sac.eval-freq 1000000 \
  --sac.num-envs 64 \
  --sac.num-epochs 64 \
  --seed 0 \
  --num-seeds 10 \
  --history-dir "$RUN_DIR" \
  --study civo-sac-rampdown \
  --wandb.group "$WANDB_GROUP" \
  --wandb.mode "$WANDB_MODE" \
  --wandb.entity flair
