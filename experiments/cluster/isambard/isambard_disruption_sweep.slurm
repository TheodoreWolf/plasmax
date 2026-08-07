#!/bin/bash
#SBATCH --job-name=ppo-disruption-kappa
#SBATCH --partition=workq
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.out

set -euo pipefail

REPO="${PLASMAX_REPO:-${SLURM_SUBMIT_DIR:-$HOME/plasmax}}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"

MANIFEST="${MANIFEST:-sweeps/ppo_disruption_kappa_isambard.tsv}"
STUDY="${STUDY:-ppo-disruption-kappa-pilot}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"
LINE_NUMBER=$((TASK_ID + 2))
ROW=$(sed -n "${LINE_NUMBER}p" "$MANIFEST")
if [ -z "$ROW" ]; then
  echo "No manifest row for task ${TASK_ID} in ${MANIFEST}" >&2
  exit 2
fi

IFS=$'\t' read -r MANIFEST_TASK ENV_SETUP BACKEND VARIANT REWARD TASK_TERMINAL_PENALTY KAPPA DISRUPTION_PENALTY TOTAL_STEPS EVAL_FREQ NUM_SEEDS SEED SLUG <<< "$ROW"
if [ "$MANIFEST_TASK" != "$TASK_ID" ]; then
  echo "Manifest task ${MANIFEST_TASK} does not match array task ${TASK_ID}" >&2
  exit 2
fi

TOTAL_STEPS="${TOTAL_STEPS_OVERRIDE:-$TOTAL_STEPS}"
EVAL_FREQ="${EVAL_FREQ_OVERRIDE:-$EVAL_FREQ}"
NUM_SEEDS="${NUM_SEEDS_OVERRIDE:-$NUM_SEEDS}"

module purge 2>/dev/null || true
module load brics/userenv
: "${SCRATCHDIR:?brics/userenv did not set SCRATCHDIR}"

SWEEP_ROOT="${SWEEP_ROOT:-$SCRATCHDIR/plasmax-sweeps}"
RUN_DIR="$SWEEP_ROOT/$STUDY/${SLURM_ARRAY_JOB_ID}_${TASK_ID}-${SLUG}"
mkdir -p "$RUN_DIR" outputs/logs
printf '%s\n%s\n' \
  "$(head -n 1 "$MANIFEST")" \
  "$ROW" > "$RUN_DIR/manifest_row.tsv"

export TMPDIR="$RUN_DIR/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

source "${VENV:-.venv}/bin/activate"

export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$SCRATCHDIR/.cache/jax-plasmax}"
export XLA_FLAGS="${XLA_FLAGS_OVERRIDE:---xla_gpu_triton_gemm_any=True --xla_gpu_enable_command_buffer=}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export WANDB_DIR="$RUN_DIR/wandb"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$WANDB_DIR"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} task=$TASK_ID arch=$(uname -m)"
echo "env=$ENV_SETUP backend=$BACKEND variant=$VARIANT reward=$REWARD"
echo "task_penalty=$TASK_TERMINAL_PENALTY kappa=$KAPPA penalty=$DISRUPTION_PENALTY"
echo "steps=$TOTAL_STEPS seeds=$NUM_SEEDS output=$RUN_DIR"
nvidia-smi -L
uv run --no-sync python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"

srun uv run --no-sync python scripts/train_ppo.py \
  --env.env-setup "$ENV_SETUP" \
  --env.backend "$BACKEND" \
  --env.reward "$REWARD" \
  --env.variant "$VARIANT" \
  --env.eval-n-envs "${EVAL_N_ENVS:-16}" \
  --env.deterministic-eval \
  --env.force-minimal-callback \
  --env.disruption-kappa "$KAPPA" \
  --env.disruption-penalty "$DISRUPTION_PENALTY" \
  --ppo.total-timesteps "$TOTAL_STEPS" \
  --ppo.eval-freq "$EVAL_FREQ" \
  --ppo.num-envs "${PPO_NUM_ENVS:-256}" \
  --seed "$SEED" \
  --num-seeds "$NUM_SEEDS" \
  --history-dir "$RUN_DIR" \
  --study "$STUDY" \
  --wandb.group "$STUDY" \
  --wandb.mode "${WANDB_MODE:-online}" \
  --wandb.entity "${WANDB_ENTITY:-flair}" \
  --wandb.tags "disruption-penalty-sweep" "kappa-$KAPPA"
