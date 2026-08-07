#!/bin/bash
#SBATCH --job-name=plasmax-baseline
#SBATCH --partition=workq
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.out

# Submit the generated manifest as one array; Slurm fair-share and QOS decide
# how many tasks run concurrently unless a measured project limit says otherwise:
#   MANIFEST=sweeps/baseline_v1.tsv STUDY=baseline-v1 \
#     sbatch --array=0-191 experiments/cluster/isambard/isambard_baselines.slurm

set -euo pipefail

# ``--chdir`` controls the job's working directory but Slurm still records the
# client's submission directory in ``SLURM_SUBMIT_DIR``.  Allow callers that
# submit from elsewhere to name the repository explicitly.
REPO="${PLASMAX_REPO:-${SLURM_SUBMIT_DIR:-$HOME/plasmax}}"
cd "$REPO"
# Use the current src-layout package and repository-only training modules even
# when an older editable install left a top-level plasmax/ tree on the cluster.
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"

MANIFEST="${MANIFEST:-sweeps/baseline_v1.tsv}"
STUDY="${STUDY:-baseline-v1}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"
LINE_NUMBER=$((TASK_ID + 2))
ROW=$(sed -n "${LINE_NUMBER}p" "$MANIFEST")
if [ -z "$ROW" ]; then
  echo "No manifest row for task ${TASK_ID} in ${MANIFEST}" >&2
  exit 2
fi

IFS=$'\t' read -r MANIFEST_TASK ALGORITHM ENV_SETUP BACKEND VARIANT REWARD TOTAL_STEPS EVAL_FREQ NUM_SEEDS SEED SLUG <<< "$ROW"
if [ "$MANIFEST_TASK" != "$TASK_ID" ]; then
  echo "Manifest task ${MANIFEST_TASK} does not match array task ${TASK_ID}" >&2
  exit 2
fi

module purge 2>/dev/null || true
module load brics/userenv
: "${SCRATCHDIR:?brics/userenv did not set SCRATCHDIR}"

BASELINE_ROOT="${BASELINE_ROOT:-$SCRATCHDIR/plasmax-baselines}"
RUN_DIR="$BASELINE_ROOT/$STUDY/${SLURM_ARRAY_JOB_ID}_${TASK_ID}-${SLUG}"
mkdir -p "$RUN_DIR" outputs/logs

# Some compute nodes expose a stale /local/user/$UID as TMPDIR.  Give Python,
# W&B, XLA, and ptxas a per-job directory that is guaranteed to exist for the
# full allocation instead of relying on that node-local path.
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
echo "algorithm=$ALGORITHM env=$ENV_SETUP backend=$BACKEND variant=$VARIANT reward=$REWARD"
echo "steps=$TOTAL_STEPS seeds=$NUM_SEEDS output=$RUN_DIR"
nvidia-smi -L
uv run --no-sync python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"

case "$ALGORITHM" in
  ppo)
    srun uv run --no-sync python scripts/train_ppo.py \
      --env.env-setup "$ENV_SETUP" \
      --env.backend "$BACKEND" \
      --env.reward "$REWARD" \
      --env.variant "$VARIANT" \
      --env.eval-n-envs "${PPO_EVAL_N_ENVS:-${EVAL_N_ENVS:-16}}" \
      --env.eval-seed "${PPO_EVAL_SEED:-0}" \
      --env.deterministic-eval \
      --env.force-minimal-callback \
      --ppo.total-timesteps "$TOTAL_STEPS" \
      --ppo.eval-freq "$EVAL_FREQ" \
      --ppo.num-envs "${PPO_NUM_ENVS:-1024}" \
      --seed "$SEED" \
      --num-seeds "$NUM_SEEDS" \
      --history-dir "$RUN_DIR" \
      --study "$STUDY" \
      --wandb.group "$STUDY" \
      --wandb.mode "${WANDB_MODE:-online}" \
      --wandb.entity "${WANDB_ENTITY:-flair}"
    ;;
  sac)
    srun uv run --no-sync python scripts/train_sac.py \
      --env.env-setup "$ENV_SETUP" \
      --env.backend "$BACKEND" \
      --env.reward "$REWARD" \
      --env.variant "$VARIANT" \
      --env.eval-n-envs "${SAC_EVAL_N_ENVS:-${EVAL_N_ENVS:-16}}" \
      --env.eval-seed "${SAC_EVAL_SEED:-10000}" \
      --sac.total-timesteps "$TOTAL_STEPS" \
      --sac.eval-freq "$EVAL_FREQ" \
      --sac.num-envs "${SAC_NUM_ENVS:-64}" \
      --sac.num-epochs "${SAC_NUM_EPOCHS:-64}" \
      --seed "$SEED" \
      --num-seeds "$NUM_SEEDS" \
      --history-dir "$RUN_DIR" \
      --study "$STUDY" \
      --wandb.group "$STUDY" \
      --wandb.mode "${WANDB_MODE:-online}" \
      --wandb.entity "${WANDB_ENTITY:-flair}"
    ;;
  direct_policy|direct_knots_1|direct_knots_10|direct_knots_100)
    DIRECT_ROLLOUT_ARG=""
    if [ -n "${DIRECT_NUM_ROLLOUTS:-}" ]; then
      DIRECT_ROLLOUT_ARG="--direct.num-rollouts $DIRECT_NUM_ROLLOUTS"
    fi
    srun uv run --no-sync python experiments/studies/train_direct_baseline.py \
      --algorithm "$ALGORITHM" \
      --env.env-setup "$ENV_SETUP" \
      --env.backend "$BACKEND" \
      --env.reward "$REWARD" \
      --env.variant "$VARIANT" \
      --env.eval-n-envs "${DIRECT_EVAL_N_ENVS:-${EVAL_N_ENVS:-4}}" \
      --env.eval-seed "${DIRECT_EVAL_SEED:-10000}" \
      --direct.total-timesteps "$TOTAL_STEPS" \
      --direct.eval-freq "$EVAL_FREQ" \
      --direct.gradient-horizon "${DIRECT_GRADIENT_HORIZON:-32}" \
      --direct.policy-learning-rate "${DIRECT_POLICY_LR:-0.000001}" \
      --direct.knot-learning-rate "${DIRECT_KNOT_LR:-0.05}" \
      --direct.nonfinite-backoff-factor "${DIRECT_NONFINITE_BACKOFF_FACTOR:-0.5}" \
      --direct.min-update-scale "${DIRECT_MIN_UPDATE_SCALE:-0.001}" \
      $DIRECT_ROLLOUT_ARG \
      --seed "$SEED" \
      --num-seeds "$NUM_SEEDS" \
      --history-dir "$RUN_DIR" \
      --study "$STUDY" \
      --wandb.group "$STUDY" \
      --wandb.mode "${WANDB_MODE:-online}" \
      --wandb.entity "${WANDB_ENTITY:-flair}"
    ;;
  *)
    echo "Unknown algorithm in manifest: $ALGORITHM" >&2
    exit 2
    ;;
esac
