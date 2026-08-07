#!/bin/bash -l
#SBATCH --account=ori_aail
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --job-name=plasmax-sac-one
#SBATCH --output=outputs/logs/%x-%A_%a.out

set -euo pipefail

REPO="${PLASMAX_REPO:-$HOME/plasmax}"
MANIFEST="${MANIFEST:-$REPO/sweeps/sac_civo/sweep_ids.tsv}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"

cd "$REPO"
mkdir -p outputs/logs
module load CUDA/12.9.1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [[ ! -x "$REPO/.venv/bin/python" ]]; then
  echo "Missing $REPO/.venv; run experiments/cluster/civo/civo_prepare.slurm first." >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing W&B sweep manifest: $MANIFEST" >&2
  exit 1
fi
if [[ ! -s "$HOME/.netrc" ]]; then
  echo "Missing $HOME/.netrc required for W&B online sweeps." >&2
  exit 1
fi
: "${TMPDIR:?CIVO did not initialize TMPDIR}"

NUM_SWEEPS="$(( $(wc -l < "$MANIFEST") - 1 ))"
if (( NUM_SWEEPS <= 0 )); then
  echo "No sweeps found in $MANIFEST" >&2
  exit 1
fi
SWEEP_INDEX="$(( TASK_ID % NUM_SWEEPS ))"
ATTEMPT="$(( TASK_ID / NUM_SWEEPS ))"
ROW="$(awk -F $'\t' -v row="$((SWEEP_INDEX + 2))" 'NR == row {print; exit}' "$MANIFEST")"
IFS=$'\t' read -r _ ENV SWEEP_ID CONFIG <<<"$ROW"

export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DIR="$REPO/outputs/wandb"
export JAX_COMPILATION_CACHE_DIR="$REPO/outputs/jax-cache/${ENV//\//_}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_triton_gemm_any=True --xla_gpu_enable_command_buffer=}"
mkdir -p "$WANDB_DIR" "$JAX_COMPILATION_CACHE_DIR"

echo "host=$(hostname) job=$SLURM_JOB_ID task=$TASK_ID attempt=$ATTEMPT env=$ENV"
echo "sweep=$SWEEP_ID config=$CONFIG count=1"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run --frozen --no-sync wandb agent --count 1 "$SWEEP_ID"
