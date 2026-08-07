#!/bin/bash
#SBATCH --job-name=plasmax-throughput
#SBATCH --partition=workq
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.out

# Benchmark one GH200 with the same 100-step vectorised rollouts used for the
# existing CPU and A100 throughput measurements.

set -euo pipefail

REPO="${PLASMAX_REPO:-${SLURM_SUBMIT_DIR:-$HOME/plasmax}}"
VENV="${VENV:-$HOME/plasmax/.venv}"
UV_PROJECT="${UV_PROJECT:-$HOME/plasmax}"
cd "$REPO"

module purge 2>/dev/null || true
module load brics/userenv
: "${SCRATCHDIR:?brics/userenv did not set SCRATCHDIR}"

RUN_DIR="${RUN_DIR:-$SCRATCHDIR/plasmax-throughput/${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR/tmp" "$SCRATCHDIR/.cache/jax-plasmax" outputs/logs

export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$RUN_DIR/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$SCRATCHDIR/.cache/jax-plasmax}"
export XLA_FLAGS="${XLA_FLAGS_OVERRIDE:---xla_gpu_triton_gemm_any=True --xla_gpu_enable_command_buffer=}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

source "$VENV/bin/activate"

read -r -a BENCH_N_SEEDS_ARRAY <<< "${BENCH_N_SEEDS:-1 4 16 64 256 1024 4096 16384}"

echo "host=$(hostname) job=$SLURM_JOB_ID arch=$(uname -m) repo=$REPO"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run --project "$UV_PROJECT" --no-sync python -c "import jax, torax; print('jax', jax.__version__, 'torax', torax.__version__, 'devices', jax.devices())"

srun uv run --project "$UV_PROJECT" --no-sync python "$REPO/benchmarks/env_throughput.py" \
  --env-setup "${ENV_SETUP:-iter/hybrid}" \
  --backend "${BACKEND:-cgm}" \
  --skip-python-loop \
  --n-seeds "${BENCH_N_SEEDS_ARRAY[@]}"
