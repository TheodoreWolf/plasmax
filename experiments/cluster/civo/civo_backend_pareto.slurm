#!/bin/bash -l
#SBATCH --account=ori_aail
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=plasmax-backend-agreement
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO="${PLASMAX_REPO:-$HOME/plasmax}"
OUTPUT="${OUTPUT:-$REPO/outputs/backend_agreement_flattop_a100.json}"

cd "$REPO"
module load CUDA/12.9.1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [[ ! -x "$REPO/.venv/bin/python" ]]; then
  echo "Missing $REPO/.venv; run experiments/cluster/civo/civo_prepare.slurm first." >&2
  exit 1
fi
: "${TMPDIR:?CIVO did not initialize TMPDIR}"

export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$REPO/.jax-cache}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_triton_gemm_any=True --xla_gpu_enable_command_buffer=}"
mkdir -p "$(dirname "$OUTPUT")" "$JAX_COMPILATION_CACHE_DIR"

echo "host=$(hostname) job=$SLURM_JOB_ID output=$OUTPUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run --frozen --no-sync python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"

uv run --frozen --no-sync python benchmarks/backend_agreement.py \
  --env-setup iter/hybrid/flattop \
  --backends cgm bohm_gyrobohm qlknn tglfnn tglfnn_nr \
  --reference-backend tglfnn_nr \
  --n-steps 100 \
  --timing-repeats 3 \
  --output "$OUTPUT" \
  --no-require-cpu
