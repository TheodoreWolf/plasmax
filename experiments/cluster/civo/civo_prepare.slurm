#!/bin/bash -l
#SBATCH --account=ori_aail
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=plasmax-prepare
#SBATCH --output=outputs/logs/%x-%j.out

set -euo pipefail

REPO="${PLASMAX_REPO:-$HOME/plasmax}"
cd "$REPO"
mkdir -p outputs/logs

module load CUDA/12.9.1

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.11.28/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

uv --version
uv sync --frozen --no-dev --group research
uv pip install \
  --python "$REPO/.venv/bin/python" \
  "jax[cuda12]==0.10.1" \
  "jaxlib==0.10.1"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run --frozen --no-sync python -c "import jax; import jaxlib; devices = jax.devices(); print('jax', jax.__version__, 'jaxlib', jaxlib.__version__, 'devices', devices); assert devices and devices[0].platform == 'gpu'"
uv run --frozen --no-sync python scripts/train_sac.py --help >/dev/null

echo "CIVO plasmax environment is ready."
