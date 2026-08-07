#!/usr/bin/env bash
# Usage: ./experiments/cluster/docker/run_docker.sh [gpu_ids] [command...]
#   gpu_ids: comma-separated IDs, 'all', or 'none' (CPU). Default: all
#
# Examples:
#   ./experiments/cluster/docker/run_docker.sh
#   ./experiments/cluster/docker/run_docker.sh 0
#   ./experiments/cluster/docker/run_docker.sh 0,1 python3 scripts/train_ppo.py
#   ./experiments/cluster/docker/run_docker.sh none pytest tests/

set -euo pipefail

USER=$(whoami)
IMAGE="${PLASMAX_DOCKER_IMAGE:-${USER}-plasmax}"
DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DOCKER_DIR/../../.." && pwd)"

# Parse GPU arg
GPU_IDS="${1:-all}"
[[ $# -gt 0 ]] && shift
GPU_LABEL=$(echo "$GPU_IDS" | tr ',' '-')

if [[ "$GPU_IDS" == "none" ]]; then
    GPU_ARG=""
elif [[ "$GPU_IDS" == "all" ]]; then
    GPU_ARG="--gpus all"
else
    GPU_ARG="--gpus device=$GPU_IDS"
fi

# .env passthrough
ENV_ARG=""
[[ -f "$REPO_ROOT/.env" ]] && ENV_ARG="--env-file $REPO_ROOT/.env"

CONTAINER_NAME="${IMAGE}-${GPU_LABEL}"
CMD=("${@:-bash}")

# Use -it only when running in a terminal; omit for non-interactive/background use.
IT_ARG=""
[ -t 0 ] && IT_ARG="-it"

docker run $IT_ARG --rm \
    $GPU_ARG \
    $ENV_ARG \
    --shm-size=4g \
    --name "$CONTAINER_NAME" \
    -v "$REPO_ROOT:/workspace" \
    -w /workspace \
    "$IMAGE" \
    "${CMD[@]}"
