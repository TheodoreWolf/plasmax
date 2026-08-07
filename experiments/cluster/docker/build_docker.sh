#!/usr/bin/env bash
# Usage: ./experiments/cluster/docker/build_docker.sh [tag]
# Default tag: $(whoami)-plasmax

set -euo pipefail

TAG="${1:-$(whoami)-plasmax}"
DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DOCKER_DIR/../../.." && pwd)"

docker build -f "$DOCKER_DIR/Dockerfile" -t "$TAG" "$REPO_ROOT" \
    --build-arg USERNAME=$(whoami) \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g)
echo "Built: $TAG"
