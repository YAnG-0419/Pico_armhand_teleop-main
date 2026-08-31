#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source_image=${PICO_TELEOP_SOURCE_IMAGE:-franka-upper-body-teleop:latest}
if docker image inspect "$source_image" >/dev/null 2>&1; then
  echo "Reusing existing image: $source_image"
  docker build \
    --build-arg "SOURCE_IMAGE=$source_image" \
    --file "${repo}/docker/Dockerfile.reuse" \
    --tag pico-armhand-teleop:latest \
    "${repo}/docker"
else
  echo "Existing source image not found; performing a full standalone build."
  docker build --tag pico-armhand-teleop:latest "${repo}/docker"
fi
docker run --rm \
  --volume "${repo}:/workspace/pico_armhand_teleop" \
  --workdir /workspace/pico_armhand_teleop \
  pico-armhand-teleop:latest \
  docker/build_workspace.sh "$@"
