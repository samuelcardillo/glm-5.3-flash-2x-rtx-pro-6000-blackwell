#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OVERLAY_DIR="$ROOT/overlays/dflash-dcp-block-table"
BASE_IMAGE='ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:v0.6.0@sha256:fe249b88d091430d8a88cd987d087d556053f0f067a649f2e9ca95895129e82b'
BASE_IMAGE_ID='sha256:fe249b88d091430d8a88cd987d087d556053f0f067a649f2e9ca95895129e82b'
TARGET_PATH='/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py'

recipe_sha256=$(
  for name in Dockerfile patch-model-runner.py; do
    path="$OVERLAY_DIR/$name"
    printf '%s\0%s\0' "$name" "$(stat -c '%s' "$path")"
    command cat "$path"
  done | sha256sum | cut -d' ' -f1
)
tag="glm53-v06-dflash-dcp:${recipe_sha256:0:16}"

if [[ "${1:-}" == --print-recipe ]]; then
  printf 'recipe_sha256=%s\ntag=%s\n' "$recipe_sha256" "$tag"
  exit 0
fi
[[ $# == 0 ]] || { echo 'usage: build-dflash-dcp-overlay.sh [--print-recipe]' >&2; exit 2; }

if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  docker pull "$BASE_IMAGE"
fi
actual_base_id=$(docker image inspect "$BASE_IMAGE" --format '{{.Id}}')
[[ "$actual_base_id" == "$BASE_IMAGE_ID" ]] || {
  echo "Base image ID mismatch: expected $BASE_IMAGE_ID, got $actual_base_id" >&2
  exit 1
}

docker build --pull=false \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "BASE_IMAGE_ID=$BASE_IMAGE_ID" \
  --build-arg "RECIPE_SHA256=$recipe_sha256" \
  --tag "$tag" "$OVERLAY_DIR"

image_id=$(docker image inspect "$tag" --format '{{.Id}}')
[[ "$image_id" == sha256:* ]] || { echo 'Built image did not resolve to an immutable ID' >&2; exit 1; }
[[ "$(docker image inspect "$image_id" --format '{{index .Config.Labels "org.nous.glm53.base-image-id"}}')" == "$BASE_IMAGE_ID" ]]
[[ "$(docker image inspect "$image_id" --format '{{index .Config.Labels "org.nous.glm53.overlay-recipe-sha256"}}')" == "$recipe_sha256" ]]

docker run --rm --network none --cap-drop ALL --entrypoint python3 "$image_id" \
  /opt/glm53-overlay/patch-model-runner.py --verify "$TARGET_PATH"

printf 'recipe_sha256=%s\ntag=%s\nimage_id=%s\n' "$recipe_sha256" "$tag" "$image_id"
