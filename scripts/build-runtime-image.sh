#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="$ROOT/Dockerfile.runtime-fixes"
PATCH_SCRIPT="$ROOT/runtime/apply-xgrammar-fixes.py"
BASE_DIGEST="sha256:da5cec95778bf6996660b52e28a6e51737fec69cfc3d508bf298c8a89f273ac5"
COMMIT_1="12f64b39d29282437e35be9aa5db432fb2a1a6e6"
COMMIT_2="c6e19b3be24338759a443e03c8325d76da9ee202"
IMAGE_REPOSITORY="${RUNTIME_IMAGE_REPOSITORY:-local/glm53-runtime-fixes}"

recipe_hash="$({
  printf 'Dockerfile.runtime-fixes\0'
  printf '%s' "$(<"$DOCKERFILE")"
  # Command substitution strips trailing newlines; restore the recipe file byte.
  printf '\n\0runtime/apply-xgrammar-fixes.py\0'
  printf '%s' "$(<"$PATCH_SCRIPT")"
  printf '\n'
} | sha256sum | cut -d' ' -f1)"

if [[ "${1:-}" == "--print-recipe-hash" ]]; then
  [[ $# -eq 1 ]] || { echo "--print-recipe-hash takes no arguments" >&2; exit 2; }
  printf '%s\n' "$recipe_hash"
  exit 0
fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--print-recipe-hash]" >&2; exit 2; }
command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }

image="${IMAGE_REPOSITORY}:${recipe_hash}"
docker build \
  --pull=false \
  --file "$DOCKERFILE" \
  --build-arg "RUNTIME_RECIPE_SHA256=$recipe_hash" \
  --tag "$image" \
  "$ROOT"

inspect_label() {
  docker image inspect "$image" --format "{{ index .Config.Labels \"$1\" }}"
}

[[ "$(inspect_label org.opencontainers.image.base.digest)" == "$BASE_DIGEST" ]] || {
  echo "derived image base digest label mismatch" >&2; exit 1;
}
[[ "$(inspect_label io.github.glm53.runtime.recipe.sha256)" == "$recipe_hash" ]] || {
  echo "derived image recipe label mismatch" >&2; exit 1;
}
[[ "$(inspect_label io.github.glm53.runtime.xgrammar.commit-1)" == "$COMMIT_1" ]] || {
  echo "derived image first XGrammar commit label mismatch" >&2; exit 1;
}
[[ "$(inspect_label io.github.glm53.runtime.xgrammar.commit-2)" == "$COMMIT_2" ]] || {
  echo "derived image second XGrammar commit label mismatch" >&2; exit 1;
}

docker run --rm \
  --network none \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --entrypoint python3 \
  "$image" \
  /usr/local/share/runtime-fixes/apply-xgrammar-fixes.py \
  --verify /usr/local/lib/python3.12/dist-packages

printf 'runtime_image=%s\nrecipe_sha256=%s\n' "$image" "$recipe_hash"
