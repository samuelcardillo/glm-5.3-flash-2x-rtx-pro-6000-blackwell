#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="$ROOT/Dockerfile.runtime-fixes-a2"
WORKSPACE_SCRIPT="$ROOT/runtime/apply-sparse-indexer-workspace.py"
BUILD_SCRIPT="$ROOT/scripts/build-runtime-image-a2.sh"
PARENT_RECIPE="d6460a952a88786828a39f44fb99b417144450047dcff446813e4480eb17a8fc"
PARENT_IMAGE="local/glm53-runtime-fixes:${PARENT_RECIPE}"
BASE_DIGEST="sha256:da5cec95778bf6996660b52e28a6e51737fec69cfc3d508bf298c8a89f273ac5"
COMMIT_1="12f64b39d29282437e35be9aa5db432fb2a1a6e6"
COMMIT_2="c6e19b3be24338759a443e03c8325d76da9ee202"
IMAGE_REPOSITORY="${RUNTIME_IMAGE_REPOSITORY:-local/glm53-runtime-fixes}"

recipe_hash="$(python3 - "$DOCKERFILE" "$WORKSPACE_SCRIPT" "$BUILD_SCRIPT" <<'PY'
import hashlib, pathlib, sys
items = (
    ("Dockerfile.runtime-fixes-a2", pathlib.Path(sys.argv[1])),
    ("runtime/apply-sparse-indexer-workspace.py", pathlib.Path(sys.argv[2])),
    ("scripts/build-runtime-image-a2.sh", pathlib.Path(sys.argv[3])),
)
h = hashlib.sha256()
for name, path in items:
    name_bytes = name.encode()
    data = path.read_bytes()
    h.update(len(name_bytes).to_bytes(8, "big"))
    h.update(name_bytes)
    h.update(len(data).to_bytes(8, "big"))
    h.update(data)
print(h.hexdigest())
PY
)"

if [[ "${1:-}" == "--print-recipe-hash" ]]; then
  [[ $# -eq 1 ]] || { echo "--print-recipe-hash takes no arguments" >&2; exit 2; }
  printf '%s\n' "$recipe_hash"
  exit 0
fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--print-recipe-hash]" >&2; exit 2; }
command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }
docker image inspect "$PARENT_IMAGE" >/dev/null 2>&1 || {
  echo "exact A1 parent image is not available locally: $PARENT_IMAGE" >&2; exit 1;
}
parent_label() {
  docker image inspect "$PARENT_IMAGE" --format "{{ index .Config.Labels \"$1\" }}"
}
[[ "$(parent_label io.github.glm53.runtime.recipe.sha256)" == "$PARENT_RECIPE" ]] || {
  echo "A1 parent recipe label mismatch" >&2; exit 1;
}
[[ "$(parent_label org.opencontainers.image.base.digest)" == "$BASE_DIGEST" ]] || {
  echo "A1 parent base digest label mismatch" >&2; exit 1;
}

image="${IMAGE_REPOSITORY}:${recipe_hash}"
docker build --pull=false --file "$DOCKERFILE" \
  --build-arg "RUNTIME_RECIPE_SHA256=$recipe_hash" --tag "$image" "$ROOT"

inspect_label() {
  docker image inspect "$image" --format "{{ index .Config.Labels \"$1\" }}"
}
[[ "$(inspect_label org.opencontainers.image.base.digest)" == "$BASE_DIGEST" ]] || {
  echo "A2 base digest label mismatch" >&2; exit 1;
}
[[ "$(inspect_label io.github.glm53.runtime.parent.recipe.sha256)" == "$PARENT_RECIPE" ]] || {
  echo "A2 parent recipe label mismatch" >&2; exit 1;
}
[[ "$(inspect_label io.github.glm53.runtime.recipe.sha256)" == "$recipe_hash" ]] || {
  echo "A2 recipe label mismatch" >&2; exit 1;
}
[[ "$(inspect_label io.github.glm53.runtime.xgrammar.commit-1)" == "$COMMIT_1" ]] || {
  echo "A2 inherited XGrammar commit 1 mismatch" >&2; exit 1;
}
[[ "$(inspect_label io.github.glm53.runtime.xgrammar.commit-2)" == "$COMMIT_2" ]] || {
  echo "A2 inherited XGrammar commit 2 mismatch" >&2; exit 1;
}
[[ "$(inspect_label io.github.glm53.runtime.workspace)" == "dcp-aware-sparse-indexer-v1" ]] || {
  echo "A2 workspace label mismatch" >&2; exit 1;
}
[[ "$(inspect_label io.github.glm53.runtime.workspace.global-cap-rows)" == "1048576" ]] || {
  echo "A2 global cap label mismatch" >&2; exit 1;
}
[[ "$(inspect_label io.github.glm53.runtime.workspace.local-cap-rows)" == "524288" ]] || {
  echo "A2 local cap label mismatch" >&2; exit 1;
}

for verifier in apply-xgrammar-fixes.py apply-sparse-indexer-workspace.py; do
  docker run --rm --network none --cap-drop ALL \
    --security-opt no-new-privileges --entrypoint python3 "$image" \
    "/usr/local/share/runtime-fixes/$verifier" --verify \
    /usr/local/lib/python3.12/dist-packages
done

printf 'runtime_image=%s\nparent_recipe_sha256=%s\nrecipe_sha256=%s\n' \
  "$image" "$PARENT_RECIPE" "$recipe_hash"
