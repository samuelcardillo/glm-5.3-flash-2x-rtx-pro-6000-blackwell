#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="$ROOT/Dockerfile.runtime-fixes-a3"
MIXED_PREFILL_SCRIPT="$ROOT/runtime/apply-mixed-prefill-policy.py"
BUILD_SCRIPT="$ROOT/scripts/build-runtime-image-a3.sh"
PARENT_RECIPE="4ef38e761892c69e7c8e90748dbc362dca405cd6cde756b4afafb68cc0babd39"
PARENT_DIGEST="sha256:ca68a67e14b77c4291a19925d7ff262ff63805cdd90834250ab7a6d7438a54a6"
PARENT_IMAGE="local/glm53-runtime-fixes@${PARENT_DIGEST}"
A1_RECIPE="e91aebecd2907d9905c6f4520c30d49fa57f4272e9e738d46c0d3edccf3d35fc"
BASE_DIGEST="sha256:da5cec95778bf6996660b52e28a6e51737fec69cfc3d508bf298c8a89f273ac5"
COMMIT_1="12f64b39d29282437e35be9aa5db432fb2a1a6e6"
COMMIT_2="c6e19b3be24338759a443e03c8325d76da9ee202"
IMAGE_REPOSITORY="${RUNTIME_IMAGE_REPOSITORY:-local/glm53-runtime-fixes}"

recipe_hash="$(python3 - "$DOCKERFILE" "$MIXED_PREFILL_SCRIPT" "$BUILD_SCRIPT" <<'PY'
import hashlib, pathlib, sys
items = (
    ("Dockerfile.runtime-fixes-a3", pathlib.Path(sys.argv[1])),
    ("runtime/apply-mixed-prefill-policy.py", pathlib.Path(sys.argv[2])),
    ("scripts/build-runtime-image-a3.sh", pathlib.Path(sys.argv[3])),
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
  echo "immutable A2 parent image is not available locally: $PARENT_IMAGE" >&2; exit 1;
}
[[ "$(docker image inspect "$PARENT_IMAGE" --format '{{.Id}}')" == "$PARENT_DIGEST" ]] || {
  echo "A2 parent image digest mismatch" >&2; exit 1;
}
parent_label() {
  docker image inspect "$PARENT_IMAGE" --format "{{ index .Config.Labels \"$1\" }}"
}
[[ "$(parent_label io.github.glm53.runtime.recipe.sha256)" == "$PARENT_RECIPE" ]] || {
  echo "A2 parent recipe label mismatch" >&2; exit 1;
}
[[ "$(parent_label io.github.glm53.runtime.parent.recipe.sha256)" == "$A1_RECIPE" ]] || {
  echo "A2 parent chain label mismatch" >&2; exit 1;
}
[[ "$(parent_label org.opencontainers.image.base.digest)" == "$BASE_DIGEST" ]] || {
  echo "A2 parent base digest label mismatch" >&2; exit 1;
}
[[ "$(parent_label io.github.glm53.runtime.workspace)" == "dcp-aware-sparse-indexer-v1" ]] || {
  echo "A2 parent workspace label mismatch" >&2; exit 1;
}

image="${IMAGE_REPOSITORY}:${recipe_hash}"
docker build --pull=false --file "$DOCKERFILE" \
  --build-arg "RUNTIME_RECIPE_SHA256=$recipe_hash" --tag "$image" "$ROOT"

inspect_label() {
  docker image inspect "$image" --format "{{ index .Config.Labels \"$1\" }}"
}
[[ "$(inspect_label org.opencontainers.image.base.digest)" == "$BASE_DIGEST" ]] || { echo "A3 base digest label mismatch" >&2; exit 1; }
[[ "$(inspect_label io.github.glm53.runtime.parent.recipe.sha256)" == "$PARENT_RECIPE" ]] || { echo "A3 parent recipe label mismatch" >&2; exit 1; }
[[ "$(inspect_label io.github.glm53.runtime.recipe.sha256)" == "$recipe_hash" ]] || { echo "A3 recipe label mismatch" >&2; exit 1; }
[[ "$(inspect_label io.github.glm53.runtime.xgrammar.commit-1)" == "$COMMIT_1" ]] || { echo "A3 inherited XGrammar commit 1 mismatch" >&2; exit 1; }
[[ "$(inspect_label io.github.glm53.runtime.xgrammar.commit-2)" == "$COMMIT_2" ]] || { echo "A3 inherited XGrammar commit 2 mismatch" >&2; exit 1; }
[[ "$(inspect_label io.github.glm53.runtime.workspace)" == "dcp-aware-sparse-indexer-v1" ]] || { echo "A3 inherited workspace mismatch" >&2; exit 1; }
[[ "$(inspect_label io.github.glm53.runtime.mixed-prefill)" == "exact-policy-v1" ]] || { echo "A3 mixed-prefill label mismatch" >&2; exit 1; }

for verifier in apply-xgrammar-fixes.py apply-sparse-indexer-workspace.py apply-mixed-prefill-policy.py; do
  docker run --rm --network none --cap-drop ALL \
    --security-opt no-new-privileges --entrypoint python3 "$image" \
    "/usr/local/share/runtime-fixes/$verifier" --verify \
    /usr/local/lib/python3.12/dist-packages
done

printf 'runtime_image=%s\nparent_recipe_sha256=%s\nrecipe_sha256=%s\n' \
  "$image" "$PARENT_RECIPE" "$recipe_hash"
