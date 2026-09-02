#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/repo/blobs" "$tmp/repo/snapshots/abc123" "$tmp/direct"
# shellcheck source=scripts/resolve-model-mount.sh
source "$ROOT/scripts/resolve-model-mount.sh"
resolve_model_mount "$tmp/repo/snapshots/abc123" model
[[ "$MOUNT_SOURCE" == "$tmp/repo" ]]
[[ "$MOUNT_TARGET" == /model-repo ]]
[[ "$CONTAINER_MODEL_PATH" == /model-repo/snapshots/abc123 ]]
resolve_model_mount "$tmp/direct" draft
[[ "$MOUNT_SOURCE" == "$tmp/direct" ]]
[[ "$MOUNT_TARGET" == /draft ]]
[[ "$CONTAINER_MODEL_PATH" == /draft ]]
if resolve_model_mount "$tmp/repo/snapshots/abc123/extra" model 2>/dev/null; then
  echo 'nested snapshot suffix was accepted' >&2; exit 1
fi
echo 'model mount resolution tests passed'
