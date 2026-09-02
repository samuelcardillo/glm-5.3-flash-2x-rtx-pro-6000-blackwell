#!/usr/bin/env bash
# Resolve a self-contained model directory or a Hugging Face cache snapshot.
# Sets MOUNT_SOURCE, MOUNT_TARGET, and CONTAINER_MODEL_PATH.
resolve_model_mount() {
  local host_path=${1:?host model path required}
  local label=${2:?mount label required}
  [[ "$label" == model || "$label" == draft ]] || { echo 'mount label must be model or draft' >&2; return 2; }
  if [[ "$host_path" == */snapshots/* ]]; then
    local repo_root=${host_path%%/snapshots/*}
    local revision=${host_path#"$repo_root/snapshots/"}
    [[ -n "$revision" && "$revision" != */* && -d "$repo_root/blobs" ]] || {
      echo "invalid Hugging Face snapshot path: $host_path" >&2
      return 2
    }
    MOUNT_SOURCE=$repo_root
    MOUNT_TARGET="/${label}-repo"
    CONTAINER_MODEL_PATH="${MOUNT_TARGET}/snapshots/${revision}"
  else
    # shellcheck disable=SC2034 # function outputs consumed by sourcing caller
    MOUNT_SOURCE=$host_path
    MOUNT_TARGET="/$label"
    # shellcheck disable=SC2034 # function outputs consumed by sourcing caller
    CONTAINER_MODEL_PATH=$MOUNT_TARGET
  fi
}
