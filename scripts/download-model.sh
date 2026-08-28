#!/usr/bin/env bash
set -euo pipefail
MODEL_REPO="brandonmusic/GLM-5.3-Flash-tr3-4bpw"
MODEL_REVISION="5ab363a8dcf6405955fd5f99671e01a1c9fb124b"
DESTINATION="${1:-${MODEL_DIR:-}}"
if [[ "${I_ACCEPT_SHAPLEYMCG_LICENSE:-}" != yes ]]; then
  cat >&2 <<'EOF'
This checkpoint is source-available under the ShapleyMCG License 1.0, not an
OSI-approved open-source license. Review:
https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/blob/5ab363a8dcf6405955fd5f99671e01a1c9fb124b/LICENSE
Then rerun with I_ACCEPT_SHAPLEYMCG_LICENSE=yes if its terms apply to you.
EOF
  exit 2
fi
[[ -n "$DESTINATION" && "$DESTINATION" == /* ]] || { echo 'Pass an absolute destination or set MODEL_DIR.' >&2; exit 2; }
command -v hf >/dev/null || { echo 'Missing Hugging Face hf CLI.' >&2; exit 1; }
hf download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$DESTINATION"
printf '%s\n' "${MODEL_REPO}@${MODEL_REVISION}" > "$DESTINATION/RECIPE_PIN.txt"
printf 'Downloaded %s at %s into %s\n' "$MODEL_REPO" "$MODEL_REVISION" "$DESTINATION"
