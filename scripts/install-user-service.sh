#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
[[ -f "$ENV_FILE" ]] || { echo 'Create and review .env first.' >&2; exit 2; }
# shellcheck source=scripts/load-env.sh
source "$ROOT/scripts/load-env.sh"
# shellcheck source=scripts/defaults.sh
source "$ROOT/scripts/defaults.sh"
[[ "$PORT" =~ ^[0-9]+$ && "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] || { echo 'Invalid service readiness numeric values' >&2; exit 2; }
[[ "$SERVED_MODEL_NAME" =~ ^[A-Za-z0-9._/-]+$ && "$CONTAINER_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid service readiness identifiers' >&2; exit 2; }
[[ "$BIND_ADDRESS" =~ ^[A-Za-z0-9.:_-]+$ ]] || { echo 'Invalid service bind address' >&2; exit 2; }
case "$BIND_ADDRESS" in
  0.0.0.0) READINESS_HOST=127.0.0.1 ;;
  ::|::1) READINESS_HOST='[::1]' ;;
  *) READINESS_HOST="$BIND_ADDRESS" ;;
esac
UNIT_DIR="$HOME/.config/systemd/user"; ENV_DIR="$HOME/.config/glm53-2x-rtxpro6000"
for path in "$ROOT" "$UNIT_DIR" "$ENV_DIR"; do
  [[ "$path" =~ ^/[A-Za-z0-9._/-]+$ ]] || {
    echo "Unsupported path for systemd rendering (spaces, %, and specifiers are rejected): $path" >&2
    exit 2
  }
done
mkdir -p "$UNIT_DIR" "$ENV_DIR"; install -m 600 "$ENV_FILE" "$ENV_DIR/env"
python3 - "$ROOT" "$ENV_DIR/env" "$ROOT/systemd/glm53-2x-rtxpro6000.service.in" "$UNIT_DIR/glm53-2x-rtxpro6000.service" "$PORT" "$SERVED_MODEL_NAME" "$MAX_MODEL_LEN" "$CONTAINER_NAME" "$READINESS_HOST" <<'PY'
from pathlib import Path
import sys
root,env,src,dst,port,model,context,container,readiness_host=sys.argv[1:]
replacements={
    '@REPO_DIR@': root,
    '@ENV_FILE@': env,
    '@PORT@': port,
    '@SERVED_MODEL_NAME@': model,
    '@MAX_MODEL_LEN@': context,
    '@CONTAINER_NAME@': container,
    '@READINESS_HOST@': readiness_host,
}
text=Path(src).read_text()
for old,new in replacements.items():
    text=text.replace(old,new)
if '@' in text:
    raise SystemExit('unrendered systemd placeholder')
Path(dst).write_text(text)
PY
systemctl --user daemon-reload
systemctl --user enable glm53-2x-rtxpro6000.service
echo 'Installed and enabled. Start with:'
echo '  systemctl --user start glm53-2x-rtxpro6000.service'
echo 'Follow with:'
echo '  journalctl --user -u glm53-2x-rtxpro6000.service -f'
echo 'The service starts with your user manager. For pre-login boot start, review:'
echo "  sudo loginctl enable-linger \"$USER\""
