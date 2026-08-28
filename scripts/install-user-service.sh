#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$ROOT/.env" ]] || { echo 'Create and review .env first.' >&2; exit 2; }
UNIT_DIR="$HOME/.config/systemd/user"; ENV_DIR="$HOME/.config/glm53-2x-rtxpro6000"
for path in "$ROOT" "$UNIT_DIR" "$ENV_DIR"; do
  [[ "$path" =~ ^/[A-Za-z0-9._/-]+$ ]] || {
    echo "Unsupported path for systemd rendering (spaces, %, and specifiers are rejected): $path" >&2
    exit 2
  }
done
mkdir -p "$UNIT_DIR" "$ENV_DIR"; install -m 600 "$ROOT/.env" "$ENV_DIR/env"
python3 - "$ROOT" "$ENV_DIR/env" "$ROOT/systemd/glm53-2x-rtxpro6000.service.in" "$UNIT_DIR/glm53-2x-rtxpro6000.service" <<'PY'
from pathlib import Path
import sys
root,env,src,dst=sys.argv[1:]
Path(dst).write_text(Path(src).read_text().replace('@REPO_DIR@',root).replace('@ENV_FILE@',env))
PY
systemctl --user daemon-reload
systemctl --user enable glm53-2x-rtxpro6000.service
echo 'Installed and enabled. Start with:'
echo '  systemctl --user start glm53-2x-rtxpro6000.service'
echo 'Follow with:'
echo '  journalctl --user -u glm53-2x-rtxpro6000.service -f'
echo 'The service starts with your user manager. For pre-login boot start, review:'
echo "  sudo loginctl enable-linger \"$USER\""
