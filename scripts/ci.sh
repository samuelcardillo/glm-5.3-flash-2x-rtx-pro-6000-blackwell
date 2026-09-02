#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bash -n "$ROOT"/scripts/*.sh
python3 -m py_compile "$ROOT"/scripts/*.py
python3 -m json.tool "$ROOT/examples/zcode-config.fragment.json" >/dev/null
"$ROOT/scripts/test-config-parser.sh"
"$ROOT/scripts/test-v06-profile.sh"
"$ROOT/scripts/test-resolve-model-mount.sh"
python3 "$ROOT/scripts/test-repetition-verifier.py"
python3 "$ROOT/scripts/test-canary-control.py"
python3 "$ROOT/scripts/test-thinking-template.py"
python3 "$ROOT/scripts/test-dflash-benchmark.py"
python3 "$ROOT/scripts/test-capture-profile.py"
python3 "$ROOT/scripts/test-bench-decode.py"
python3 "$ROOT/scripts/test-bench-prefill.py"
python3 "$ROOT/scripts/test-runtime-probes.py"
python3 "$ROOT/scripts/test-compare-benchmarks.py"
if command -v shellcheck >/dev/null; then shellcheck "$ROOT"/scripts/*.sh; else echo 'shellcheck not installed; skipped'; fi
python3 - "$ROOT" <<'PY'
from pathlib import Path
import hashlib, re, subprocess, sys
root = Path(sys.argv[1])
required = [
    'README.md', 'ATTRIBUTIONS.md', 'THIRD_PARTY_NOTICES.md', 'PROVENANCE.md',
    'NOTICE', 'LICENSE', 'THIRD_PARTY_LICENSES/ZAI-GLM-5.3-Flash-BF16-MIT.txt',
    'THIRD_PARTY_LICENSES/CC-BY-NC-ND-4.0.txt',
]
for name in required:
    if not (root / name).is_file():
        raise SystemExit(f'missing {name}')
license_path = root / 'THIRD_PARTY_LICENSES/ZAI-GLM-5.3-Flash-BF16-MIT.txt'
if hashlib.sha256(license_path.read_bytes()).hexdigest() != '30b85b6b9659f2e78aa259f8faf5d920a68dee7c9ced3fa6dba1f19f2bc4fca1':
    raise SystemExit('Z.ai license hash mismatch')
cc_path = root / 'THIRD_PARTY_LICENSES/CC-BY-NC-ND-4.0.txt'
if hashlib.sha256(cc_path.read_bytes()).hexdigest() != 'cb6303892198afb24723a78e59be37222ffd7494690fca00d9307347df50e0b6':
    raise SystemExit('CC BY-NC-ND 4.0 license hash mismatch')
git_tree = (root / '.git').exists()
if git_tree:
    staged = subprocess.check_output(['git', '-C', str(root), 'ls-files', '--stage', 'scripts'], text=True)
    for line in staged.splitlines():
        mode, _, _, path = line.split(maxsplit=3)
        if path.endswith(('.sh', '.py')) and mode != '100755':
            raise SystemExit(f'script is not executable in Git: {path} ({mode})')
patterns = [
    re.compile(r'gh[pousr]_[A-Za-z0-9_]{20,}'),
    re.compile(r'AKIA[0-9A-Z]{16}'),
    re.compile(r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'/(?:home|media)/[A-Za-z0-9._-]+/'),
    re.compile(r'192\.168\.[0-9]{1,3}\.[0-9]{1,3}'),
    re.compile(r'GPU-[0-9a-fA-F-]{16,}'),
    re.compile(r'00000000:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]'),
]
if git_tree:
    scan_paths = [root / p for p in subprocess.check_output(['git', '-C', str(root), 'ls-files'], text=True).splitlines()]
else:
    scan_paths = [p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
for path in scan_paths:
    if path.is_file():
        text = path.read_text(errors='ignore')
        for pattern in patterns:
            for match in pattern.finditer(text):
                relative = path.relative_to(root).as_posix()
                allowed_fixture = '/home/' + 'alice/'
                if relative == 'scripts/test-capture-profile.py' and match.group(0) == allowed_fixture:
                    continue
                raise SystemExit(f'possible secret in {path.relative_to(root)}: {pattern.pattern}')
print('repository checks passed')
PY
