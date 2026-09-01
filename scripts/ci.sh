#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bash -n "$ROOT"/scripts/*.sh
python3 -m py_compile "$ROOT"/scripts/*.py "$ROOT"/runtime/*.py
python3 -m json.tool "$ROOT/examples/zcode-config.fragment.json" >/dev/null
"$ROOT/scripts/test-config-parser.sh"
PYTHONDONTWRITEBYTECODE=1 python3 "$ROOT/scripts/test-template-patch.py"
python3 "$ROOT/scripts/test-repetition-verifier.py"
python3 "$ROOT/scripts/test-canary-control.py"
python3 "$ROOT/scripts/test-capture-profile.py"
python3 "$ROOT/scripts/test-bench-decode.py"
python3 "$ROOT/scripts/test-bench-prefill.py"
python3 "$ROOT/scripts/test-runtime-probes.py"
python3 "$ROOT/scripts/test-compare-benchmarks.py"
python3 "$ROOT/scripts/test-compare-mixed-prefill.py"
python3 "$ROOT/runtime/test-xgrammar-fixes.py"
python3 "$ROOT/runtime/test-sparse-indexer-workspace.py"
python3 "$ROOT/runtime/test-mixed-prefill-policy.py"
python3 "$ROOT/scripts/test-runtime-image-recipe.py"
if command -v shellcheck >/dev/null; then shellcheck "$ROOT"/scripts/*.sh; else echo 'shellcheck not installed; skipped'; fi
python3 - "$ROOT" <<'PY'
from pathlib import Path
import hashlib,re,subprocess,sys
root=Path(sys.argv[1]); required=['README.md','ATTRIBUTIONS.md','THIRD_PARTY_NOTICES.md','PROVENANCE.md','NOTICE','LICENSE','THIRD_PARTY_LICENSES/ShapleyMCG-LICENSE-1.0.txt','THIRD_PARTY_LICENSES/ZAI-GLM-5.3-Flash-BF16-MIT.txt']
for f in required:
    if not (root/f).is_file(): raise SystemExit(f'missing {f}')
expected_hashes={
 'THIRD_PARTY_LICENSES/ShapleyMCG-LICENSE-1.0.txt':'9a354667162e40201fa556e29ae7a327cdb112eacaa8ef100106e6063635e28a',
 'THIRD_PARTY_LICENSES/ZAI-GLM-5.3-Flash-BF16-MIT.txt':'30b85b6b9659f2e78aa259f8faf5d920a68dee7c9ced3fa6dba1f19f2bc4fca1',
}
for f,want in expected_hashes.items():
 got=hashlib.sha256((root/f).read_bytes()).hexdigest()
 if got!=want: raise SystemExit(f'license hash mismatch for {f}: {got}')
if (root/'.git').exists():
 staged=subprocess.check_output(['git','-C',str(root),'ls-files','--stage','scripts'],text=True)
 for line in staged.splitlines():
  mode,_,_,path=line.split(maxsplit=3)
  if path.endswith(('.sh','.py')) and mode!='100755': raise SystemExit(f'script is not executable in Git: {path} ({mode})')
notice='This work includes or was produced using ShapleyMcg, created by Brandon M. Music'
for f in ['README.md','ATTRIBUTIONS.md']:
    if notice not in (root/f).read_text(): raise SystemExit(f'missing required attribution in {f}')
patterns=[re.compile(r'gh[pousr]_[A-Za-z0-9_]{20,}'),re.compile(r'AKIA[0-9A-Z]{16}'),re.compile(r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----')]
if (root/'.git').exists():
    scan_paths=[root/p for p in subprocess.check_output(['git','-C',str(root),'ls-files'],text=True).splitlines()]
else:
    scan_paths=[p for p in root.rglob('*') if p.is_file()]
for p in scan_paths:
    if p.is_file():
        text=p.read_text(errors='ignore')
        for pat in patterns:
            if pat.search(text): raise SystemExit(f'possible secret in {p.relative_to(root)}: {pat.pattern}')
print('repository checks passed')
PY
