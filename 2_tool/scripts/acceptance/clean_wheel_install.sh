#!/usr/bin/env bash
# Clean wheel install and entry-point acceptance (batch 1, packaging): build the wheel from `git archive <commit>`
# (no build artefacts in the working tree), install it into a fresh virtual environment that sees only the
# interpreter's site-packages (torch / triton / numpy / scipy / gmpy2), and from a directory outside the source tree
#   1. import the package and check it comes from the venv,
#   2. run both console entry points (--help),
#   3. run a real mode-A measurement through `kernel-analyzer-measure` on a call file that lives in the run directory,
#   4. rerun it in-process under an audit hook and fail on any open / import of a path under 2_tool/ (sources, scripts,
#      tests): the installed package must not reach back into the repository.
# Usage: clean_wheel_install.sh <commit> <out-dir>      (out-dir: under /data1/tzh/kernel-analyzer/.cache)
set -euo pipefail
COMMIT=${1:-HEAD}
OUT=${2:?out dir}
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
PY=${PY:-/data1/tzh/envs/ka_main/bin/python}
export HOME=$REPO/.cache XDG_CACHE_HOME=$REPO/.cache TRITON_CACHE_DIR=$REPO/.cache/triton TMPDIR=$REPO/.cache/tmp \
       PIP_CACHE_DIR=$REPO/.cache/pip PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
rm -rf "$OUT"
mkdir -p "$OUT"/src "$OUT"/wheel "$OUT"/run
SHA=$(git -C "$REPO" rev-parse "$COMMIT")
echo "commit $SHA" | tee "$OUT"/log.txt
git -C "$REPO" archive "$SHA" | tar -x -C "$OUT"/src
( cd "$OUT"/src && "$PY" -m pip wheel --no-deps --no-build-isolation -q -w "$OUT"/wheel . ) 2>&1 | tee -a "$OUT"/log.txt
WHL=$(ls "$OUT"/wheel/*.whl)
echo "wheel $(basename "$WHL") sha256 $(sha256sum "$WHL" | cut -d' ' -f1)" | tee -a "$OUT"/log.txt
"$PY" -m venv --system-site-packages "$OUT"/venv
"$OUT"/venv/bin/python -m pip install -q --no-deps --no-index "$WHL" 2>&1 | tee -a "$OUT"/log.txt
"$OUT"/venv/bin/python -m pip show -f kernel-analyzer | sed -n '1,12p' | tee -a "$OUT"/log.txt

cd "$OUT"/run
# 1. the package comes from the venv
"$OUT"/venv/bin/python -I -c "import kernel_analyzer, sys; p = kernel_analyzer.__file__; print('package', p); \
assert p.startswith('$OUT/venv/'), p" | tee -a "$OUT"/log.txt
# 2. entry points
"$OUT"/venv/bin/kernel-analyzer-measure --help | head -3 | tee -a "$OUT"/log.txt
"$OUT"/venv/bin/kernel-analyzer-analyze --help | head -3 | tee -a "$OUT"/log.txt
# 3. a real measurement from outside the source tree
cat > calls.py <<'EOF'
import torch
import triton
import triton.language as tl


@triton.jit
def _affine(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) * 3.0 + 1.0, mask=m)


def affine(inp):
    x = inp["x"]
    y = torch.empty_like(x)
    _affine[(triton.cdiv(x.numel(), 128),)](x, y, x.numel(), BLOCK=128)
    return {"y": y}
EOF
cat > decl.json <<'EOF'
{"call": "calls.py:affine", "inputs": {"x": {"sampler": {"uniform": [1, 2]}, "shape": [1024], "dtype": "float32"}},
 "compare": {"mode": "A", "measure": ["y"]},
 "budget": {"cpu_seconds": 600, "gpu_seconds": 600, "case_timeout": 300, "max_units": 18},
 "units": {"development": 2, "confirmation": 16}}
EOF
"$OUT"/venv/bin/kernel-analyzer-measure --declaration decl.json --out report.json | tee -a "$OUT"/log.txt
"$OUT"/venv/bin/python -I - "$OUT"/run/report.json <<'EOF' | tee -a "$OUT"/log.txt
import json, sys
r = json.load(open(sys.argv[1]))
lv = r["levels"][0]
y = lv["outputs"]["y"]
assert lv["status"] == "ok" and y["status"] == "evaluated", lv
assert y["reference"]["complete_rate"] == 1.0 and y["reference"]["reference_scope"] == "call-level", y["reference"]
assert lv["refinement"]["category"] == "met", lv["refinement"]
print("measurement ok:", y["reference"]["reference_scope"], lv["refinement"]["outcome"],
      y["statistics"]["fixed_mean"]["summary"])
EOF
# 4. no access to the repository's tool tree at run time
cat > audit.py <<EOF
import sys
TOOL = "$REPO/2_tool/"
seen = []
def hook(event, args):
    if event in ("open", "import", "os.listdir", "os.scandir") and args and isinstance(args[0], str) \
            and args[0].startswith(TOOL):
        seen.append((event, args[0]))
sys.addaudithook(hook)
from kernel_analyzer.cli import measure_main
rc = measure_main(["--declaration", "decl.json", "--out", "report_audit.json"])
assert rc == 0, rc
print("repository tool paths touched:", seen)
assert not seen, seen
EOF
"$OUT"/venv/bin/python -I audit.py | tail -2 | tee -a "$OUT"/log.txt
echo "CLEAN_WHEEL_ACCEPTANCE_OK" | tee -a "$OUT"/log.txt
