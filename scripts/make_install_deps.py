#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


SCRIPT = """#!/usr/bin/env bash
set -euo pipefail

cd /data1/tzh/forkcert
export PIP_CACHE_DIR=/data1/tzh/forkcert/cache/pip
export HF_HOME=/data1/tzh/forkcert/cache/huggingface
export XDG_CACHE_HOME=/data1/tzh/forkcert/cache/xdg
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" logs

PY={python}
"$PY" -m pip install --force-reinstall fsspec numpy
"$PY" -m pip install --upgrade \\
  transformers \\
  accelerate \\
  datasets \\
  trl \\
  sentencepiece \\
  protobuf

PYTHONPATH=/data1/tzh/forkcert/src "$PY" scripts/preflight.py \\
  --config configs/hf_pair.example.yaml \\
  --samples data/prompt_pairs.example.jsonl \\
  --require-ml \\
  --out results/preflight.after_install.json
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dependency installer for the selected ForkCert Python env.")
    parser.add_argument("--python", default="/data1/tzh/conda-envs/forkcert/bin/python")
    parser.add_argument("--out", default="install_deps.sh")
    args = parser.parse_args()
    out = Path(args.out)
    out.write_text(SCRIPT.format(python=args.python), encoding="utf-8")
    out.chmod(0o755)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
