#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from forkcert.env import audit_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ForkCert runtime environment.")
    parser.add_argument("--out", default="results/env_audit.json")
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    audit = audit_environment().to_json_dict()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
