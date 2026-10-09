#!/usr/bin/env python3
"""Verify listed package files without changing any file."""
from pathlib import Path
import hashlib
import json
import sys

root=Path(__file__).resolve().parent
failures=[]
count=0
for line in (root/'SHA256SUMS').read_text().splitlines():
    if not line.strip():
        continue
    expected,relative=line.split(None,1)
    relative=relative.strip()
    path=(root/relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        failures.append({'path':relative,'error':'outside package'})
        continue
    count+=1
    if not path.is_file():
        failures.append({'path':relative,'error':'missing'})
        continue
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:
        failures.append({'path':relative,'error':'hash mismatch','expected':expected,'actual':actual})
print(json.dumps({'files_checked':count,'integrity_ok':not failures,'failures':failures},indent=2))
raise SystemExit(1 if failures else 0)
