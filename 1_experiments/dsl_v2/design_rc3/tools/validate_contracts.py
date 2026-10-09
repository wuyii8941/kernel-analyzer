#!/usr/bin/env python3
"""Structural rule checks only. They are NOT proof or implementation validation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import jsonschema


def validate_record(record: dict, schema: dict) -> None:
    jsonschema.Draft202012Validator(schema).validate(record)
    if record["implementation_status"] == "validated-scope":
        if not record["trigger_evidence"]:
            raise ValueError("validated-scope requires executed signature evidence")
        if not all(record["tests"][kind] for kind in ("positive", "boundary", "premise_violation")):
            raise ValueError("validated-scope requires all three test categories")
        if record["proof_status"] not in {"reviewed-argument", "machine-checked"}:
            raise ValueError("validated-scope requires a reviewed correctness argument")
    for profile in record["device_validation"]:
        if profile["status"] == "device-tested" and not profile["evidence"]:
            raise ValueError("device-tested requires device evidence")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", nargs="+", type=Path)
    p.add_argument("--schema", type=Path,
                   default=Path(__file__).resolve().parents[1]/"design/rule_contract.schema.json")
    a=p.parse_args(); schema=json.loads(a.schema.read_text())
    failed=[]
    for f in a.records:
        try:
            validate_record(json.loads(f.read_text()),schema)
        except (OSError, ValueError, jsonschema.ValidationError) as e:
            failed.append({"file":str(f),"error":str(e)})
    print(json.dumps({"records":len(a.records),"failures":failed,
                      "meaning":"shape/evidence-presence checks, not semantic proof"},ensure_ascii=False,indent=2))
    return 1 if failed else 0

if __name__=="__main__":
    raise SystemExit(main())
