#!/usr/bin/env python3
"""W0 (rc3 04): the registered inventory from an enumeration of the build's registry (registry_dump.cpp, built with
triton-opt's own commands by build_registry_dump.sh), in the reconcile tool's format.

    python scripts/dsl_v2/w0_registry_enum.py --ops OPS.tsv --dialects DIALECTS.txt --binary registry_dump \
        --source SOURCE.json --out OUT.json

The registry gives operation names only: each record's contract_hash is copied from the source record of the same
name (stated in the record), so the reconcile tool's source/registered contract comparison is vacuous here; the name
sets are independent (TableGen records vs. the loaded registry).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", type=Path, required=True)
    ap.add_argument("--dialects", type=Path, required=True)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite {a.out}")
    src = json.loads(a.source.read_text())
    hashes = {r["id"]: r["contract_hash"] for r in src["records"]}
    names = sorted({line.split("\t")[1].strip() for line in a.ops.read_text().splitlines() if "\t" in line})
    records = [{"id": n, "contract_hash": hashes.get(n, "not-in-source"),
                "contract_hash_origin": "copied from the source record (the registry enumerates names only)"
                if n in hashes else "no source record"} for n in names]
    doc = {"kind": "registered", "profile_id": src["profile_id"], "source_commit": src["source_commit"],
           "producer": "scripts/dsl_v2/registry_dump.cpp (MLIRContext::getRegisteredOperations after "
                       "registerTritonDialects and loadAllAvailableDialects), built with the compile / link commands "
                       "of bin/triton-opt (scripts/dsl_v2/build_registry_dump.sh)",
           "receipt": {"registry_dump_sha256": hashlib.sha256(a.binary.read_bytes()).hexdigest(),
                       "ops_tsv_sha256": hashlib.sha256(a.ops.read_bytes()).hexdigest(),
                       "dialects": sorted(line.split("\t")[1].strip() for line in a.dialects.read_text().splitlines()
                                          if line.startswith("dialect\t")),
                       "registered_operations": len(names)},
           "complete_for_profile": True,
           "completeness_note": "every operation registered in an MLIRContext built from triton-opt's dialect registry "
                                "with all available dialects loaded; contract hashes are the source's (names only)",
           "records": records}
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(len(names), "registered;", sum(r["contract_hash"] == "not-in-source" for r in records), "without source")


if __name__ == "__main__":
    main()
