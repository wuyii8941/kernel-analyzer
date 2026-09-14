"""Inventory every saved measured position and historical candidate for causal review."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "catalog": "results/property/numerical_coverage_v1/observed_kernel_catalog_with_signature_measurements_v3.json.gz",
    "historical": "results/property/declared_persistent_4096/all_bias_case_audit.json",
    "roles": "results/mainline_case_roles.json",
    "reaudit": "results/coverage/existing_case_reaudit.json",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        parser.error("choose a new repository output")
    documents = {}
    for key,name in SOURCES.items():
        path = ROOT / name
        raw = path.read_bytes()
        documents[key] = json.loads(gzip.decompress(raw) if name.endswith(".gz") else raw)
    rows=[]
    seen=set()
    for index,row in enumerate(documents["catalog"]["positions"]):
        if row.get("support_status") != "VALID_MEASUREMENT_COMPLETED":
            continue
        identity=(row.get("canonical_release", row.get("release")),row.get("task_id"))
        if identity in seen:
            continue
        seen.add(identity)
        rows.append({"audit_id":f"catalog:{identity[0]}:{identity[1]}",
                     "source":SOURCES["catalog"],"source_row_index":index,"record":row})
    if len(seen) != 551:
        raise ValueError(f"expected current 551 measured positions, got {len(seen)}; reconcile catalog scope")
    for key,field in (("historical","rows"),("roles","cases"),("reaudit","rows")):
        for index,row in enumerate(documents[key][field]):
            rows.append({"audit_id":f"{key}:{index}","source":SOURCES[key],"source_row_index":index,"record":row})
    for row in rows:
        row["causal_review"]={
            "code_location":"PENDING_SOURCE_REVIEW",
            "numerical_error":"PENDING_CONTRAST_REVIEW",
            "nonzero_bias_reason":"PENDING_MATHEMATICAL_AND_DISTRIBUTIONAL_REVIEW",
            "intervention":"PENDING_PROTOCOL_MATCH_REVIEW",
            "status":"NOT_YET_CAUSALLY_REVIEWED",
        }
    result={"schema":"all-case-causal-audit-inventory-v1","status":"INVENTORY_ONLY_REVIEW_INCOMPLETE",
            "scope":"Canonical measured catalog positions plus every historical/role/reaudit record; overlapping sources retained, not independent problems.",
            "counts":dict(Counter(r["source"] for r in rows)),
            "record_count":len(rows),"rows":rows,
            "additional_scope_reconciliation_required":["New family evidence not represented in catalog", "Post-catalog AdamW8bit and Liger intervention variants"],
            "inventoried_is_not_reviewed":True}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write("\n")
    print(json.dumps({"counts":result["counts"],"record_count":len(rows)}))


if __name__ == "__main__":
    main()
