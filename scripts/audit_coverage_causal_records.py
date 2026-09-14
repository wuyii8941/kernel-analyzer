"""Trace saved coverage measurements without inferring root causes from family names."""
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/numerical_coverage_v1/coverage_with_grouped_softmax_embedding_v2.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family-audit", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        parser.error("choose a new repository output")
    source = args.family_audit.resolve() if args.family_audit else SOURCE
    raw = source.read_bytes()
    document = json.loads(raw)
    if args.family_audit:
        document = {"records": [
            {"audit_family": family["name"], "task_id": r["task_id"],
             "measurement_evidence": r}
            for family in document["families"] for r in family["records"]
        ]}
    rows = []
    seen = set()
    for index, record in enumerate(document["records"]):
        evidence = record.get("measurement_evidence") or {}
        name = evidence.get("raw_artifact")
        if not name:
            continue
        identity = (record.get("release", record.get("audit_family")), record.get("task_id"), name)
        if identity in seen:
            continue
        seen.add(identity)
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        row = {"source_row": index, "release": record.get("release"),
               "audit_family": record.get("audit_family"),
               "task_id": record.get("task_id"), "raw_artifact": name,
               "symbol": record.get("symbol"), "implementation_kind": record.get("implementation_kind"),
               "reference_candidates": record.get("reference_candidates"),
               "causal_assessment": "MEASUREMENT_LINK_REVIEW_ONLY_ROOT_NOT_ESTABLISHED"}
        if not path.is_file():
            row["artifact_status"] = "MISSING"
            rows.append(row)
            continue
        row["artifact_status"] = "AVAILABLE"
        data = json.loads(path.read_bytes())
        row["code_location"] = data.get("runtime_boundary")
        row["numerical_comparison"] = data.get("reference_comparison_scope")
        row["contrast_id"] = data.get("contrast_id")
        row["parameter_write"] = data.get("parameter_write_protocol")
        row["carrier"] = data.get("carrier")
        row["state_count"] = len(data.get("state_ids", []))
        row["original_coordinate_stages"] = list(data.get("original_coordinate_statistics", {}))
        row["observation_sham"] = data.get("determinism")
        row["why_bias"] = "NOT_PROVED_BY_ENERGY_OR_REFERENCE_SUBSTITUTION; case-specific mechanism evidence still requires linking"
        row["intervention"] = "DECLARED_REFERENCE_CONTRAST_ONLY; no isolated numerical-source intervention inferred"
        row["scope"] = data.get("claim_boundary")
        rows.append(row)
    result = {"schema": "coverage-causal-links-v1", "status": "PARTIAL_CAUSAL_AUDIT",
              "source": str(source.relative_to(ROOT)),
              "counts": dict(Counter(r["artifact_status"] for r in rows)), "rows": rows,
              "limitations": ["This coverage snapshot predates later signature measurements",
                              "Saved metadata does not independently reproduce historical runtime execution",
                              "Missing linked mechanism evidence is not proof none exists elsewhere"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"rows": len(rows), "counts": result["counts"]}))


if __name__ == "__main__":
    main()
