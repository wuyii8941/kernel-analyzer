"""Review every historical row's direct/consequence evidence without upgrading labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/declared_persistent_4096/all_bias_case_audit.json"
METADATA = ("case_id", "protocol", "claim_boundary", "carrier", "carriers", "case_plan",
            "prediction", "prediction_file", "runtime_release", "release", "runner",
            "source_runner", "state_role", "measurement_geometry", "gradient_scope",
            "only_declared_parameter_updated", "optimizer", "status", "steps",
            "planned_horizon_steps", "consequence_only_early_stop")


def inspect_link(root, link):
    name = link.get("artifact")
    if not name:
        return {"status": "NO_LINK", "historical_status": link.get("status")}
    path = Path(name)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        return {"status": "MISSING", "artifact": name, "historical_status": link.get("status")}
    data = json.loads(path.read_bytes())
    return {"status": "AVAILABLE", "artifact": name,
            "metadata": {k:data[k] for k in METADATA if k in data},
            "top_level_keys": list(data),
            "interpretation": "Long trajectory metadata does not itself prove a numerical source or nonzero mean mechanism"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        parser.error("choose new repository output")
    payload = SOURCE.read_bytes()
    data = json.loads(payload)
    rows = []
    for index, record in enumerate(data["rows"]):
        rows.append({"source_row": index, "case": record["case"],
                     "matrix_case_id": record.get("matrix_case_id"), "scope": record.get("scope"),
                     "historical_label": record.get("final_label"),
                     "historical_formation_description": record.get("formation_path"),
                     "direct": inspect_link(ROOT, record.get("long_direct", {})),
                     "consequence": inspect_link(ROOT, record.get("paired_consequence", {})),
                     "causal_review_status": "SOURCE_AND_INTERVENTION_LINKS_REQUIRE_SEPARATE_REVIEW"})
    result = {"schema": "historical-causal-links-v1", "status": "PARTIAL_CAUSAL_REVIEW",
              "rows": rows,
              "counts": {k:dict(Counter(r[k]["status"] for r in rows)) for k in ("direct", "consequence")},
              "scope": "All 301 historical rows retained including controls, unresolved and early terminations; no causal verdict inferred from names"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
