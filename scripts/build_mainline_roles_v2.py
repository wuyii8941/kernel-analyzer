#!/usr/bin/env python3
"""Index retained records without merging different protocols by case name."""
import argparse
import hashlib
import json
from pathlib import Path
from kernel_analyzer.reference_provenance import recorded_reference_scope

ROOT=Path(__file__).resolve().parents[1]


def source(path, expected=None):
    p=ROOT/path
    digest=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return {"path":path,"sha256":digest,"expected_sha256":expected,
            "status":"MISSING" if digest is None else "DIGEST_CHANGED" if expected and expected!=digest else "AVAILABLE"}


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists(): raise SystemExit("Choose a new index path; old ledgers are immutable.")
    ledger_path="results/evidence_v2/case_stage_matrix.json"
    roles_path="results/mainline_case_roles.json"
    ledger=json.loads((ROOT/ledger_path).read_text()); old=json.loads((ROOT/roles_path).read_text())
    records=[]
    for row in ledger["records"]:
        records.append({
            "record_id":row["record_id"],"case_id":row["case_id"],
            "mathematical_operation":row.get("backward_region","UNKNOWN"),
            "forward_endpoint":row.get("forward_endpoint","UNKNOWN"),
            "candidate_backend":"UNKNOWN_IN_THIS_RECORD","reference_backend":"UNKNOWN_IN_THIS_RECORD",
            "error_location":"NOT_INFERRED_FROM_MODEL_OR_OPERATOR_NAME",
            "exact_contrast":row.get("exact_contrast_id","UNKNOWN"),
            "parameter_scope":row.get("parameter_scope","UNKNOWN"),
            "state_source":row.get("state_protocol","UNKNOWN"),
            "optimizer":row.get("optimizer","UNKNOWN"),"moment_state":row.get("moment_state","UNKNOWN"),
            "statistics_version":ledger["schema"],"data_use":"HISTORICAL_PROTOCOL_RECORD",
            "recorded_status":row.get("status"),
            "source_artifacts":[source(s["path"],s.get("sha256")) for s in row.get("source_artifacts",[])],
            "ledger_source":source(ledger_path),
        })
    for row in old["cases"]:
        records.append({
            "record_id":"runtime_role::"+row["case_id"],"case_id":row["case_id"],
            "mathematical_operation":row.get("semantic_family","UNKNOWN"),
            "candidate_backend":row.get("candidate_role","UNKNOWN"),
            "actual_backend_evidence":row.get("actual_backend_evidence",{}),
            "reference_backend":"REQUIRES_PER_CONTRAST_CHECK",
            "error_location":"REQUIRES_SOURCE_LEVEL_CONFIRMATION",
            "parameter_scope":row.get("carrier","UNKNOWN"),
            "state_source":"SEE_BOUND_ARTIFACT_NOT_INFERRED_FROM_ROLE",
            "statistics_version":old.get("schema","UNKNOWN"),
            "data_use":row.get("purpose","UNKNOWN"),
            "source_artifacts":[source(row["artifact"],row.get("artifact_sha256"))],
            "ledger_source":source(roles_path),
        })
    raw_paths = set()
    for name in ('training_numerical_analysis_v2', 'numerical_coverage_v1'):
        current = ROOT / 'results/property' / name
        raw_paths.update(current.rglob('raw.json'))
        raw_paths.update(current.glob('**/raw/*.json'))
    # Shared capture exposes the same raw file through per-case links. Do not
    # count those links as independent measurements.
    for path in sorted({p.resolve() for p in raw_paths}):
        row = json.loads(path.read_text())
        if row.get("status") != "COMPLETE":
            continue
        relative = str(path.relative_to(ROOT))
        records.append({
            "record_id": "actual_write_v2::" + relative,
            "case_id": row["case_id"],
            "mathematical_operation": "SEE_DECLARED_RUNTIME_BOUNDARY",
            "candidate_backend": "SEE_EXECUTION_EVIDENCE_NOT_INFERRED_FROM_MODEL",
            "actual_backend_evidence": row.get("runtime_boundary", {}),
            "reference_backend": "SEE_RECORDED_REFERENCE_SCOPE",
            "reference_scope_audit": recorded_reference_scope(row, path, ROOT),
            "exact_contrast": row.get("contrast_id", "UNKNOWN"),
            "parameter_scope": row.get("carrier", "UNKNOWN"),
            "parameter_representation": row.get("parameter_write_protocol", {}),
            "state_source": row.get("input_bank", "SEE_SOURCE_ARTIFACT"),
            "state_ids": row.get("state_ids", row.get("calibration_state_ids", []) + row.get("confirmation_state_ids", [])),
            "optimizer": row.get("optimizer", "SEE_SOURCE_ARTIFACT"),
            "statistics_version": "training-numerical-analysis-readback-v2",
            "data_use": "SEE_BOUND_PROTOCOL_NEW_OR_HISTORICAL_NOT_INFERRED_FROM_CASE_NAME",
            "claim_boundary": row.get("claim_boundary", "FIXED_SUITE_DECLARED_PARAMETER_ONLY"),
            "source_artifacts": [source(relative)],
            "bias_or_loss_not_inferred_from_capture_completion": True,
        })
    result={"schema":"mainline-record-roles-v2","status":"INDEXED_WITH_EXPLICIT_GAPS",
            "record_count":len(records),"case_count":len({r['case_id'] for r in records}),
            "records":records,
            "rule":"Records with the same case ID remain separate when contrasts or protocols differ. UNKNOWN is not an eligibility pass; this index is not an independent source audit.",
            "source_gap_count":sum(s['status']!='AVAILABLE' for r in records for s in r['source_artifacts'])}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f: json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=="__main__":main()
