import gzip
import hashlib
import json
from pathlib import Path
import sys

from scripts.merge_same_dtype_semantic_oracle import digest, main


def write_gzip(path: Path, value: dict) -> None:
    with gzip.open(path, "wt") as handle:
        json.dump(value, handle)


def test_same_dtype_state_shards_merge_exactly_once(tmp_path: Path, monkeypatch) -> None:
    bank = {"states": [{"state_id": str(index)} for index in range(32)]}
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(bank))
    bank_sha = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    plan = {
        "result_sha256": "plan",
        "denominator": {
            "stored_candidate_ports": 1,
            "candidate_compute_regions": 1,
            "internal_ports_closed_by_semantic_endpoint": 0,
            "compiler_added_ports_closed_by_exact_theorem": 0,
            "candidate_regions_without_observed_output_port": 0,
        },
        "rows": [{
            "task_id": "task", "candidate_region_id": "forward:0",
            "exact_aot_endpoint_id": "forward:graph0:add",
            "exact_semantic_endpoint_id": "forward:graph0:add",
        }],
    }
    plan_path = tmp_path / "plan.json.gz"
    write_gzip(plan_path, plan)
    metric = {
        "exact": False, "nonzero_elements": 1, "signed_mean": 1.0,
        "rms": 1.0, "max_abs": 1.0, "candidate_finite": True,
        "reference_finite": True, "nonfinite_mismatch": 0,
        "directional_error_sketch": {
            "flat_coordinate_indices": [0], "signed_delta_values": [1.0],
        },
    }
    paths = []
    for shard_index in range(2):
        assigned = [index for index in range(32) if index % 2 == shard_index]
        states = {
            str(index): {"repeats": [
                {"endpoint_metrics": {"task": {"error": metric}}},
                {"endpoint_metrics": {"task": {"error": metric}}},
            ]}
            for index in assigned
        }
        shard = {
            "status": "COMPLETE_SAME_DTYPE_OPTIMIZATION_ORACLE_SHARD",
            "architecture": "synthetic", "sequence_length": 1,
            "input_bank_sha256": bank_sha, "campaign_result_sha256": "campaign",
            "inventory_result_sha256": "inventory",
            "task_plan_result_sha256": "plan",
            "environment_result_sha256": "environment",
            "shard": {"index": shard_index, "count": 2,
                      "assigned_state_indices": assigned},
            "reference_structure": {"reference_cut_runtime": {"gates": {"valid": True}}},
            "states": states,
        }
        shard["result_sha256"] = digest(shard)
        path = tmp_path / f"shard{shard_index}.json.gz"
        write_gzip(path, shard)
        paths.append(path)
    output = tmp_path / "merged.json.gz"
    monkeypatch.setattr(sys, "argv", [
        "merge", "--inputs", *(str(path) for path in paths),
        "--input-bank", str(bank_path), "--task-plan", str(plan_path),
        "--output", str(output), "--bootstrap-draws", "20",
    ])
    main()
    with gzip.open(output, "rt") as handle:
        merged = json.load(handle)
    assert merged["status"] == "COMPLETE_SAME_DTYPE_OPTIMIZATION_ORACLE"
    assert merged["denominator"]["states"] == 32
    assert merged["verdict_counts"] == {"DIRECTIONAL_OPTIMIZATION_BIAS": 1}
    assert merged["gates"]["state_shards_disjoint_and_complete"]
