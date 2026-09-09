import gzip
import json
from pathlib import Path

from scripts.build_uncovered_semantic_review_queue import compact, enrich, release_root


def write_gzip_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as stream:
        json.dump(value, stream)


def test_enrich_attaches_compiler_semantics_without_assigning_family(tmp_path):
    root = tmp_path / "release"
    source = root / "trace" / "model__1" / "output_code.py"
    source.parent.mkdir(parents=True)
    source.write_text("saved source")
    write_gzip_json(root / "candidate_fb_bridge.json.gz", {"rows": [{
        "source_path": "model__1/output_code.py", "symbol": "kernel",
        "candidate_region_id": "backward:7", "status": "BOUND", "method": "EXACT",
        "aot_node_ids": ["backward:graph0:sum_1"],
    }]})
    write_gzip_json(root / "campaign.json.gz", {"rows": [{
        "region_id": "backward:7", "symbol": "kernel",
        "original_aten": ["aten.mul", "aten.sum"], "source_nodes": ["sum_1"],
        "reference_kind": "SAVED_PROGRAM_REPLAY",
    }]})
    write_gzip_json(root / "default_aot_capture.json.gz", {"capture": {"graphs": [{
        "nodes": [{"name": "sum_1", "fwd_nn_module_stack": {
            "1": ["L['self'].subject.norm", "RMSNorm"]}, "stack_trace": "trace"}]
    }]}})
    priority = {
        "scope": "test", "uncovered_definition_count": 1,
        "structural_cluster_count": 1, "clusters": [{
            "representative_symbol": "kernel", "members": [{
                "source": str(source), "symbol": "kernel"}],
        }],
    }
    result = enrich(priority)
    cluster = result["clusters"][0]
    assert cluster["compiler_original_aten"] == ["aten.mul", "aten.sum"]
    assert cluster["compiler_module_paths"] == ["L['self'].subject.norm"]
    assert cluster["semantic_review_status"] == "REQUIRES_HUMAN_FAMILY_REVIEW"
    assert cluster["members"][0]["semantic_binding_status"] == "COMPILER_SEMANTICS_ATTACHED"
    assert result["selection_uses_numerical_results"] is False


def test_compact_keeps_member_identity_and_cluster_semantics(tmp_path):
    root = tmp_path / "release"
    source = root / "trace" / "model__1" / "output_code.py"
    source.parent.mkdir(parents=True)
    source.write_text("saved source")
    write_gzip_json(root / "campaign.json.gz", {"rows": [{
        "region_id": "backward:7", "phase": "BACKWARD", "symbol": "kernel",
        "source_path": "model__1/output_code.py",
        "original_aten": ["aten.mul", "aten.sum"], "source_nodes": [],
    }]})
    result = compact(enrich({
        "scope": "test", "uncovered_definition_count": 1,
        "structural_cluster_count": 1, "clusters": [{
            "representative_symbol": "kernel", "members": [{
                "source": str(source), "symbol": "kernel"}],
        }],
    }))
    cluster = result["clusters"][0]
    assert result["schema"] == "uncovered-triton-semantic-review-queue-compact-v1"
    assert cluster["compiler_original_aten"] == ["aten.mul", "aten.sum"]
    assert cluster["member_binding_status_counts"] == {
        "COMPILER_SEMANTICS_ATTACHED_FROM_CAMPAIGN_SOURCE_PATH": 1,
    }
    assert "compiler_semantics" not in cluster["members"][0]
    assert cluster["members"][0]["source"] == str(source)


def test_release_root_requires_trace_component(tmp_path):
    source = tmp_path / "release" / "trace" / "x.py"
    assert release_root(source) == (tmp_path / "release").resolve()


def test_enrich_accepts_raw_aot_capture_filename(tmp_path):
    root = tmp_path / "release"
    source = root / "trace" / "model__1" / "output_code.py"
    source.parent.mkdir(parents=True)
    source.write_text("saved source")
    write_gzip_json(root / "candidate_fb_bridge.json.gz", {"rows": [{
        "source_path": "model__1/output_code.py", "symbol": "kernel",
        "candidate_region_id": "backward:7", "status": "BOUND", "method": "EXACT",
    }]})
    write_gzip_json(root / "campaign.json.gz", {"rows": [{
        "region_id": "backward:7", "symbol": "kernel",
        "original_aten": ["aten.sum"], "source_nodes": [],
    }]})
    write_gzip_json(root / "default_aot_capture_raw.json.gz", {
        "capture": {"graphs": []}})
    result = enrich({
        "scope": "test", "uncovered_definition_count": 1,
        "structural_cluster_count": 1, "clusters": [{
            "representative_symbol": "kernel", "members": [{
                "source": str(source), "symbol": "kernel"}],
        }],
    })
    assert result["clusters"][0]["compiler_original_aten"] == ["aten.sum"]


def test_enrich_keeps_campaign_semantics_when_aot_file_is_absent(tmp_path):
    root = tmp_path / "release"
    source = root / "trace" / "model__1" / "output_code.py"
    source.parent.mkdir(parents=True)
    source.write_text("saved source")
    write_gzip_json(root / "candidate_fb_bridge.json.gz", {"rows": [{
        "source_path": "model__1/output_code.py", "symbol": "kernel",
        "candidate_region_id": "backward:7", "status": "BOUND", "method": "EXACT",
    }]})
    write_gzip_json(root / "campaign.json.gz", {"rows": [{
        "region_id": "backward:7", "symbol": "kernel",
        "original_aten": ["aten.sum"], "source_nodes": ["sum_1"],
    }]})
    result = enrich({
        "scope": "test", "uncovered_definition_count": 1,
        "structural_cluster_count": 1, "clusters": [{
            "representative_symbol": "kernel", "members": [{
                "source": str(source), "symbol": "kernel"}],
        }],
    })
    assert result["clusters"][0]["compiler_original_aten"] == ["aten.sum"]
    assert result["release_aot_metadata_status"][str(root.resolve())] == "NOT_PRESENT_IN_RELEASE"


def test_enrich_records_missing_campaign_and_bridge_without_guessing(tmp_path):
    root = tmp_path / "release"
    source = root / "trace" / "model__1" / "output_code.py"
    source.parent.mkdir(parents=True)
    source.write_text("saved source")
    result = enrich({
        "scope": "test", "uncovered_definition_count": 1,
        "structural_cluster_count": 1, "clusters": [{
            "representative_symbol": "kernel", "members": [{
                "source": str(source), "symbol": "kernel"}],
        }],
    })
    member = result["clusters"][0]["members"][0]
    assert member["semantic_binding_status"] == "NO_COMPILER_BINDING_FOUND"
    assert member["compiler_semantics"] == []
    assert result["release_semantic_metadata_status"][str(root.resolve())] == "MISSING_REQUIRED_PROVENANCE"


def test_unique_campaign_phase_symbol_is_review_only_fallback(tmp_path):
    root = tmp_path / "release"
    source = root / "trace" / "model__1_backward" / "output_code.py"
    source.parent.mkdir(parents=True)
    source.write_text("saved source")
    write_gzip_json(root / "campaign.json.gz", {"rows": [{
        "region_id": "backward:7", "phase": "BACKWARD", "symbol": "kernel",
        "original_aten": ["aten.sum"], "source_nodes": [],
    }]})
    result = enrich({
        "scope": "test", "uncovered_definition_count": 1,
        "structural_cluster_count": 1, "clusters": [{
            "representative_symbol": "kernel", "members": [{
                "source": str(source), "symbol": "kernel"}],
        }],
    })
    member = result["clusters"][0]["members"][0]
    assert member["semantic_binding_status"] == "REVIEW_SEMANTICS_ATTACHED_WITHOUT_EXACT_SOURCE_BRIDGE"
    assert member["compiler_semantics"][0]["original_aten"] == ["aten.sum"]
    assert member["compiler_semantics"][0]["binding_method"] == "UNIQUE_CAMPAIGN_PHASE_SYMBOL_FOR_REVIEW_ONLY"


def test_enrich_keeps_non_trace_source_explicitly_unbound(tmp_path):
    source = tmp_path / "standard_aot" / "torchinductor" / "output_code.py"
    source.parent.mkdir(parents=True)
    source.write_text("saved source")
    result = enrich({
        "scope": "test", "uncovered_definition_count": 1,
        "structural_cluster_count": 1, "clusters": [{
            "representative_symbol": "kernel", "members": [{
                "source": str(source), "symbol": "kernel"}],
        }],
    })
    member = result["clusters"][0]["members"][0]
    assert member["semantic_binding_status"] == "NOT_A_RUNTIME_RELEASE_TRACE"
    assert member["release_root"] is None
    assert member["trace_source"] is None
    assert member["compiler_semantics"] == []


def test_campaign_source_path_is_used_without_bridge(tmp_path):
    root = tmp_path / "release"
    source = root / "trace" / "model__1_backward" / "output_code.py"
    source.parent.mkdir(parents=True)
    source.write_text("saved source")
    write_gzip_json(root / "campaign.json.gz", {"rows": [{
        "region_id": "backward:7", "phase": "BACKWARD", "symbol": "kernel",
        "source_path": "model__1_backward/output_code.py",
        "original_aten": ["aten.tanh_backward"], "source_nodes": [],
        "status": "EXACT_STATIC_PROGRAM_AND_POINTER_ABI_REPLAY_PLAN",
    }]})
    result = enrich({
        "scope": "test", "uncovered_definition_count": 1,
        "structural_cluster_count": 1, "clusters": [{
            "representative_symbol": "kernel", "members": [{
                "source": str(source), "symbol": "kernel"}],
        }],
    })
    member = result["clusters"][0]["members"][0]
    assert member["semantic_binding_status"] == (
        "COMPILER_SEMANTICS_ATTACHED_FROM_CAMPAIGN_SOURCE_PATH")
    assert member["compiler_semantics"][0]["original_aten"] == [
        "aten.tanh_backward"]
    assert member["compiler_semantics"][0]["binding_method"] == (
        "CAMPAIGN_SOURCE_PATH_AND_SYMBOL")
