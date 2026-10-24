import importlib.util
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "sufficiency" / "analyze_compiled.py"
SPEC = importlib.util.spec_from_file_location("analyze_compiled_for_test", PATH)
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)


def complete_record(**extra):
    return {"status": "full reference comparison", "reference_status": "complete", **extra}


def test_complete_development_reference_is_not_automatically_admitted():
    record = complete_record()
    assert ANALYZE.graph_admission(record) == (False, "per-call GRAPH admission not recorded")
    category, reason = ANALYZE.classify({"candidate_status": "matched", "status": "fragment matched, operations covered"},
                                        "verified", record)
    assert category == ANALYZE.EXTEND
    assert "complete development reference" in reason
    assert "only the composition contract" not in reason


def test_partial_or_incomplete_admission_fields_are_not_counted():
    base = {"status": "admitted", "scope": "complete_graph", "implementation_status": "validated",
            "implementation_version": "graph-example-version",
            "checks": {"node_semantics": "passed", "input_domain": "passed", "all_outputs": "passed",
                       "call_binding": "passed"}}
    for change in ({"scope": "output_slice"}, {"implementation_status": "pending"},
                   {"implementation_version": ""}, {"checks": {"node_semantics": "passed"}}):
        admission = dict(base, **change)
        admitted, _ = ANALYZE.graph_admission(complete_record(graph_admission=admission))
        assert not admitted


def test_only_complete_per_call_evidence_is_counted():
    admission = {"status": "admitted", "scope": "complete_graph", "implementation_status": "validated",
                 "implementation_version": "graph-example-version",
                 "checks": {"node_semantics": "passed", "input_domain": "passed", "all_outputs": "passed",
                            "call_binding": "passed"}}
    record = complete_record(graph_admission=admission)
    assert ANALYZE.graph_admission(record)[0]
    category, _ = ANALYZE.classify({"candidate_status": "matched", "status": "fragment matched, operations covered"},
                                   "verified", record)
    assert category == ANALYZE.USABLE
