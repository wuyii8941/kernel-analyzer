import gzip
import json

from scripts.preflight_triton_signature_runtimes import MISMATCH, preflight_cases


def test_mismatch_parser_requires_exact_port_match_language():
    text = (
        "ValueError: reference cut AOT graph hash mismatch after exact port match: "
        "same-dtype:backward:graph2:add_4 expected=" + "a" * 64 + " actual=" + "b" * 64
    )
    match = MISMATCH.search(text)
    assert match.groups() == (
        "same-dtype:backward:graph2:add_4", "backward", "2", "a" * 64, "b" * 64
    )
    assert MISMATCH.search(text.replace("after exact port match", "without port proof")) is None


def test_preflight_checks_every_selected_task_not_one_per_graph(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    plan = {
        "rows": [
            {"task_id": "a", "exact_aot_endpoint_id": "backward:graph0:x"},
            {"task_id": "b", "exact_aot_endpoint_id": "backward:graph0:y"},
            {"task_id": "c", "exact_aot_endpoint_id": "backward:graph1:z"},
        ],
        "reference_cut_tasks": [
            {"task_id": "same-dtype:backward:graph0:x", "phase": "BACKWARD", "graph_index": 0},
            {"task_id": "same-dtype:backward:graph0:y", "phase": "BACKWARD", "graph_index": 0},
            {"task_id": "same-dtype:backward:graph1:z", "phase": "BACKWARD", "graph_index": 1},
        ],
    }
    with gzip.open(release / "same_dtype_tasks.json.gz", "wt") as stream:
        json.dump(plan, stream)
    group = {"release": str(release), "cases": [
        {"task_id": "a"}, {"task_id": "b"}, {"task_id": "c"},
    ]}
    assert [row["task_id"] for row in preflight_cases(group)] == ["a", "b", "c"]


def test_preflight_cases_uses_runtime_cut_id_not_stale_graph_field(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    plan = {
        "rows": [
            {"task_id": "a", "exact_aot_endpoint_id": "backward:graph0:x"},
            {"task_id": "b", "exact_aot_endpoint_id": "backward:graph1:y"},
        ],
        "reference_cut_tasks": [
            {"task_id": "same-dtype:backward:graph0:x", "phase": "BACKWARD", "graph_index": 1},
            {"task_id": "same-dtype:backward:graph1:y", "phase": "BACKWARD", "graph_index": 0},
        ],
    }
    with gzip.open(release / "same_dtype_tasks.json.gz", "wt") as stream:
        json.dump(plan, stream)
    group = {"release": str(release), "cases": [{"task_id": "a"}, {"task_id": "b"}]}
    assert [row["task_id"] for row in preflight_cases(group)] == ["a", "b"]


def test_distinct_cuts_with_same_graph_hash_pair_are_not_the_same_refresh():
    first = ("same-dtype:backward:graph0:x", "a" * 64, "b" * 64)
    second = ("same-dtype:backward:graph0:y", "a" * 64, "b" * 64)
    assert first != second
