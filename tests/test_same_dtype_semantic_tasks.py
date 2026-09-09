from scripts.build_same_dtype_semantic_tasks import (
    nearest_downstream_endpoint_tasks,
)


def test_nearest_downstream_endpoint_tasks_keeps_only_first_reachable_level() -> None:
    successors = {
        "source": {"left", "right"},
        "left": {"near"},
        "right": {"middle"},
        "middle": {"far"},
        "near": set(),
        "far": set(),
    }
    endpoints = {"near": ["near:output"], "far": ["far:output"]}

    closures = nearest_downstream_endpoint_tasks(successors, endpoints)

    assert closures["source"] == ["near:output"]
    assert closures["right"] == ["far:output"]


def test_nearest_downstream_endpoint_tasks_unions_tied_endpoints() -> None:
    successors = {
        "source": {"left", "right"},
        "left": {"end_left"},
        "right": {"end_right"},
    }
    endpoints = {
        "end_left": ["left:output"],
        "end_right": ["right:output"],
    }

    assert nearest_downstream_endpoint_tasks(successors, endpoints)["source"] == [
        "left:output",
        "right:output",
    ]


def test_nearest_downstream_endpoint_tasks_requires_a_positive_path() -> None:
    successors = {
        "endpoint": set(),
        "cycle_a": {"cycle_b"},
        "cycle_b": {"cycle_a", "endpoint"},
        "unresolved": set(),
    }
    endpoints = {"endpoint": ["endpoint:output"], "cycle_a": ["cycle_a:other"]}

    closures = nearest_downstream_endpoint_tasks(successors, endpoints)

    assert closures["endpoint"] == []
    assert closures["cycle_b"] == ["cycle_a:other", "endpoint:output"]
    assert closures["unresolved"] == []
