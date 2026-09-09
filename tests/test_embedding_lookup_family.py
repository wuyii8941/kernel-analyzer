import ast
from types import SimpleNamespace

import pytest
import torch

from kernel_analyzer.embedding_lookup_reference import select_output, snapshot_pre_call
from kernel_analyzer.embedding_lookup_source import BODY, check_source
from kernel_analyzer.embedding_lookup_observer import observer_class
from kernel_analyzer.source_reference_registry import get_reference
from scripts.bind_embedding_lookup_forward import bind
from scripts.run_embedding_lookup_capture import select, unique_argument


def generated_source(body=None):
    body = body or BODY.format(elements=12, width=3, vocab=7)
    literal = repr(
        "@triton_heuristics.pointwise(\n"
        "    size_hints=[16], filename=__file__, triton_meta={'signature': "
        "{'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}})\n"
        "@triton.jit\n"
        "def triton_poi_fused_embedding_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):\n"
        + "\n".join("    " + line for line in body.strip().splitlines()) + "\n"
    )
    return "triton_poi_fused_embedding_4 = async_compile.triton('triton_', " + literal + ", device_str='cuda')\n"


def test_source_contract_accepts_only_complete_embedding_body():
    result = check_source(generated_source(), "triton_poi_fused_embedding_4")
    assert (result["tokens"], result["width"], result["vocabulary_size"]) == (4, 3, 7)
    tree = ast.parse(BODY.format(elements=12, width=3, vocab=7))
    tree.body[-1].value.args[1] = ast.Name(id="tmp0")
    bad = ast.unparse(tree)
    with pytest.raises(ValueError, match="arithmetic"):
        check_source(generated_source(bad), "triton_poi_fused_embedding_4")


def test_reference_uses_same_indices_and_weight():
    indices = torch.tensor([2, 0, 6, 2], dtype=torch.int64)
    weight = torch.arange(21, dtype=torch.float32).to(torch.bfloat16).reshape(7, 3)
    output = torch.empty((1, 4, 3), dtype=torch.bfloat16)
    saved = snapshot_pre_call(
        {"in_ptr0": indices, "in_ptr1": weight, "out_ptr0": output},
        tokens=4, width=3, vocabulary_size=7,
    )
    result = select_output(saved, output, tokens=4, width=3, vocabulary_size=7)
    assert torch.equal(result, weight.index_select(0, indices).reshape_as(output))
    indices[0] = 1
    assert result[0, 0, 0] != weight[1, 0]


def test_binding_and_capture_selection_reuse_generic_path_algorithm():
    contract = {
        "symbol": "triton_poi_fused_embedding_4", "output_pointers": ["out_ptr0"],
    }
    tasks = [{
        "task_id": "forward:4:out_ptr0", "symbol": "triton_poi_fused_embedding_4",
        "formal_pointer": "out_ptr0", "exact_aot_endpoint_id": "forward:graph0:embedding",
        "status": "EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT",
        "implementation_kind": "TRITON",
    }]
    mapping = {"rows": [{
        "endpoint": "forward:graph0:embedding",
        "parameters": [{"name": "model.embed_tokens.weight", "aot_distance": 1}],
    }]}
    plan = bind(tasks, {contract["symbol"]: contract}, mapping)
    assert plan["schema"] == "embedding-lookup-forward-bound-plan-v1"
    assert plan["cases"][0]["carrier"] == "model.embed_tokens.weight"
    assert plan["cases"][0]["reference_method"] == "EMBEDDING_LOOKUP_COMMON_INPUT"
    assert select(plan, plan["cases"])[contract["symbol"]] == contract


def test_embedding_runner_requires_unique_explicit_state_count():
    assert unique_argument(["--states", "3"], "--states") == "3"
    assert unique_argument([], "--warmup-steps", "0") == "0"
    with pytest.raises(ValueError, match="Unique"):
        unique_argument(["--states", "1", "--states", "2"], "--states")


def test_embedding_is_available_to_the_shared_reference_registry():
    spec = get_reference("EMBEDDING_LOOKUP")
    contract = check_source(generated_source(), "triton_poi_fused_embedding_4")
    indices = torch.tensor([2, 0, 6, 2], dtype=torch.int64)
    weight = torch.arange(21, dtype=torch.float32).to(torch.bfloat16).reshape(7, 3)
    candidate = torch.empty((4, 3), dtype=torch.bfloat16)
    metadata = {"runtime_pointers": {"in_ptr0": indices, "in_ptr1": weight}}
    assert torch.equal(
        spec.evaluate(metadata, candidate, contract, variant="FP32_NATIVE"),
        weight.index_select(0, indices),
    )


def test_runtime_observer_accepts_identical_inductor_module_aliases(tmp_path):
    source = generated_source()
    paths = [tmp_path / "wrapper.py", tmp_path / "cache.py"]
    for path in paths:
        path.write_text(source)
    kernel = SimpleNamespace(run=lambda *args, **kwargs: None)
    modules = [
        SimpleNamespace(__file__=str(path), triton_poi_fused_embedding_4=kernel)
        for path in paths
    ]

    class Base:
        def __init__(self, **kwargs):
            self.modules = kwargs["modules"]

    contract = check_source(source, "triton_poi_fused_embedding_4")
    checked = observer_class(Base, {contract["symbol"]: contract}, lambda unused: [])
    instance = checked(modules=modules)
    assert len(instance.embedding_runtime_sources[contract["symbol"]]) == 2


def test_runtime_observer_rejects_a_different_alias_implementation(tmp_path):
    good = generated_source()
    bad = generated_source(BODY.format(elements=15, width=3, vocab=7))
    paths = [tmp_path / "wrapper.py", tmp_path / "cache.py"]
    paths[0].write_text(good)
    paths[1].write_text(bad)
    modules = [
        SimpleNamespace(
            __file__=str(path),
            triton_poi_fused_embedding_4=SimpleNamespace(run=lambda *args, **kwargs: None),
        )
        for path in paths
    ]

    class Base:
        def __init__(self, **kwargs):
            self.modules = kwargs["modules"]

    contract = check_source(good, "triton_poi_fused_embedding_4")
    checked = observer_class(Base, {contract["symbol"]: contract}, lambda unused: [])
    with pytest.raises(ValueError, match="contract differs"):
        checked(modules=modules)
