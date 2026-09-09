import copy
import json

from scripts.round2_vl_math import _derive_backward_only_partition_replays
from scripts.round2_vl_math import _verify_nonlinear_and_normalization_composite


def fixture():
    origin = [["transpose", "transpose"]]
    source = {"name": "x", "ordinal": 1,
              "tensor_meta": [[2, 3, 4], "torch.float32", False, [], None, False, {}]}
    node = {"name": "y", "ordinal": 2, "target": "aten.permute.default",
            "original_aten": "aten.transpose.int", "partitioner_tag": "is_backward",
            "source_fn_stack": None, "fwd_source_fn_stack": origin, "seq_nr": 42,
            "arguments": {"args": [{"node": "x"}, [0, 2, 1]], "kwargs": {}},
            "tensor_meta": [[2, 4, 3], "torch.float32", False, [], None, False, {}]}
    return source, node, json.dumps(origin)


def evaluate(source, node, origin):
    proofs, _ = _derive_backward_only_partition_replays({origin: [node]}, {"x": source, "y": node})
    return proofs["y"]["passed"]


def test_permutation_rematerialization():
    assert evaluate(*fixture())


def test_permutation_rejects_wrong_axes_metadata_source_and_origin():
    source, node, origin = fixture()
    mutations = [
        lambda n: n["arguments"]["args"].__setitem__(1, [0, 1, 1]),
        lambda n: n["arguments"]["args"].__setitem__(1, [0, 1, 3]),
        lambda n: n["arguments"]["args"].__setitem__(0, {"node": "missing"}),
        lambda n: n["tensor_meta"].__setitem__(0, [2, 3, 4]),
        lambda n: n.__setitem__("seq_nr", None),
        lambda n: n.__setitem__("original_aten", "aten.add.Tensor"),
    ]
    for mutate in mutations:
        altered = copy.deepcopy(node)
        mutate(altered)
        assert not evaluate(source, altered, origin)


def test_three_cycle_is_not_a_transpose():
    source, node, origin = fixture()
    node["arguments"]["args"][1] = [1, 2, 0]
    node["tensor_meta"][0] = [3, 4, 2]
    assert not evaluate(source, node, origin)
    node["original_aten"] = "aten.permute.default"
    assert evaluate(source, node, origin)


def test_reshape_rematerialization_checks_element_count():
    source, node, origin = fixture()
    node.update(target="aten.view.default", original_aten="aten.reshape.default")
    node["arguments"]["args"][1] = [6, 4]
    node["tensor_meta"][0] = [6, 4]
    assert evaluate(source, node, origin)
    node["arguments"]["args"][1] = [7, 4]
    node["tensor_meta"][0] = [7, 4]
    assert not evaluate(source, node, origin)


def test_sum_rematerialization_checks_axes_and_dtype():
    source, node, origin = fixture()
    node.update(target="aten.sum.dim_IntList", original_aten="aten.sum.dim_IntList")
    node["arguments"]["args"] = [{"node": "x"}, [1], True]
    node["tensor_meta"][0] = [2, 1, 4]
    assert evaluate(source, node, origin)
    node["arguments"]["args"][1] = [2]
    assert not evaluate(source, node, origin)
    node["arguments"]["args"][1] = [1]
    node["tensor_meta"][1] = "torch.bfloat16"
    assert not evaluate(source, node, origin)


def test_tanh_exact_chain_and_wrong_sign():
    origin = [["tanh", "tanh"]]
    def node(name, target, args, dtype="torch.float32"):
        return {"name": name, "target": target, "arguments": {"args": args, "kwargs": {}},
                "input_edges": [{"source_node": a["node"]} for a in args if isinstance(a, dict) and "node" in a],
                "tensor_meta": [[2, 3], dtype, False, [], None, False, {}],
                "source_fn_stack": origin, "fwd_source_fn_stack": origin}
    x = node("x", "placeholder", [], "torch.bfloat16")
    f = node("y", "aten.tanh.default", [{"node": "x"}], "torch.bfloat16")
    cast = "prims.convert_element_type.default"
    b = [node("q", cast, [{"node": "incoming"}, "torch.float32"]),
         node("s", cast, [{"node": "y"}, "torch.float32"]),
         node("sq", "aten.mul.Tensor", [{"node": "s"}, {"node": "s"}]),
         node("c", "aten.sub.Tensor", [1, {"node": "sq"}]),
         node("p", "aten.mul.Tensor", [{"node": "q"}, {"node": "c"}]),
         node("dx", cast, [{"node": "p"}, "torch.bfloat16"], "torch.bfloat16")]
    assert _verify_nonlinear_and_normalization_composite([f], b, {"x": x})["passed"]
    b[3]["arguments"]["kwargs"]["alpha"] = -1
    assert not _verify_nonlinear_and_normalization_composite([f], b, {"x": x})["passed"]
