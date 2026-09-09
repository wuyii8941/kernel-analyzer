import copy
from scripts.round2_vl_math import _verify_reused_bmm_composite


def fixture():
    origin = [["matmul", "matmul"]]
    def n(name, target, args, shape):
        return {"name": name, "target": target, "arguments": {"args": args, "kwargs": {}},
                "tensor_meta": [shape, "torch.float32", False, [], None, False, {}],
                "input_edges": [{"source_node": a["node"]} for a in args if isinstance(a, dict) and "node" in a],
                "source_fn_stack": origin, "fwd_source_fn_stack": origin}
    def ref(name):
        return {"node": name}
    x = n("x", "placeholder", [], [1, 2, 3, 4])
    w = n("w", "placeholder", [], [1, 2, 4, 5])
    f = [n("ea", "aten.expand.default", [ref("x"), [1, 2, 3, 4]], [1, 2, 3, 4]),
         n("a", "aten.view.default", [ref("ea"), [2, 3, 4]], [2, 3, 4]),
         n("eb", "aten.expand.default", [ref("w"), [1, 2, 4, 5]], [1, 2, 4, 5]),
         n("b", "aten.view.default", [ref("eb"), [2, 4, 5]], [2, 4, 5]),
         n("prod", "aten.bmm.default", [ref("a"), ref("b")], [2, 3, 5]),
         n("y", "aten.view.default", [ref("prod"), [1, 2, 3, 5]], [1, 2, 3, 5]),
         n("at", "aten.permute.default", [ref("a"), [0, 2, 1]], [2, 4, 3])]
    b = [n("q", "aten.view.default", [ref("incoming"), [2, 3, 5]], [2, 3, 5]),
         n("db", "aten.bmm.default", [ref("at"), ref("q")], [2, 4, 5]),
         n("bt", "aten.permute.default", [ref("b"), [0, 2, 1]], [2, 5, 4]),
         n("da", "aten.bmm.default", [ref("q"), ref("bt")], [2, 3, 4]),
         n("dbv", "aten.view.default", [ref("db"), [1, 2, 4, 5]], [1, 2, 4, 5]),
         n("dav", "aten.view.default", [ref("da"), [1, 2, 3, 4]], [1, 2, 3, 4])]
    return f, b, {n["name"]: n for n in [x, w, *f]}, {n["name"]: n for n in b}


def test_exact_reused_bmm_chain():
    assert _verify_reused_bmm_composite(*fixture())["passed"]


def test_wrong_values_with_same_shapes_do_not_bind():
    f, b, fi, bi = fixture()
    different = copy.deepcopy(fi["b"])
    different["name"] = "other_b"
    fi["other_b"] = different
    b[2]["arguments"]["args"][0] = {"node": "other_b"}
    assert not _verify_reused_bmm_composite(f, b, fi, bi)["passed"]


def test_wrong_cotangent_axis_and_shape_rejected():
    for mutation in ("q", "axes", "shape"):
        f, b, fi, bi = fixture()
        if mutation == "q":
            b[1]["arguments"]["args"][1] = {"node": "another_q"}
        elif mutation == "axes":
            b[2]["arguments"]["args"][1] = [0, 1, 2]
        else:
            b[-1]["tensor_meta"][0] = [1, 2, 4, 3]
        assert not _verify_reused_bmm_composite(f, b, fi, bi)["passed"]
