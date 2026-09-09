from scripts.round2_vl_math import _verify_nonlinear_and_normalization_composite


def fixture():
    origin = [["gelu", "gelu"]]
    def n(name, target, args, dtype="torch.float32"):
        return {"name": name, "target": target, "arguments": {"args": args, "kwargs": {}},
                "tensor_meta": [[2, 3], dtype, False, [], None, False, {}],
                "input_edges": [{"source_node": a["node"]} for a in args if isinstance(a, dict) and "node" in a],
                "source_fn_stack": origin, "fwd_source_fn_stack": origin}
    r = lambda name: {"node": name}
    mul, add, cast = "aten.mul.Tensor", "aten.add.Tensor", "prims.convert_element_type.default"
    x = n("input", "placeholder", [], "torch.bfloat16")
    f = [n("x", cast, [r("input"), "torch.float32"]),
         n("x2", mul, [r("x"), r("x")]), n("x3", mul, [r("x2"), r("x")]),
         n("ax3", mul, [r("x3"), .044715]), n("inner", add, [r("x"), r("ax3")]),
         n("scaled", mul, [r("inner"), .7978845608028654]), n("hx", mul, [r("x"), .5]),
         n("t", "aten.tanh.default", [r("scaled")]), n("one", add, [r("t"), 1]),
         n("y", mul, [r("hx"), r("one")]), n("out", cast, [r("y"), "torch.bfloat16"], "torch.bfloat16")]
    b = [n("q", cast, [r("incoming"), "torch.float32"]), n("left", mul, [r("one"), .5]),
         n("t2", mul, [r("t"), r("t")]), n("sech2", "aten.sub.Tensor", [1, r("t2")]),
         n("three", mul, [r("x2"), .134145]), n("one_der", add, [r("three"), 1]),
         n("der", mul, [r("one_der"), .7978845608028654]), n("right_head", mul, [r("hx"), r("sech2")]),
         n("right", mul, [r("right_head"), r("der")]), n("total", add, [r("left"), r("right")]),
         n("dq", mul, [r("q"), r("total")]), n("dx", cast, [r("dq"), "torch.bfloat16"], "torch.bfloat16")]
    return f, b, {"input": x}


def test_nominal_derivative_does_not_claim_binary_constant_identity():
    proof = _verify_nonlinear_and_normalization_composite(*fixture())
    assert proof["passed"]
    assert proof["coefficient_interpretation"] == "NOMINAL_DECIMAL_CONSTANTS_BEFORE_BINARY_REPRESENTATION"
    assert proof["binary_coefficient_relation_error"] != "0"
    assert "NOT claimed" in proof["claim_boundary"]


def test_wrong_derivative_or_saved_value_rejected_without_tolerance():
    for mutation in ("coefficient", "saved", "alpha"):
        f, b, index = fixture()
        if mutation == "coefficient":
            b[4]["arguments"]["args"][1] = .13414500000000004
        elif mutation == "saved":
            b[2]["arguments"]["args"][1] = {"node": "other_tanh"}
        else:
            b[9]["arguments"]["kwargs"]["alpha"] = -1
        assert not _verify_nonlinear_and_normalization_composite(f, b, index)["passed"]
