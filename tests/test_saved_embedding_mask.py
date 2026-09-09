from scripts.round2_vl_math import _verify_index_embedding_conv_composite


def fixture():
    origin = [["embedding", "embedding"]]
    def n(name, target, args, shape, dtype="torch.float32"):
        return {"name": name, "target": target, "arguments": {"args": args, "kwargs": {}},
                "tensor_meta": [shape, dtype, False, [], None, False, {}],
                "source_fn_stack": origin, "fwd_source_fn_stack": origin}
    r = lambda name: {"node": name}
    weight = n("weight", "placeholder", [], [3, 2], "torch.bfloat16")
    indices = n("indices", "placeholder", [], [1], "torch.int64")
    f = n("embedding", "aten.embedding.default", [r("weight"), r("indices"), 0], [1, 2], "torch.bfloat16")
    mask = n("mask", "aten.eq.Scalar", [r("indices"), 0], [1], "torch.bool")
    col = n("column", "aten.unsqueeze.default", [r("mask"), -1], [1, 1], "torch.bool")
    zero = n("zero", "aten.full.default", [[], 0], [])
    cast = "prims.convert_element_type.default"
    b = [n("q", cast, [r("incoming"), "torch.float32"], [1, 2]),
         n("masked", "aten.where.self", [r("column"), r("zero"), r("q")], [1, 2]),
         n("base", "aten.full.default", [[3, 2], 0], [3, 2]),
         n("scatter", "aten.index_put.default", [r("base"), [r("indices")], r("masked"), True], [3, 2]),
         n("dw", cast, [r("scatter"), "torch.bfloat16"], [3, 2], "torch.bfloat16")]
    fi = {x["name"]: x for x in [weight, indices, f, mask, col, zero]}
    bi = {x["name"]: x for x in b}
    return [f], b, fi, bi


def test_saved_padding_mask_is_verified():
    assert _verify_index_embedding_conv_composite(*fixture())["passed"]


def test_wrong_padding_and_nonaccumulating_scatter_rejected():
    for mutation in ("padding", "accumulate", "mask"):
        f, b, fi, bi = fixture()
        if mutation == "padding":
            fi["mask"]["arguments"]["args"][1] = 2
        elif mutation == "accumulate":
            b[3]["arguments"]["args"][3] = False
        else:
            fi["mask"]["target"] = "aten.ne.Scalar"
        assert not _verify_index_embedding_conv_composite(f, b, fi, bi)["passed"]
