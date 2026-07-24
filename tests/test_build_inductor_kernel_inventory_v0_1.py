from build_inductor_kernel_inventory_v0_1 import kernel_kind, select_graph_manifest


def test_kernel_kind_does_not_conflate_external_and_triton():
    assert kernel_kind("triton_red_fused_sum_0") == "triton"
    assert kernel_kind("extern_kernels.mm") == "external"


def test_manifest_alignment_uses_node_evidence_not_compile_order():
    manifests = [
        {"graph_code_sha256": "wrong", "nodes": [{"name": "x"}]},
        {
            "graph_code_sha256": "right",
            "nodes": [{"name": "a"}, {"name": "b"}],
        },
    ]
    selected, overlap = select_graph_manifest({"a", "b"}, manifests)
    assert selected["graph_code_sha256"] == "right"
    assert overlap == 2
