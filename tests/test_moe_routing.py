import torch

from kernel_analyzer.moe_routing import MoERoutingRecorder, compare_routing


def test_routing_capture_and_discrete_divergence():
    router = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        router.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
    with MoERoutingRecorder({"layer0": router}, top_k=1) as recorder:
        router(torch.tensor([[2.0, 1.0], [1.0, 2.0]]))
    candidate = recorder.certificate()
    repair_rows = [dict(candidate["rows"][0])]
    repair_rows[0]["selected_experts"] = [[1], [1]]
    result = compare_routing(candidate["rows"], repair_rows)
    assert result["routing_regime"] == "DISCRETE_ROUTING_REGIME"
    assert result["flipped_tokens"] == 1
    assert result["removed_from_coverage_denominator"] is False


def test_identical_routing_is_smooth_transport_eligible():
    rows = [{
        "router": "layer0", "invocation": 0,
        "selected_experts": [[0], [1]], "expert_load": [1, 1],
    }]
    result = compare_routing(rows, rows)
    assert result["routing_regime"] == "SAME_ROUTE_SMOOTH_TRANSPORT"
    assert result["hamming_rate"] == 0.0
