from kernel_analyzer.operator_taxonomy import classify_observed_position


def classify(**kwargs):
    defaults = dict(reference_families=[], exact_semantic_endpoint_id=None,
                    exact_aot_endpoint_id=None, symbol="unknown_kernel")
    defaults.update(kwargs)
    return classify_observed_position(**defaults)


def test_audited_reference_precedes_structural_symbol():
    result = classify(reference_families=["RMS_BACKWARD"], symbol="triton_red_fused_sum_3")
    assert result["family"] == "NORMALIZATION"
    assert result["classification_confidence"] == "AUDITED"


def test_exact_endpoint_separates_outputs_of_one_fused_kernel():
    symbol = "triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0"
    embedding = classify(exact_semantic_endpoint_id="forward:graph0:embedding", symbol=symbol)
    normalization = classify(exact_semantic_endpoint_id="forward:graph0:rsqrt", symbol=symbol)
    assert embedding["family"] == "EMBEDDING"
    assert normalization["family"] == "NORMALIZATION"


def test_direct_external_matmul_is_linear():
    result = classify(exact_semantic_endpoint_id="backward:graph0:mm_default", symbol="mm")
    assert result["family"] == "LINEAR"
    assert result["classification_confidence"] == "EXACT_ENDPOINT"


def test_unresolved_is_explicit_not_silently_dropped():
    result = classify(symbol="opaque_generated_kernel_9")
    assert result["family"] == "UNRESOLVED_LOW_LEVEL"
    assert result["classification_confidence"] == "UNRESOLVED"


def test_fused_multiple_families_remain_mixed():
    result = classify(symbol="triton_fused_gelu_softplus_4")
    assert result["family"] == "FUSED_MIXED"
    assert result["candidate_families"] == ["GELU", "SOFTPLUS"]
