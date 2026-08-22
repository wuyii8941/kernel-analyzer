from kernel_analyzer.joint_bias import even_odd_response, parity_decomposition, prefix_resultant


def test_even_odd_decomposition_is_exact():
    even, odd = even_odd_response([3.0, -1.0], [1.0, 5.0])
    assert even == (2.0, 2.0)
    assert odd == (1.0, -3.0)


def test_parity_decomposition_closes():
    result = parity_decomposition([[3.0, 1.0], [5.0, -1.0]], [[1.0, 1.0], [1.0, 3.0]])
    assert result["closure_l2"] == 0.0
    assert result["mu"] == tuple(a + b for a, b in zip(result["mu_even"], result["mu_odd"]))


def test_prefix_requires_vector_or_scalar_trace_explicitly():
    result = prefix_resultant([{"step": 1, "value": 1.0}, {"step": 2, "value": -1.0}], "value")
    assert result["status"] == "COMPLETE"
    assert "2" in result["prefixes"]
    missing = prefix_resultant([{"step": 1, "norm": 1.0}], "vector")
    assert missing["status"] == "UNRESOLVED_MISSING_VECTOR_TRACE"
