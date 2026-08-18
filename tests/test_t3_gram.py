from kernel_analyzer.statistics import (
    coherence_certificate, coherence_certificate_from_gram,
)


def test_streamed_gram_certificate_matches_complete_vectors():
    vectors = [[1.0, 0.1], [1.1, -0.1], [0.9, 0.2], [1.2, 0.0]]
    gram = [[sum(a * b for a, b in zip(left, right)) for right in vectors]
            for left in vectors]
    direct = coherence_certificate(vectors, alpha=0.05, bootstrap_samples=400, seed=7)
    streamed = coherence_certificate_from_gram(
        gram, coordinate_count=2, alpha=0.05, bootstrap_samples=400, seed=7,
    )
    assert streamed["status"] == direct["status"] == "PASS"
    assert streamed["u_statistic"] == direct["u_statistic"]
    assert streamed["cluster_bootstrap_lower"] == direct["cluster_bootstrap_lower"]
