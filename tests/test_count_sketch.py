from __future__ import annotations

import numpy as np

from kernel_analyzer.short_persistence import count_sketch_mapping


def test_count_sketch_sign_is_not_determined_by_bucket() -> None:
    buckets, signs = count_sketch_mapping(
        np.arange(65536, dtype=np.uint64), projection_dim=4096, seed=20260831
    )
    assert any(
        np.unique(signs[buckets == bucket]).size > 1
        for bucket in np.unique(buckets)
    )


def test_count_sketch_mapping_is_reproducible() -> None:
    indices = np.arange(1000, dtype=np.uint64)
    first = count_sketch_mapping(indices, projection_dim=128, seed=17)
    second = count_sketch_mapping(indices, projection_dim=128, seed=17)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
