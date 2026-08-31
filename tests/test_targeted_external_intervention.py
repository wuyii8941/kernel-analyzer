from types import SimpleNamespace

import torch
import numpy as np

from scripts.targeted_external_intervention import TargetedExternalIntervention, _count_sketch
from scripts.run_targeted_full_coordinate import nondegenerate_bootstrap_counts


def test_rejects_non_external_target() -> None:
    target = {
        "implementation_kind": "TRITON", "function": "triton_x",
        "source_line_sha256": "x",
    }
    try:
        TargetedExternalIntervention(modules=[], target=target, mode="OBSERVE")
    except ValueError as error:
        assert "EXTERN" in str(error)
    else:
        raise AssertionError("non-external target was accepted")


def test_rejects_unknown_mode() -> None:
    target = {
        "implementation_kind": "EXTERN", "function": "extern_kernels.mm",
        "source_line_sha256": "x",
    }
    try:
        TargetedExternalIntervention(
            modules=[SimpleNamespace(extern_kernels=SimpleNamespace(mm=torch.mm))],
            target=target, mode="UNKNOWN",
        )
    except ValueError as error:
        assert "mode" in str(error)
    else:
        raise AssertionError("unknown mode was accepted")


def test_pilot_bootstrap_counts_are_never_degenerate() -> None:
    counts = nondegenerate_bootstrap_counts(4, 4000, 14031)
    assert counts.shape == (4000, 4)
    assert np.all(np.count_nonzero(counts, axis=1) >= 2)
    assert np.all(counts.sum(axis=1) == 4)


def test_v2_count_sketch_does_not_periodically_alias_stride_dimension() -> None:
    dimension = 64
    basis = []
    for coordinate in (7, 7 + dimension, 7 + 2 * dimension, 7 + 3 * dimension):
        vector = torch.zeros(4 * dimension)
        vector[coordinate] = 1.0
        basis.append(_count_sketch(vector, dimension=dimension, seed=23))
    assert any(not torch.equal(basis[0], other) for other in basis[1:])


def test_v2_count_sketch_seed_is_part_of_the_measurement() -> None:
    vector = torch.arange(1024, dtype=torch.float32)
    first = _count_sketch(vector, dimension=64, seed=11)
    second = _count_sketch(vector, dimension=64, seed=12)
    assert not torch.equal(first, second)
