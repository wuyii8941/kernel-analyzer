from pathlib import Path

import numpy as np

from kernel_analyzer.streaming import (
    StreamingGramAccumulator,
    direction_certificate_from_vector_files,
)


def test_streaming_complete_gram_detects_direction_and_cleans(tmp_path: Path):
    accumulator = StreamingGramAccumulator(tmp_path, "positive", chunk_elements=2)
    for index, scale in enumerate((1.0, 0.9, 1.1, 0.8, 1.2, 1.0)):
        accumulator.add_array(f"s{index}", np.array([scale, 0.1, 0.0]))
    result = accumulator.finalize(bootstrap_draws=400, seed=3)
    assert result["status"] == "PASS"
    assert result["coordinates"] == 3
    assert result["streamed_complete_gram"] is True
    assert list(tmp_path.glob("*.f64")) == []


def test_streaming_complete_gram_rejects_sign_changing_vectors(tmp_path: Path):
    accumulator = StreamingGramAccumulator(tmp_path, "changing")
    for index, sign in enumerate((1, -1, 1, -1, 1, -1)):
        accumulator.add_array(f"s{index}", np.array([sign, 0.0]))
    result = accumulator.finalize(bootstrap_draws=400, seed=3)
    assert result["status"] == "FAIL_CAUSAL_NONCOHERENT"


def test_streaming_bootstrap_does_not_turn_diagonal_energy_into_bias(tmp_path: Path):
    accumulator = StreamingGramAccumulator(tmp_path, "orthogonal")
    for index in range(6):
        value = np.zeros(6)
        value[index] = 1.0
        accumulator.add_array(f"s{index}", value)
    result = accumulator.finalize(bootstrap_draws=400, seed=3)
    assert result["cross_state_inner_product_u"] == 0.0
    assert result["cluster_bootstrap_95"]["lower_95"] == 0.0
    assert result["status"] == "FAIL_CAUSAL_NONCOHERENT"
    assert result["bootstrap_excludes_same_original_cluster_pairs"] is True


def test_direct_float32_spools_and_zero_marker_match_complete_gram(tmp_path: Path):
    rows = []
    for state_id, values in (("a", np.array([1, 2], np.float32)),
                             ("b", np.array([2, 1], np.float32))):
        path = tmp_path / f"{state_id}.f32"
        path.write_bytes(values.tobytes())
        rows.append({"state_id": state_id, "path": str(path),
                     "storage_dtype": "float32", "coordinates": 2,
                     "sha256": state_id, "constant_zero": False})
    rows.append({"state_id": "z", "path": None, "storage_dtype": "float32",
                 "coordinates": 2, "sha256": "z", "constant_zero": True})
    result = direction_certificate_from_vector_files(rows, bootstrap_draws=100, seed=2)
    assert result["gram"] == [[5.0, 4.0, 0.0], [4.0, 5.0, 0.0], [0.0, 0.0, 0.0]]
