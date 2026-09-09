import torch

from kernel_analyzer.directional_error_sketch import (
    fixed_coordinate_error_sketch,
    fixed_flat_coordinate_indices,
)


def test_fixed_coordinates_do_not_depend_on_values():
    expected = fixed_flat_coordinate_indices(10, sample_size=4)
    assert expected.tolist() == [0, 3, 6, 9]
    left = torch.arange(10, dtype=torch.float32)
    right = left + 1
    result = fixed_coordinate_error_sketch(left, right, sample_size=4)
    assert result["flat_coordinate_indices"] == expected.tolist()
    assert result["signed_delta_values"] == [-1.0] * 4
    assert result["candidate_values_used_to_select_coordinates"] is False
