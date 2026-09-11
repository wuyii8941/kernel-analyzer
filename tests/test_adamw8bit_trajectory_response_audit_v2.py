from scripts.run_adamw8bit_trajectory_response_audit_v2 import distance


def test_parameter_distance_uses_all_coordinates():
    import torch

    result = distance([torch.tensor([2.0]), torch.tensor([3.0])],
                      [torch.tensor([1.0]), torch.tensor([1.0])])
    assert result["l2"] == 5 ** 0.5
    assert result["relative_l2"] == (5 / 2) ** 0.5
