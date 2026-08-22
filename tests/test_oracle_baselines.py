from scripts.analyze_oracle_baselines import auc, operating_point, split_top_level
from scripts.freeze_phi_carrier_distribution import evenly_spaced_indices


def test_auc_uses_pairwise_ranking_and_ties():
    assert auc([True, False], [2.0, 1.0]) == 1.0
    assert auc([True, False], [1.0, 2.0]) == 0.0
    assert auc([True, False], [1.0, 1.0]) == 0.5


def test_operating_point_reports_miss_and_false_positive_rates():
    result = operating_point([True, False, False], [1.2, 1.1, 0.9], 1.0)
    assert result["recall"] == 1.0
    assert result["miss_rate"] == 0.0
    assert result["false_positive_rate"] == 0.5
    assert result["flag_rate"] == 2 / 3


def test_parser_keeps_nested_call_arguments_together():
    assert split_top_level("reinterpret_tensor(x, (2, 3), (3, 1), 0), y") == [
        "reinterpret_tensor(x, (2, 3), (3, 1), 0)", "y"
    ]


def test_carrier_depth_grid_is_deterministic_and_unique():
    assert evenly_spaced_indices(31, 6) == [0, 6, 12, 19, 25, 31]
    assert evenly_spaced_indices(31, 5) == [0, 8, 16, 23, 31]
