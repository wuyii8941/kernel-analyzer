import json

import pytest

from scripts.audit_aligned_projection_migration import ROOT, audit, compare, legacy_stages


def test_identical_reference_lengths_rescale_the_interval_without_changing_verdict():
    result = compare([2., 3., 4., 5.], [9.] * 4, alpha=.05)
    assert result["decision_changed"] is False
    assert result["unit_projection"]["mean"] == pytest.approx(3 * result["legacy_gain"]["mean"])
    assert result["unit_projection"]["interval"] == pytest.approx(
        [3 * value for value in result["legacy_gain"]["interval"]])


def test_varying_reference_lengths_can_change_the_mean_in_either_direction():
    for sign in (-1, 1):
        result = compare([sign * x for x in [1., -100.] * 32], [1., 10000.] * 32, alpha=.025)
        assert result["legacy_gain"]["label"] == ("POSITIVE" if sign == 1 else "NEGATIVE")
        assert result["unit_projection"]["label"] == "NOT_CONFIRMED"
        assert result["decision_changed"] is True


def test_missing_direction_is_separate_from_a_normalization_change():
    result = compare([0., 1., 2.], [0., 1., 1.], alpha=.05)
    assert result["same_observation_set"] is False
    assert result["unit_projection"]["label"] == "NOT_ASSESSED"


def test_inventory_does_not_count_convenience_aliases_twice():
    stage = {"aligned_statewise_gain_interval": {"mean": 1.}}
    report = {"stages": {"OUTPUT": stage, "BACKWARD": stage}, "output": stage, "backward": stage}
    assert [path for path, _ in legacy_stages(report)] == ["/stages/OUTPUT", "/stages/BACKWARD"]


def test_saved_migration_result_is_reproducible_without_rerunning_kernels():
    saved = json.loads((ROOT / "results/property/root_cause_closure_v1/aligned_projection_migration_v1.json").read_text())
    recomputed = audit()
    assert recomputed["summary"] == saved["summary"]
    for section in ("history_comparisons", "finite_bank_comparisons", "exact_benchmark_sensitivity"):
        for expected, actual in zip(saved[section], recomputed[section]):
            assert actual["decision_changed"] == expected["decision_changed"]
            for endpoint in ("legacy_gain", "unit_projection"):
                assert actual[endpoint]["interval"] == pytest.approx(expected[endpoint]["interval"])
    assert recomputed["summary"]["history_endpoint_changes"] == 0
    assert recomputed["summary"]["exact_benchmark_sensitivity_changes"] == 3


@pytest.mark.parametrize("case", ["forward_asymmetric", "forward_symmetric",
                                  "interleaved_asymmetric", "interleaved_symmetric"])
def test_current_checker_replays_saved_scalar_observations(case):
    import torch
    from kernel_analyzer import check_bias

    # Reuse measured scalar outputs, not a substitute GPU implementation.
    source = "results/property/tilelang_sum_spec_experiment_v2.json"
    saved = json.loads((ROOT / source).read_text())
    protocol = saved["protocol"]
    row = saved["results"][case]
    distribution = "asymmetric" if case.endswith("_asymmetric") else "symmetric"
    census = {item["index"]: item for item in row["census"]}

    def inputs(i):
        item = census[protocol["draws"][distribution][i]]
        return torch.tensor([item["candidate"], item["exact_sum"]], dtype=torch.float64)

    actual = check_bias(lambda x: x[:1], lambda x: x[1:], inputs,
                        samples=protocol["samples"], calibration_samples=protocol["samples"] // 4,
                        alpha=protocol["case_alpha"])
    audit_record = json.loads((ROOT / "results/property/root_cause_closure_v1/aligned_projection_migration_v1.json").read_text())
    expected = next(item for item in audit_record["finite_bank_comparisons"]
                    if item["source"] == source and item["case"] == case)
    assert actual["output"]["aligned_projection_interval"]["interval"] == pytest.approx(
        expected["unit_projection"]["interval"])
    assert actual["output"]["mean_bias_decision"] == row["sampled_bias_report"]["output"]["mean_bias_decision"]
