"""Real CPU execution through the existing capture and analysis components.

These are engineering checks, not new bias cases. The wrapper explicitly
supplies operands, reference, loss and optimizer; the collector does not infer
them from an arbitrary operator file.
"""

import math

import pytest
import torch

from kernel_analyzer.fixed_suite_implementation_capture import FixedSuiteImplementationCapture
from kernel_analyzer.local_tolerance import compare_outputs
from kernel_analyzer.training_numerical_analysis import analyze_artifact


STAGES = ("LOCAL", "PARAMETER_GRADIENT", "PARAMETER_WRITE")
PROTOCOL = {
    "schema": "operator-pipeline-cpu-smoke-v1",
    "claim_scope": "FIXED_SUITE_UPDATE",
    "primary_stage": "PARAMETER_WRITE",
    "fixed_suite_margins": {"full_update_rms": 0.01},
}


def implementation(family, variant):
    if family == "silu":
        reference = torch.nn.functional.silu
        alternative = lambda x: x * torch.sigmoid(x)
    else:
        reference = lambda x: x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
        alternative = lambda x: x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-5)
    if variant == "same":
        return reference, reference
    if variant == "alternative":
        return alternative, reference
    # Deliberately wrong semantics: ensures an executed error is not silently
    # accepted. This is not a proposed implementation variant or natural case.
    return lambda x: -reference(x), reference


def execute_step(operator, initial, operands, cotangent):
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.0, foreach=False)
    before = parameter.detach().clone()
    output = operator(parameter * operands)
    (output * cotangent).sum().backward()
    gradient = parameter.grad.detach().clone()
    optimizer.step()
    return {
        "LOCAL": output.detach().clone(),
        "PARAMETER_GRADIENT": gradient,
        "PARAMETER_WRITE": parameter.detach().clone() - before,
    }


@pytest.mark.parametrize("family", ["silu", "rms_norm"])
@pytest.mark.parametrize("variant", ["same", "alternative", "deliberately_wrong"])
def test_operator_backward_write_capture_and_analysis(family, variant):
    candidate, reference = implementation(family, variant)
    state_ids = [f"fixed-state-{index}" for index in range(32)]
    capture = FixedSuiteImplementationCapture(state_ids, STAGES)
    direct = {stage: [] for stage in STAGES}
    for index in range(32):
        initial = torch.linspace(-1.7, 2.1, 16).reshape(2, 8) + index * 0.007
        operands = torch.linspace(0.6, 1.3, 16).reshape(2, 8)
        cotangent = torch.cos(torch.arange(16, dtype=torch.float32)).reshape(2, 8)
        measured = execute_step(candidate, initial, operands, cotangent)
        baseline = execute_step(reference, initial, operands, cotangent)
        local = compare_outputs(measured["LOCAL"], baseline["LOCAL"], rtol=1e-5, atol=1e-7)
        assert local["status"] == "FINITE_COMPARISON"
        if variant == "same":
            assert local["allclose"] is True
        elif variant == "deliberately_wrong":
            assert local["allclose"] is False
        capture.append({stage: (measured[stage], baseline[stage]) for stage in STAGES})
        for stage in STAGES:
            effect = (measured[stage] - baseline[stage]).double().reshape(-1)
            repair = baseline[stage].double().reshape(-1)
            direct[stage].append((float(effect @ effect), float(repair @ repair), float(effect @ repair)))

    rows, profiles = capture.finish()
    for stage in STAGES:
        for row, expected in zip(rows[stage], direct[stage]):
            assert row["effect_energy"] == pytest.approx(expected[0], rel=1e-12, abs=1e-30)
            assert row["repair_energy"] == pytest.approx(expected[1], rel=1e-12, abs=1e-30)
            assert row["effect_repair_inner_product"] == pytest.approx(expected[2], rel=1e-12, abs=1e-30)

    raw = {
        "case_id": f"cpu-smoke-{family}-{variant}",
        "contrast_id": "CONTROLLED_ENGINEERING_TEST",
        "status": "COMPLETE",
        "runtime_boundary": {"kind": "CPU_PYTORCH_AUTOGRAD", "operator": family},
        "parameter_write_protocol": {
            "version": "adamw-readback-v2",
            "measurement": "parameter_after_step_minus_parameter_before_step",
        },
        "state_ids": state_ids,
        "calibration_state_ids": state_ids[:16],
        "confirmation_state_ids": state_ids[16:],
        "original_coordinate_statistics": rows,
        "stages": profiles,
    }
    report = analyze_artifact(raw, PROTOCOL)
    assert report["measurement_status"] == "VALID"
    assert report["bias_analysis"]["population_guarantee"] is False
    confirmation = direct["PARAMETER_WRITE"][16:]
    expected_rms = math.sqrt(math.fsum(row[0] for row in confirmation) / math.fsum(row[1] for row in confirmation))
    expected_aligned = math.fsum(row[2] for row in confirmation) / math.fsum(row[1] for row in confirmation)
    assert report["bias_analysis"]["fixed_suite_total_rms"] == pytest.approx(expected_rms)
    assert report["bias_analysis"]["fixed_suite_aligned_ratio_of_sums"] == pytest.approx(expected_aligned)
    if variant == "same":
        assert report["equivalence_decision"] == "EQUIVALENT"
        assert report["bias_analysis"]["identity_verified_by_coordinate_count"] is True
    elif variant == "deliberately_wrong":
        assert report["equivalence_decision"] == "NON_EQUIVALENT"

    # Merely naming a proposed update as a write must not certify it.
    raw["parameter_write_protocol"]["measurement"] = "optimizer_proposal_before_write"
    rejected = analyze_artifact(raw, PROTOCOL)
    assert rejected["measurement_status"] == "PARTIAL"
    assert rejected["equivalence_decision"] == "NOT_ASSESSED"
