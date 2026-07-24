from scripts.phase15_bf16_external_audit import audit_payload
from scripts.phase15_bf16_preflight import classify_device


def test_preflight_rejects_t4() -> None:
    passed, errors = classify_device(
        device_count=1,
        capability=(7, 5),
        bf16_supported=False,
        require_flash=False,
    )
    assert not passed
    assert any("SM80" in error for error in errors)


def test_preflight_accepts_native_bf16() -> None:
    passed, errors = classify_device(
        device_count=1,
        capability=(8, 0),
        bf16_supported=True,
        require_flash=False,
    )
    assert passed
    assert errors == []


def test_external_audit_requires_true_bf16_labels() -> None:
    preflight = {
        "passed": True,
        "device": {"capability": [8, 0], "bf16_supported": True, "name": "A100"},
    }
    training = {"training_compute_dtype": "bf16"}
    rows = [
        {
            "training_compute_dtype": "bf16",
            "path_ref": "hf-eager-bf16-sdpa-math-online",
            "path_alt": "hf-compile-bf16-sdpa-math-online",
            "delta_self_ref": 0.0,
            "delta_self_alt": 0.0,
        }
    ]
    certificates = [{"advantage_sign": 1, "actual_fork": True, "region": "unknown"}]
    payload = audit_payload(preflight, training, rows, certificates, expected_rows=1)
    assert payload["passed"]
    assert payload["actual_clipping_forks"] == 1


def test_external_audit_rejects_fp16_masquerading_as_bf16() -> None:
    preflight = {
        "passed": True,
        "device": {"capability": [8, 0], "bf16_supported": True, "name": "A100"},
    }
    training = {"training_compute_dtype": "fp16"}
    rows = [
        {
            "training_compute_dtype": "fp16",
            "path_ref": "hf-eager-fp16-sdpa-math-online",
            "path_alt": "hf-compile-fp16-sdpa-math-online",
            "delta_self_ref": 0.0,
            "delta_self_alt": 0.0,
        }
    ]
    certificates = [{"advantage_sign": 1, "actual_fork": False, "region": "unknown"}]
    assert not audit_payload(preflight, training, rows, certificates, expected_rows=1)["passed"]
