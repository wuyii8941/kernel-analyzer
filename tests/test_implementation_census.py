import importlib.util
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "build_implementation_census",
    Path(__file__).parents[1] / "scripts" / "build_implementation_census.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def _record(shape, ordinal):
    return {
        "region_id": "forward:0",
        "phase": "FORWARD",
        "symbol": "triton_red_fused_sum_0",
        "runtime_invocation_ordinal": ordinal,
        "typed_reference_program_sha256": "a" * 64,
        "runtime_pointer_contracts": {
            "in_ptr0": {
                "shape": shape,
                "stride": [1],
                "dtype": "torch.bfloat16",
                "device_type": "cuda",
                "layout": "torch.strided",
                "storage_offset": 0,
            }
        },
    }


def test_repeated_invocations_remain_denominator_but_share_exact_identity(monkeypatch):
    document = {
        "states": {
            "s0": {
                "repeats": [
                    {"summary": {"records": [_record([8], 0), _record([8], 1)]}},
                    {"summary": {"records": [_record([8], 0), _record([8], 1)]}},
                ]
            }
        }
    }
    monkeypatch.setattr(_MODULE, "_read", lambda _: document)
    result = _MODULE.build([Path("screen.json")])
    assert result["denominator"]["runtime_invocations"] == 2
    assert result["denominator"]["unique_exact_implementations"] == 1
    assert result["implementations"][0]["invocation_count"] == 2


def test_abi_change_creates_new_exact_implementation(monkeypatch):
    document = {
        "states": {
            "s0": {
                "repeats": [{"summary": {"records": [_record([8], 0), _record([16], 1)]}}]
            }
        }
    }
    monkeypatch.setattr(_MODULE, "_read", lambda _: document)
    result = _MODULE.build([Path("screen.json")])
    assert result["denominator"]["runtime_invocations"] == 2
    assert result["denominator"]["unique_exact_implementations"] == 2


def test_missing_runtime_abi_is_unresolved_not_assumed_duplicate(monkeypatch):
    record = _record([8], 0)
    record.pop("runtime_pointer_contracts")
    document = {"states": {"s0": {"repeats": [{"summary": {"records": [record]}}]}}}
    monkeypatch.setattr(_MODULE, "_read", lambda _: document)
    result = _MODULE.build([Path("screen.json")])
    assert result["status"] == "PARTIAL_LEGACY_ABI_UNRESOLVED"
    assert result["denominator"]["identity_unresolved_invocations"] == 1
