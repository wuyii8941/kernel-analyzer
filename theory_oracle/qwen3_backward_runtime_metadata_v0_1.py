#!/usr/bin/env python
"""Record metadata for every generated call in one valid Qwen3 compiled backward."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import theory_oracle.qwen3_grpo_natural_transition_v0_2 as base


def value_metadata(value: Any) -> dict[str, Any]:
    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "stride"):
        return {
            "kind": "tensor",
            "shape": [int(item) for item in value.shape],
            "stride": [int(item) for item in value.stride()],
            "dtype": str(value.dtype),
            "device": str(value.device),
            "storage_offset": int(value.storage_offset()),
            "requires_grad": bool(value.requires_grad),
        }
    if isinstance(value, (bool, int, float, str)):
        return {"kind": "scalar", "type": type(value).__name__, "value": value}
    try:
        return {"kind": "scalar_like", "type": type(value).__name__, "value": int(value)}
    except (TypeError, ValueError):
        return {"kind": "opaque", "type": f"{type(value).__module__}.{type(value).__name__}"}


def main() -> None:
    if "--kernel-summary" not in sys.argv:
        raise ValueError("--kernel-summary is required")
    summary_index = sys.argv.index("--kernel-summary")
    summary_path = Path(sys.argv[summary_index + 1]).resolve()
    del sys.argv[summary_index : summary_index + 2]
    if "--out-dir" not in sys.argv or "--arm" not in sys.argv:
        raise ValueError("base natural-transition arguments are required")
    out_dir = Path(sys.argv[sys.argv.index("--out-dir") + 1]).resolve()
    if sys.argv[sys.argv.index("--arm") + 1] != "compiled":
        raise ValueError("runtime metadata requires the compiled arm")

    summary = json.loads(summary_path.read_text())
    if summary["status"] != "VALID_BACKWARD_GENERATED_KERNEL_SUMMARY":
        raise ValueError("backward generated-kernel summary is invalid")
    expected_rows = {row["name"]: row for row in summary["kernel_families"]}
    expected_families = list(expected_rows)
    expected_extern = summary["extern_kernel_call_site_counts"]

    import torch
    from torch._inductor.select_algorithm import extern_kernels

    original_tensor_backward = torch.Tensor.backward
    census: dict[str, Any] = {
        "hook_calls": 0,
        "resolved_modules": {},
        "family_calls": {family: [] for family in expected_families},
        "external_calls": {"mm": [], "bmm": []},
    }

    class KernelProxy:
        def __init__(self, family: str, original: Any, module_name: str):
            self.family = family
            self.original = original
            self.module_name = module_name
            self.calls = 0

        def run(self, *values: Any, **kwargs: Any) -> Any:
            index = self.calls
            self.calls += 1
            census["family_calls"][self.family].append(
                {
                    "index": index,
                    "module": self.module_name,
                    "positional": [value_metadata(value) for value in values],
                    "keyword_types": {
                        key: f"{type(value).__module__}.{type(value).__name__}"
                        for key, value in kwargs.items()
                    },
                }
            )
            return self.original.run(*values, **kwargs)

    class ExternProxy:
        def __init__(self, operation: str, original: Any):
            self.operation = operation
            self.original = original
            self.calls = 0

        def __call__(self, *values: Any, **kwargs: Any) -> Any:
            index = self.calls
            self.calls += 1
            census["external_calls"][self.operation].append(
                {
                    "index": index,
                    "positional": [value_metadata(value) for value in values],
                    "keyword": {key: value_metadata(value) for key, value in kwargs.items()},
                }
            )
            return self.original(*values, **kwargs)

    def instrumented_backward(self: Any, *args: Any, **kwargs: Any) -> Any:
        census["hook_calls"] += 1
        if census["hook_calls"] != 1:
            raise RuntimeError("natural transition unexpectedly called Tensor.backward more than once")

        originals: dict[tuple[str, str], Any] = {}
        for module_name, module in list(sys.modules.items()):
            if module is None or not module_name.startswith("torch._inductor.runtime.compile_tasks."):
                continue
            for family in expected_families:
                if not hasattr(module, family):
                    continue
                key = (module_name, family)
                original = getattr(module, family)
                originals[key] = original
                setattr(module, family, KernelProxy(family, original, module_name))
                census["resolved_modules"].setdefault(family, []).append(module_name)

        extern_originals = {operation: getattr(extern_kernels, operation) for operation in ("mm", "bmm")}
        for operation, original in extern_originals.items():
            setattr(extern_kernels, operation, ExternProxy(operation, original))
        try:
            return original_tensor_backward(self, *args, **kwargs)
        finally:
            for (module_name, family), original in originals.items():
                setattr(sys.modules[module_name], family, original)
            for operation, original in extern_originals.items():
                setattr(extern_kernels, operation, original)

    torch.Tensor.backward = instrumented_backward
    try:
        base.main()
    finally:
        torch.Tensor.backward = original_tensor_backward

    result_path = out_dir / "result.json"
    if not result_path.is_file():
        raise RuntimeError("base natural transition did not produce result.json")
    result = json.loads(result_path.read_text())
    observed_counts = {family: len(rows) for family, rows in census["family_calls"].items()}
    expected_counts = {family: int(row["call_count"]) for family, row in expected_rows.items()}
    observed_external = {
        operation: len(rows) for operation, rows in census["external_calls"].items()
    }
    unresolved = sorted(set(expected_families) - set(census["resolved_modules"]))
    gates = {
        "base_transition_valid": result.get("verdict") == "VALID" and result.get("valid") is True,
        "candidate_identity_valid": result.get("compiler", {}).get("candidate_identity_valid") is True,
        "scorer_anchor_exact": result.get("anchors", {}).get("scorer_anchor_exact") is True,
        "single_backward_hook_call": census["hook_calls"] == 1,
        "all_static_families_resolved_before_backward": not unresolved,
        "family_call_counts_exact": observed_counts == expected_counts,
        "external_call_counts_exact": observed_external
        == {name: int(count) for name, count in expected_extern.items()},
        "metadata_complete_for_every_call": all(
            len(row["positional"]) > 0
            for rows in list(census["family_calls"].values())
            + list(census["external_calls"].values())
            for row in rows
        ),
    }
    payload = {
        "schema_version": "forkcert.qwen3-backward-runtime-metadata.v0.1",
        "status": "VALID_BACKWARD_RUNTIME_METADATA" if all(gates.values()) else "INVALID_METADATA",
        "source_static_summary": str(summary_path),
        "gates": gates,
        "resolved_modules": census["resolved_modules"],
        "unresolved_family_names": unresolved,
        "family_calls": census["family_calls"],
        "external_calls": census["external_calls"],
        "claim_limits": [
            "metadata only: shape, stride, dtype, device, storage offset and requires-grad",
            "no tensor values or storage contents are retained",
            "one compiled natural transition at one selected state",
            "counting proxies delegate unchanged arithmetic",
            "no repair, injection, causal attribution, population or correctness credit",
        ],
    }
    result["schema_version"] = "forkcert.qwen3-natural-transition-with-backward-metadata.v0.1"
    result["backward_runtime_metadata_status"] = payload["status"]
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out_dir / "backward_runtime_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "gates": gates,
                "family_calls": sum(observed_counts.values()),
                "external_calls": observed_external,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if payload["status"] != "VALID_BACKWARD_RUNTIME_METADATA":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
