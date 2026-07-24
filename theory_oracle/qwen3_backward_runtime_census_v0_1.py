#!/usr/bin/env python
"""Instrument one valid compiled natural transition to census backward kernels."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import theory_oracle.qwen3_grpo_natural_transition_v0_2 as base


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
        raise ValueError("runtime census requires the compiled arm")

    summary = json.loads(summary_path.read_text())
    if summary["status"] != "VALID_BACKWARD_GENERATED_KERNEL_SUMMARY":
        raise ValueError("backward generated-kernel summary is invalid")
    expected_families = [row["name"] for row in summary["kernel_families"]]

    import torch
    from torch._inductor.select_algorithm import extern_kernels

    original_tensor_backward = torch.Tensor.backward
    census: dict[str, Any] = {
        "hook_calls": 0,
        "resolved_modules": {},
        "family_call_counts": {},
        "external_call_counts": {"mm": 0, "bmm": 0},
    }

    class KernelProxy:
        def __init__(self, family: str, original: Any):
            self.family = family
            self.original = original
            self.calls = 0

        def run(self, *values: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return self.original.run(*values, **kwargs)

    class ExternProxy:
        def __init__(self, operation: str, original: Any):
            self.operation = operation
            self.original = original
            self.calls = 0

        def __call__(self, *values: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return self.original(*values, **kwargs)

    def instrumented_backward(self: Any, *args: Any, **kwargs: Any) -> Any:
        census["hook_calls"] += 1
        if census["hook_calls"] != 1:
            raise RuntimeError("natural transition unexpectedly called Tensor.backward more than once")

        originals: dict[tuple[str, str], Any] = {}
        proxies: dict[tuple[str, str], KernelProxy] = {}
        for module_name, module in list(sys.modules.items()):
            if module is None or not module_name.startswith("torch._inductor.runtime.compile_tasks."):
                continue
            for family in expected_families:
                if not hasattr(module, family):
                    continue
                key = (module_name, family)
                original = getattr(module, family)
                proxy = KernelProxy(family, original)
                originals[key] = original
                proxies[key] = proxy
                setattr(module, family, proxy)
                census["resolved_modules"].setdefault(family, []).append(module_name)

        extern_originals = {operation: getattr(extern_kernels, operation) for operation in ("mm", "bmm")}
        extern_proxies = {
            operation: ExternProxy(operation, original)
            for operation, original in extern_originals.items()
        }
        for operation, proxy in extern_proxies.items():
            setattr(extern_kernels, operation, proxy)
        try:
            return original_tensor_backward(self, *args, **kwargs)
        finally:
            for (module_name, family), original in originals.items():
                setattr(sys.modules[module_name], family, original)
            for operation, original in extern_originals.items():
                setattr(extern_kernels, operation, original)
            census["family_call_counts"] = {
                family: sum(
                    proxy.calls for (_, proxy_family), proxy in proxies.items() if proxy_family == family
                )
                for family in expected_families
            }
            census["external_call_counts"] = {
                operation: proxy.calls for operation, proxy in extern_proxies.items()
            }

    torch.Tensor.backward = instrumented_backward
    try:
        base.main()
    finally:
        torch.Tensor.backward = original_tensor_backward

    result_path = out_dir / "result.json"
    if not result_path.is_file():
        raise RuntimeError("base natural transition did not produce result.json")
    result = json.loads(result_path.read_text())
    family_counts = census["family_call_counts"]
    active_families = sorted(family for family, count in family_counts.items() if count > 0)
    unresolved = sorted(set(expected_families) - set(census["resolved_modules"]))
    gates = {
        "base_transition_valid": result.get("verdict") == "VALID" and result.get("valid") is True,
        "candidate_identity_valid": result.get("compiler", {}).get("candidate_identity_valid") is True,
        "scorer_anchor_exact": result.get("anchors", {}).get("scorer_anchor_exact") is True,
        "single_backward_hook_call": census["hook_calls"] == 1,
        "all_static_families_resolved_before_backward": not unresolved,
        "at_least_one_backward_family_called": bool(active_families),
        "external_backward_calls_observed": all(
            census["external_call_counts"][operation] > 0 for operation in ("mm", "bmm")
        ),
    }
    census_payload = {
        "schema_version": "forkcert.qwen3-backward-runtime-census.v0.1",
        "status": "VALID_BACKWARD_RUNTIME_CENSUS" if all(gates.values()) else "INVALID_CENSUS",
        "source_static_summary": str(summary_path),
        "gates": gates,
        "expected_static_triton_families": len(expected_families),
        "resolved_triton_families": len(census["resolved_modules"]),
        "active_triton_families": len(active_families),
        "active_family_names": active_families,
        "unresolved_family_names": unresolved,
        "family_call_counts": family_counts,
        "external_call_counts_during_backward": census["external_call_counts"],
        "claim_limits": [
            "one compiled natural transition at one selected state",
            "runtime census only; proxy delegates unchanged kernels and external calls",
            "gradient-checkpoint recomputation is part of the observed backward execution",
            "no repair, injection, causal attribution, population or correctness credit",
        ],
    }
    result["schema_version"] = "forkcert.qwen3-natural-transition-with-backward-census.v0.1"
    result["backward_runtime_census"] = census_payload
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out_dir / "backward_runtime_census.json").write_text(
        json.dumps(census_payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(census_payload, indent=2, sort_keys=True))
    if census_payload["status"] != "VALID_BACKWARD_RUNTIME_CENSUS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
