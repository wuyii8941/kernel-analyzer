#!/usr/bin/env python
"""Proxy-forward sham for the repeated Qwen3 SiLU backward family."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import theory_oracle.qwen3_grpo_natural_transition_v0_2 as base


FAMILY = "triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_32"
EXPECTED_CALLS = 27


def main() -> None:
    if "--out-dir" not in sys.argv or "--arm" not in sys.argv:
        raise ValueError("base natural-transition arguments are required")
    out_dir = Path(sys.argv[sys.argv.index("--out-dir") + 1]).resolve()
    if sys.argv[sys.argv.index("--arm") + 1] != "compiled":
        raise ValueError("proxy sham requires the compiled arm")

    import torch

    original_tensor_backward = torch.Tensor.backward
    counters: dict[str, Any] = {
        "backward_hooks": 0,
        "family_calls": 0,
        "original_forwards": 0,
        "modules": [],
    }

    class KernelProxy:
        def __init__(self, original: Any):
            self.original = original

        def run(self, *values: Any, **kwargs: Any) -> Any:
            counters["family_calls"] += 1
            result = self.original.run(*values, **kwargs)
            counters["original_forwards"] += 1
            return result

    def instrumented_backward(self: Any, *args: Any, **kwargs: Any) -> Any:
        counters["backward_hooks"] += 1
        if counters["backward_hooks"] != 1:
            raise RuntimeError("natural transition unexpectedly called backward more than once")
        originals: dict[tuple[str, str], Any] = {}
        for module_name, module in list(sys.modules.items()):
            if module is None or not module_name.startswith("torch._inductor.runtime.compile_tasks."):
                continue
            if not hasattr(module, FAMILY):
                continue
            originals[(module_name, FAMILY)] = getattr(module, FAMILY)
            setattr(module, FAMILY, KernelProxy(getattr(module, FAMILY)))
            counters["modules"].append(module_name)
        if not originals:
            raise RuntimeError(f"generated family {FAMILY!r} was not resolved")
        try:
            return original_tensor_backward(self, *args, **kwargs)
        finally:
            for (module_name, name), original in originals.items():
                setattr(sys.modules[module_name], name, original)

    torch.Tensor.backward = instrumented_backward
    try:
        base.main()
    finally:
        torch.Tensor.backward = original_tensor_backward

    result_path = out_dir / "result.json"
    result = json.loads(result_path.read_text())
    gates = {
        "base_transition_valid": result.get("valid") is True and result.get("verdict") == "VALID",
        "candidate_identity_valid": result.get("compiler", {}).get("candidate_identity_valid") is True,
        "scorer_anchor_exact": result.get("anchors", {}).get("scorer_anchor_exact") is True,
        "single_backward_hook": counters["backward_hooks"] == 1,
        "expected_family_calls": counters["family_calls"] == EXPECTED_CALLS,
        "all_calls_forwarded_to_original": counters["original_forwards"] == EXPECTED_CALLS,
        "single_generated_module": len(counters["modules"]) == 1,
    }
    status = "VALID_BACKWARD_REPEATED_FAMILY_PROXY_SHAM" if all(gates.values()) else "INVALID_SHAM"
    result["schema_version"] = (
        "forkcert.qwen3-natural-transition-with-backward-repeated-family-proxy-sham.v0.1"
    )
    result["backward_repeated_family_proxy_sham"] = {
        "status": status,
        "kernel_family": FAMILY,
        "expected_family_calls": EXPECTED_CALLS,
        "counters": counters,
        "gates": gates,
        "control_scope": (
            "tests module monkey-patching and Python proxy dispatch while forwarding every "
            "call to the original compiled kernel"
        ),
        "uncontrolled_repair_mechanics": [
            "additional eager arithmetic kernel launches",
            "temporary tensor allocation and allocator-state perturbation",
            "changed fusion boundary and synchronization behavior",
        ],
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "gates": gates}, indent=2))
    if status != "VALID_BACKWARD_REPEATED_FAMILY_PROXY_SHAM":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
