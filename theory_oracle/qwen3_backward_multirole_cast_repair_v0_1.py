#!/usr/bin/env python
"""Repair one role-mapped call of a repeated multi-role Qwen3 cast family."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import theory_oracle.qwen3_grpo_natural_transition_v0_2 as base


FAMILY = "triton_poi_fused__to_copy_26"
EXPECTED_CALLS = 84
ROLES = {
    "up_projection_weight_gradient_cast": 0,
    "gate_projection_weight_gradient_cast": 1,
    "down_projection_weight_gradient_cast": 2,
}


def pop_required_arg(flag: str) -> str:
    if flag not in sys.argv:
        raise ValueError(f"{flag} is required")
    index = sys.argv.index(flag)
    value = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    return value


def main() -> None:
    role = pop_required_arg("--selected-role")
    if role not in ROLES:
        raise ValueError(f"selected role must be one of {sorted(ROLES)}")
    selected_call = ROLES[role]
    if "--out-dir" not in sys.argv or "--arm" not in sys.argv:
        raise ValueError("base natural-transition arguments are required")
    out_dir = Path(sys.argv[sys.argv.index("--out-dir") + 1]).resolve()
    if sys.argv[sys.argv.index("--arm") + 1] != "compiled":
        raise ValueError("multi-role cast repair requires the compiled arm")

    import torch

    original_tensor_backward = torch.Tensor.backward
    counters: dict[str, Any] = {
        "backward_hooks": 0,
        "family_calls": 0,
        "repairs": 0,
        "selected_call_hits": 0,
        "modules": [],
    }

    class KernelProxy:
        def __init__(self, original: Any):
            self.original = original

        def run(self, *values: Any, **kwargs: Any) -> Any:
            call_index = counters["family_calls"]
            counters["family_calls"] += 1
            if call_index == selected_call:
                counters["selected_call_hits"] += 1
                source, destination = values[:2]
                destination.copy_(source.to(destination.dtype))
                counters["repairs"] += 1
                return None
            return self.original.run(*values, **kwargs)

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
        "single_repair": counters["repairs"] == 1,
        "selected_call_hit_once": counters["selected_call_hits"] == 1,
        "single_generated_module": len(counters["modules"]) == 1,
    }
    status = "VALID_BACKWARD_MULTIROLE_CAST_REPAIR" if all(gates.values()) else "INVALID_REPAIR"
    result["schema_version"] = (
        "forkcert.qwen3-natural-transition-with-backward-multirole-cast-repair.v0.1"
    )
    result["backward_multirole_cast_repair"] = {
        "status": status,
        "kernel_family": FAMILY,
        "expected_family_calls": EXPECTED_CALLS,
        "selected_role": role,
        "selected_call_index_zero_based": selected_call,
        "role_mapping_basis": (
            "first execution triple mapped from adjacent source-graph mm and convert nodes: "
            "up-projection, gate-projection, then down-projection weight gradients"
        ),
        "semantic_boundary": "one FP16 weight-gradient tensor converted to FP32",
        "counters": counters,
        "gates": gates,
        "claim_limits": [
            "one role-mapped invocation in the first-executed transformer-block triple",
            "no equivalence credit across the remaining layer positions",
            "eager conversion is a baseline rather than a correctness authority",
            "repair effect is intervention-dependent attribution rather than root cause",
            "repair only: no injection, necessity, sufficiency, population or long-run claim",
        ],
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "role": role, "gates": gates}, indent=2))
    if status != "VALID_BACKWARD_MULTIROLE_CAST_REPAIR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
