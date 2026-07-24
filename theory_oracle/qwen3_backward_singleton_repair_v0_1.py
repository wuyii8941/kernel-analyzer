#!/usr/bin/env python
"""Run one fail-closed eager-semantic repair of a singleton Qwen3 backward kernel."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import theory_oracle.qwen3_grpo_natural_transition_v0_2 as base


Repair = Callable[[Any, tuple[Any, ...]], None]


def repair_cast_view(torch: Any, values: tuple[Any, ...]) -> None:
    source, destination = values[:2]
    destination.copy_(source.reshape_as(destination).to(destination.dtype))


def repair_embedding_zero(torch: Any, values: tuple[Any, ...]) -> None:
    values[0].zero_()


def repair_silu_mul(torch: Any, values: tuple[Any, ...]) -> None:
    gate, up, destination = values[:3]
    destination.copy_(torch.nn.functional.silu(gate) * up)


def repair_silu_mul_backward(torch: Any, values: tuple[Any, ...]) -> None:
    gate_gradient_destination, output_gradient, gate, up_gradient_destination = values[:4]
    upstream = gate_gradient_destination.clone()
    up_gradient_destination.copy_(output_gradient * torch.nn.functional.silu(gate))
    gate_gradient_destination.copy_(
        torch.ops.aten.silu_backward(output_gradient * upstream, gate)
    )


def repair_fp16_fp32_add(torch: Any, values: tuple[Any, ...]) -> None:
    half_input, float_input, destination = values[:3]
    destination.copy_(half_input.float() + float_input)


TREATMENTS: dict[str, dict[str, Any]] = {
    "cast_view": {
        "family": "triton_poi_fused__to_copy_view_0",
        "repair": repair_cast_view,
        "semantic_boundary": "FP32 tangent reshaped and converted to FP16",
    },
    "embedding_zero": {
        "family": "triton_poi_fused_embedding_dense_backward_5",
        "repair": repair_embedding_zero,
        "semantic_boundary": "dense embedding-gradient accumulation buffer initialized to zero",
    },
    "silu_mul": {
        "family": "triton_poi_fused__unsafe_view_mul_silu_15",
        "repair": repair_silu_mul,
        "semantic_boundary": "FP16 SiLU followed by FP16 multiply",
    },
    "silu_mul_backward": {
        "family": "triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_17",
        "repair": repair_silu_mul_backward,
        "semantic_boundary": "separate FP16 multiply and SiLU-backward operations",
    },
    "fp16_fp32_add": {
        "family": "triton_poi_fused__to_copy_add_38",
        "repair": repair_fp16_fp32_add,
        "semantic_boundary": "explicit FP16-to-FP32 conversion followed by FP32 add",
    },
}


def pop_required_arg(flag: str) -> str:
    if flag not in sys.argv:
        raise ValueError(f"{flag} is required")
    index = sys.argv.index(flag)
    value = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    return value


def main() -> None:
    treatment_name = pop_required_arg("--treatment")
    if treatment_name not in TREATMENTS:
        raise ValueError(f"unsupported treatment {treatment_name!r}")
    treatment = TREATMENTS[treatment_name]
    if "--out-dir" not in sys.argv or "--arm" not in sys.argv:
        raise ValueError("base natural-transition arguments are required")
    out_dir = Path(sys.argv[sys.argv.index("--out-dir") + 1]).resolve()
    if sys.argv[sys.argv.index("--arm") + 1] != "compiled":
        raise ValueError("singleton repair requires the compiled arm")

    import torch

    original_tensor_backward = torch.Tensor.backward
    counters = {"backward_hooks": 0, "family_calls": 0, "repairs": 0, "modules": []}

    class KernelProxy:
        def __init__(self, original: Any):
            self.original = original

        def run(self, *values: Any, **kwargs: Any) -> None:
            counters["family_calls"] += 1
            if counters["family_calls"] != 1:
                raise RuntimeError("singleton family executed more than once")
            treatment["repair"](torch, values)
            counters["repairs"] += 1

    def instrumented_backward(self: Any, *args: Any, **kwargs: Any) -> Any:
        counters["backward_hooks"] += 1
        if counters["backward_hooks"] != 1:
            raise RuntimeError("natural transition unexpectedly called backward more than once")
        originals: dict[tuple[str, str], Any] = {}
        family = treatment["family"]
        for module_name, module in list(sys.modules.items()):
            if module is None or not module_name.startswith("torch._inductor.runtime.compile_tasks."):
                continue
            if not hasattr(module, family):
                continue
            originals[(module_name, family)] = getattr(module, family)
            setattr(module, family, KernelProxy(getattr(module, family)))
            counters["modules"].append(module_name)
        if not originals:
            raise RuntimeError(f"generated family {family!r} was not resolved")
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
        "single_family_call": counters["family_calls"] == 1,
        "single_repair": counters["repairs"] == 1,
        "single_generated_module": len(counters["modules"]) == 1,
    }
    status = "VALID_BACKWARD_SINGLETON_REPAIR" if all(gates.values()) else "INVALID_REPAIR"
    result["schema_version"] = "forkcert.qwen3-natural-transition-with-backward-singleton-repair.v0.1"
    result["backward_singleton_repair"] = {
        "status": status,
        "treatment": treatment_name,
        "kernel_family": treatment["family"],
        "semantic_boundary": treatment["semantic_boundary"],
        "counters": counters,
        "gates": gates,
        "claim_limits": [
            "one generated singleton invocation at one selected state",
            "eager-semantic local replacement, not mathematical ground truth",
            "repair effect is intervention-dependent attribution, not root cause",
            "repair only: no injection, sufficiency, population or long-run claim",
        ],
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "treatment": treatment_name, "gates": gates}, indent=2))
    if status != "VALID_BACKWARD_SINGLETON_REPAIR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
