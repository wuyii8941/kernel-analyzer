#!/usr/bin/env python
"""Revision 2 for singleton repairs whose generated buffers use different views."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import theory_oracle.qwen3_backward_singleton_repair_v0_1 as v1


def repair_silu_mul(torch: Any, values: tuple[Any, ...]) -> None:
    gate, up, destination = values[:3]
    repaired = torch.nn.functional.silu(gate) * up
    destination.reshape_as(repaired).copy_(repaired)


def repair_silu_mul_backward(torch: Any, values: tuple[Any, ...]) -> None:
    gate_gradient_destination, output_gradient, gate, up_gradient_destination = values[:4]
    gate_destination_view = gate_gradient_destination.reshape_as(output_gradient)
    up_destination_view = up_gradient_destination.reshape_as(output_gradient)
    upstream = gate_destination_view.clone()
    up_destination_view.copy_(output_gradient * torch.nn.functional.silu(gate))
    gate_destination_view.copy_(torch.ops.aten.silu_backward(output_gradient * upstream, gate))


def main() -> None:
    if "--treatment" not in sys.argv:
        raise ValueError("--treatment is required")
    treatment = sys.argv[sys.argv.index("--treatment") + 1]
    if treatment not in ("silu_mul", "silu_mul_backward"):
        raise ValueError("revision 2 only supports the two view-corrected SiLU treatments")
    if "--out-dir" not in sys.argv:
        raise ValueError("--out-dir is required")
    out_dir = Path(sys.argv[sys.argv.index("--out-dir") + 1]).resolve()
    v1.TREATMENTS["silu_mul"]["repair"] = repair_silu_mul
    v1.TREATMENTS["silu_mul_backward"]["repair"] = repair_silu_mul_backward
    v1.main()
    result_path = out_dir / "result.json"
    result = json.loads(result_path.read_text())
    result["schema_version"] = (
        "forkcert.qwen3-natural-transition-with-backward-singleton-repair.v0.2"
    )
    result["backward_singleton_repair"]["repair_revision"] = "v0.2-view-corrected"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
