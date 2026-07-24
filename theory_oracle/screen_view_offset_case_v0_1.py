#!/usr/bin/env python
"""Environment screen for a closed storage-offset/view wrong-output case."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def do_view(value: torch.Tensor) -> torch.Tensor:
    dout, din = value.shape
    return value.view(-1, 1).view(dout, din)


def main() -> None:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    inputs = torch.randn(4, 128, 16, device="cuda", dtype=torch.float32)
    second = torch.randn(4, 171, 6, device="cuda", dtype=torch.float32)
    compiled = torch.compile(do_view, fullgraph=True, backend="inductor")
    first_rows = []
    for row in inputs:
        first_rows.append(bool(torch.equal(compiled(row), row)))
    second_rows = []
    for row in second:
        second_rows.append(bool(torch.equal(compiled(row), row)))
    torch.cuda.synchronize()
    result = {
        "schema_version": "forkcert.view_offset_environment_screen.v0.1",
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "first_shape_equal": first_rows,
        "second_shape_equal": second_rows,
        "silent_disagreement_observed": not all(second_rows),
        "claim_scope": "environment screening only; no patch or root-cause claim",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    Path("/data1/tzh/forkcert/results/operator_oracle/view_offset_environment_screen_v0_1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
