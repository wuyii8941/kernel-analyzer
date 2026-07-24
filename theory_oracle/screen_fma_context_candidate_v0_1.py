#!/usr/bin/env python
"""Qualification-only screen for a historical GPU fusion/precision witness.

This is deliberately a *case adapter*, not part of the generic locator.  It
records a declared contextual numerical contract at several backends without
reading any patch or choosing a candidate operation.
"""
from __future__ import annotations

import json
from pathlib import Path


def finite_one(value) -> bool:
    import torch
    return bool(torch.isfinite(value).all() and torch.equal(value, torch.ones_like(value)))


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("this qualification witness requires CUDA")
    torch.set_default_device("cuda")

    # The semantic contract is algebraic at these exact float32 inputs:
    # max_scaled and x * scale are the same eager expression, so the exponent
    # argument is zero and the result must be finite one.
    def model(x, scale):
        max_scaled = x * scale
        return torch.exp(max_scaled - x * scale)

    stages: dict[str, list[dict[str, object]]] = {}
    for stage_id, backend in (("eager", None), ("dynamo_eager", "eager"),
                              ("aot_eager", "aot_eager"), ("inductor", "inductor")):
        rows: list[dict[str, object]] = []
        for _ in range(2):
            torch._dynamo.reset()
            x = torch.tensor(1134139801600.0, dtype=torch.float32)
            scale = torch.tensor(0.180336877703666687, dtype=torch.float32)
            fn = model if backend is None else torch.compile(model, backend=backend, fullgraph=True)
            value = fn(x, scale)
            rows.append({
                "value": float(value.detach().cpu().item()),
                "finite": bool(torch.isfinite(value).all()),
                "contract_holds": finite_one(value),
            })
        stages[stage_id] = rows

    summary = {
        stage: {
            "repeatable": rows[0] == rows[1],
            "contract_holds": rows[0]["contract_holds"],
            "value": rows[0]["value"],
        }
        for stage, rows in stages.items()
    }
    repeatable = all(row["repeatable"] for row in summary.values())
    lower = bool(
        summary["eager"]["contract_holds"]
        and summary["dynamo_eager"]["contract_holds"]
        and summary["aot_eager"]["contract_holds"]
        and not summary["inductor"]["contract_holds"]
        and repeatable
    )
    output = Path("results/historical_candidate_screen/fma_context_122260_v0_1")
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "forkcert.historical-candidate-screen.v0.1",
        "case": "pytorch_issue_122260_fused_fma_context",
        "role": "candidate_qualification_only",
        "environment": {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "semantic_contract": "for the declared identical float32 products, exp((x*scale)-(x*scale)) is finite one",
        "stages": stages,
        "stage_summary": summary,
        "qualification": "QUALIFIED_LOWER_GPU_CANDIDATE" if lower else "REJECTED_ON_BOUND_RUNTIME",
        "not_a_claim": ["blind localization", "external patch validation", "root cause", "Triton source cause"],
    }
    (output / "screen.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"qualification": record["qualification"], "stage_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
