#!/usr/bin/env python
"""Qualification screen for an external historical PyTorch candidate.

This is deliberately *not* a blind-localization certificate.  It establishes
only whether a public historical reproducer still yields a stable, stage-
separated numerical violation on the bound old runtime.  A failing candidate
may later be packaged without fixed-revision/patch data for Phase 3.
"""
from __future__ import annotations

import json
from pathlib import Path


def scalar(tensor):
    return {
        "value": float(tensor.detach().cpu().item()),
        "isfinite": bool(tensor.detach().isfinite().item()),
        "dtype": str(tensor.dtype),
    }


def main() -> None:
    import torch

    out = Path("results/historical_candidate_screen/polygamma_147450_v0_1")
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    x = torch.tensor([-1.0], dtype=torch.float32, device=device)

    def fn(v):
        return torch.special.polygamma(1, v)

    eager_runs = [scalar(fn(x.clone())) for _ in range(2)]
    stages: dict[str, list[dict[str, object]]] = {"eager": eager_runs}
    for stage, backend in (
        ("dynamo_eager", "eager"),
        ("aot_eager", "aot_eager"),
        ("inductor", "inductor"),
    ):
        values = []
        for _ in range(2):
            torch._dynamo.reset()
            compiled = torch.compile(fn, backend=backend)
            values.append(scalar(compiled(x.clone())))
            torch.cuda.synchronize()
        stages[stage] = values

    def stable(values):
        return values[0] == values[1]

    expected = eager_runs[0]
    stage_summary = {}
    for stage, values in stages.items():
        stage_summary[stage] = {
            "repeatable": stable(values),
            "matches_eager_value": values[0]["value"] == expected["value"],
            "matches_eager_finiteness": values[0]["isfinite"] == expected["isfinite"],
        }
    candidate = bool(
        stage_summary["eager"]["repeatable"]
        and stage_summary["dynamo_eager"]["matches_eager_value"]
        and stage_summary["aot_eager"]["matches_eager_value"]
        and stage_summary["inductor"]["repeatable"]
        and not stage_summary["inductor"]["matches_eager_finiteness"]
    )
    record = {
        "schema_version": "forkcert.historical-candidate-screen.v0.1",
        "case": "pytorch_issue_147450_polygamma_n1",
        "role": "candidate_qualification_only",
        "not_a_claim": [
            "blind localization", "external patch validation", "root cause",
            "eager mathematical truth",
        ],
        "environment": {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "device": device,
        },
        "input": {"n": 1, "x": -1.0, "dtype": "float32"},
        "stages": stages,
        "stage_summary": stage_summary,
        "qualification": "QUALIFIED_FOR_BLIND_PACKAGING" if candidate else "REJECTED_ON_BOUND_RUNTIME",
        "next_action": (
            "build an opaque failing-revision package; do not expose patch/fixed revision to locator"
            if candidate else "do not promote this candidate; select another external historical case"
        ),
    }
    (out / "screen.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"qualification": record["qualification"], "stage_summary": stage_summary}, indent=2))


if __name__ == "__main__":
    main()
