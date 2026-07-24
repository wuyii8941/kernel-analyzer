#!/usr/bin/env python
"""Screen PyTorch #115260 as a low-level generated-code candidate.

The contract is contextual equivalence: exposing an already-computed
intermediate as an additional return value must not alter the tan output.
This avoids treating eager as a mathematical oracle; eager is a null control
for the declared context transformation.  This remains qualification only,
not a blind certificate or a patch-localization claim.
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    import torch

    torch.manual_seed(115260)
    p0 = torch.tensor([1.0879], dtype=torch.float16)
    args = (
        torch.randn((17, 5, 1, 7), dtype=torch.float16) * 0.1 + 5,
        torch.randn((17, 5, 1, 7), dtype=torch.float16) * 0.1 + 5,
        torch.randn((17, 5, 11, 7), dtype=torch.float16) * 0.1 + 5,
        torch.randn((17, 5, 1, 7), dtype=torch.float16) * 0.1 + 5,
        torch.tensor(4.39, dtype=torch.float16),
    )

    def hidden(*values):
        cat = torch.cat((values[3], values[2], values[1], values[0]), dim=2)
        mul = torch.mul(cat, torch.max(values[4], p0))
        return torch.tan(mul)

    def exposed(*values):
        cat = torch.cat((values[3], values[2], values[1], values[0]), dim=2)
        mul = torch.mul(cat, torch.max(values[4], p0))
        return mul, torch.tan(mul)

    def observe(backend: str | None):
        rows = []
        for _ in range(2):
            torch._dynamo.reset()
            a = hidden if backend is None else torch.compile(hidden, backend=backend, fullgraph=True)
            b = exposed if backend is None else torch.compile(exposed, backend=backend, fullgraph=True)
            left = a(*[x.clone() for x in args])
            _, right = b(*[x.clone() for x in args])
            delta = (left - right).abs()
            rows.append({"max_abs": float(delta.max().item()), "equal": bool(torch.equal(left, right)),
                         "finite": bool(torch.isfinite(left).all().item() and torch.isfinite(right).all().item())})
        return rows

    stages = {"eager": observe(None)}
    for stage in ("eager", "aot_eager", "inductor"):
        stages["dynamo_eager" if stage == "eager" else stage] = observe(stage)
    def summary(rows):
        return {"repeatable": rows[0] == rows[1], "contextual_contract_holds": rows[0]["equal"],
                "max_abs": rows[0]["max_abs"], "finite": rows[0]["finite"]}
    summaries = {name: summary(rows) for name, rows in stages.items()}
    qualified = bool(
        summaries["eager"]["repeatable"] and summaries["eager"]["contextual_contract_holds"]
        and summaries["dynamo_eager"]["contextual_contract_holds"]
        and summaries["aot_eager"]["contextual_contract_holds"]
        and summaries["inductor"]["repeatable"] and not summaries["inductor"]["contextual_contract_holds"]
    )
    out = Path("results/historical_candidate_screen/tan_output_context_115260_v0_1")
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "forkcert.historical-candidate-screen.v0.1",
        "case": "pytorch_issue_115260_tan_output_context",
        "role": "candidate_qualification_only",
        "environment": {"torch": torch.__version__, "device": "cpu"},
        "semantic_contract": "adding an already-computed mul to the return tuple must not alter tan",
        "stages": stages, "stage_summary": summaries,
        "qualification": "QUALIFIED_FOR_BLIND_PACKAGING" if qualified else "REJECTED_ON_BOUND_RUNTIME",
        "not_a_claim": ["blind localization", "external patch validation", "unique source", "GPU/Triton behavior"],
    }
    (out / "screen.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"qualification": record["qualification"], "stage_summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
