#!/usr/bin/env python
"""Qualification screen for PyTorch #117019 on a bound historical runtime."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    import torch

    torch.manual_seed(117019)
    x = torch.rand((8, 6, 8, 6, 6, 1), dtype=torch.float64)

    def fn(v):
        return torch.diag_embed(v, dim1=-1, dim2=0, offset=1)

    reference = fn(x)
    stages: dict[str, list[dict[str, object]]] = {}
    for stage_id, backend in (("eager", None), ("dynamo_eager", "eager"), ("aot_eager", "aot_eager"), ("inductor", "inductor")):
        rows = []
        for _ in range(2):
            torch._dynamo.reset()
            run = fn if backend is None else torch.compile(fn, backend=backend, fullgraph=True)
            value = run(x.clone())
            delta = (value - reference).abs()
            rows.append({"equal": bool(torch.equal(value, reference)), "max_abs": float(delta.max().item()),
                         "shape": list(value.shape), "dtype": str(value.dtype)})
        stages[stage_id] = rows
    summary = {stage: {"repeatable": rows[0] == rows[1], "contract_holds": rows[0]["equal"],
                       "max_abs": rows[0]["max_abs"]} for stage, rows in stages.items()}
    qualified = bool(summary["eager"]["contract_holds"] and summary["dynamo_eager"]["contract_holds"]
                     and summary["aot_eager"]["contract_holds"] and not summary["inductor"]["contract_holds"]
                     and all(row["repeatable"] for row in summary.values()))
    out = Path("results/historical_candidate_screen/diag_embed_117019_v0_1")
    out.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": "forkcert.historical-candidate-screen.v0.1",
              "case": "pytorch_issue_117019_diag_embed_negative_dims", "role": "candidate_qualification_only",
              "environment": {"torch": torch.__version__, "device": "cpu"},
              "semantic_contract": "diag_embed must preserve its declared dim/offset placement exactly in float64",
              "stages": stages, "stage_summary": summary,
              "qualification": "QUALIFIED_FOR_BLIND_PACKAGING" if qualified else "REJECTED_ON_BOUND_RUNTIME",
              "not_a_claim": ["blind localization", "external patch validation", "root cause"]}
    (out / "screen.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"qualification": record["qualification"], "stage_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
