#!/usr/bin/env python
"""Qualification screen for a TorchDispatch semantic-capture historical case."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class RewriteAddToMul(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            if func is torch.ops.aten.add.Tensor:
                func = torch.ops.aten.mul.Tensor
            return func(*args, **kwargs)

    def model(x): return x + x
    x = torch.tensor([3.0])
    expected = torch.tensor([9.0])
    stages: dict[str, list[dict[str, object]]] = {}
    for stage_id, backend in (("eager", None), ("dynamo_eager", "eager"), ("aot_eager", "aot_eager"), ("inductor", "inductor")):
        rows=[]
        for _ in range(2):
            torch._dynamo.reset()
            run = model if backend is None else torch.compile(model, backend=backend, fullgraph=True)
            with RewriteAddToMul():
                value = run(x.clone())
            rows.append({"value": float(value.item()), "contract_holds": bool(torch.equal(value, expected))})
        stages[stage_id]=rows
    summary={stage:{"repeatable":rows[0]==rows[1],"contract_holds":rows[0]["contract_holds"],"value":rows[0]["value"]} for stage,rows in stages.items()}
    repeatable = all(row["repeatable"] for row in summary.values())
    higher_stopping = bool(summary["eager"]["contract_holds"] and not summary["dynamo_eager"]["contract_holds"] and repeatable)
    lower_stage = bool(summary["eager"]["contract_holds"] and summary["dynamo_eager"]["contract_holds"]
                       and summary["aot_eager"]["contract_holds"] and not summary["inductor"]["contract_holds"]
                       and repeatable)
    out=Path("results/historical_candidate_screen/torchdispatch_105929_v0_1"); out.mkdir(parents=True,exist_ok=True)
    record={"schema_version":"forkcert.historical-candidate-screen.v0.1","case":"pytorch_issue_105929_torchdispatch_capture",
            "role":"candidate_qualification_only","environment":{"torch":torch.__version__,"device":"cpu"},
            "semantic_contract":"TorchDispatchMode rewrite of aten.add to aten.mul must be observed by compiled execution",
            "stages":stages,"stage_summary":summary,
            "qualification":("QUALIFIED_HIGHER_STOPPING_CANDIDATE" if higher_stopping
                             else "QUALIFIED_LOWER_STAGE_CANDIDATE_ROLE_MISMATCH" if lower_stage
                             else "REJECTED_ON_BOUND_RUNTIME"),
            "not_a_claim":["blind localization","external patch validation","root cause"]}
    (out/"screen.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"qualification":record["qualification"],"stage_summary":summary},indent=2))

if __name__ == "__main__": main()
