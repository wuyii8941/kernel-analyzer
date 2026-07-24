#!/usr/bin/env python
"""Pre-patch locator run for the TorchDispatch semantic-contract witness."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from forkcert.localization_runtime import run_localization


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main() -> None:
    import torch
    import torch.fx
    from torch.utils._python_dispatch import TorchDispatchMode

    class RewriteAddToMul(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten.add.Tensor:
                func = torch.ops.aten.mul.Tensor
            return func(*(args or ()), **({} if kwargs is None else kwargs))

    def model(x): return x + x
    x = torch.tensor([3.0])
    expected = torch.tensor([9.0])
    gm = torch.fx.symbolic_trace(model)
    regions = tuple(node.name for node in gm.graph.nodes if node.op not in ("placeholder", "output"))

    def run(stage: str):
        torch._dynamo.reset()
        fn = model if stage == "eager" else torch.compile(model, backend=stage, fullgraph=True)
        with RewriteAddToMul():
            y = fn(x.clone())
        return {"contract_holds": bool(torch.equal(y, expected)), "value": float(y.item())}

    rows = {}
    for label, backend in (("eager", "eager"), ("dynamo_eager", "eager"), ("aot_eager", "aot_eager"), ("inductor", "inductor")):
        measurements = [run(backend) for _ in range(2)]
        rows[label] = {"contract_holds": measurements[0]["contract_holds"] and measurements[1]["contract_holds"],
                       "measurements": measurements, "repeatable": measurements[0] == measurements[1],
                       "notes": ("TorchDispatch add-to-mul contract",)}

    queries=[]
    def predicate(enabled):
        # A one-region FX graph cannot be reduced further.  Still execute the
        # endpoint rather than returning a predeclared source label.
        values=[run("inductor") for _ in range(2)]
        result=bool(enabled and not values[0]["contract_holds"] and values[0] == values[1])
        queries.append({"enabled_regions":list(enabled),"measurements":values,"preserves_symptom":result})
        return result

    class Adapter:
        def case_identity(self): return {"case_id":"pytorch_105929_torchdispatch", "role":"blind_pre_patch_historical_candidate", "torch":torch.__version__}
        def semantic_contract(self): return {"kind":"dispatch_semantics", "endpoint":"rewritten tensor value", "statement":"TorchDispatch rewrite must remain visible"}
        def stage_ids(self): return tuple(rows)
        def run_stage(self, stage_id): return rows[stage_id]
        def region_ids(self): return regions
        def preserves_symptom(self, enabled): return predicate(enabled)
        def provenance(self, candidates):
            table={node.name:node for node in gm.graph.nodes}
            return {"provenance_level":"FX", "all_regions":[{"id":n.name,"op":n.op,"target":str(n.target)} for n in gm.graph.nodes if n.name in regions],
                    "candidate_regions":[{"id":n,"op":table[n].op,"target":str(table[n].target)} for n in candidates],
                    "lower_level":"UNINSTANTIATED_PENDING_DEBUG_ARTIFACT_REPLAY"}
        def evidence(self, candidates): return {"reducer_queries":queries,"production":"UNINSTANTIATED: dispatch mode is an execution-context semantic contract", "mediation":"UNINSTANTIATED", "limitations":["single natural FX region limits reduction-strength evidence","no fixed revision or patch accessed"]}
    result=run_localization(Adapter())
    cert=result.certificate
    cert["pre_reveal"]={"patch_or_fixed_revision_accessed":False,"certificate_sha256":digest(cert),"allowed_claim":"STAGE_FX_CANDIDATE_PRE_REVEAL","not_claimed":["root cause","source line","kernel cause","patch agreement"]}
    out=Path("results/historical_blind/torchdispatch_105929_v0_1");out.mkdir(parents=True,exist_ok=True)
    (out/"pre_reveal_certificate.json").write_text(json.dumps(cert,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"stage_screen":result.stage_screen,"reduction":result.reduction,"hash":cert["pre_reveal"]["certificate_sha256"]},indent=2))


if __name__ == "__main__":main()
