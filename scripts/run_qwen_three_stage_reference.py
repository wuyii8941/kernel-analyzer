#!/usr/bin/env python3
"""Measure Qwen lm-head dX error at endpoint, gradient, and update stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "archive/round1_code/src")]

from scripts.generated_nontriton_fp32_observer import fp32_external_reference  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402

TARGET_LEFT = (256, 151936)
TARGET_RIGHT = (151936, 2048)


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def curve(vectors: list[torch.Tensor]) -> list[dict[str, float | int]]:
    total = torch.zeros_like(vectors[0], dtype=torch.float64)
    energy = 0.0
    out = []
    for i, v in enumerate(vectors, 1):
        x = v.double(); total.add_(x); energy += float(torch.dot(x.reshape(-1), x.reshape(-1)))
        if i in (2, 4, 8, 16, 32):
            den = math.sqrt(max(energy, 0.0)); out.append({"horizon": i, "resultant_l2": float(torch.linalg.vector_norm(total)), "path_rms_l2": den, "coherence_amplification": float(torch.linalg.vector_norm(total)) / max(den, 1e-30)})
    return out


class ErrorObserver:
    def __init__(self, modules):
        self.modules = modules; self.restores = []; self.calls = 0; self.vector = None; self.summary = None

    def __enter__(self):
        seen = set()
        for module in self.modules:
            namespace = getattr(module, "extern_kernels", None)
            if namespace is None or id(namespace) in seen: continue
            seen.add(id(namespace)); original = namespace.mm
            def wrapped(*args: Any, _original=original, **kwargs: Any):
                result = _original(*args, **kwargs)
                if tuple(args[0].shape) != TARGET_LEFT or tuple(args[1].shape) != TARGET_RIGHT:
                    return result
                actual = kwargs.get("out", result)
                before = actual.detach().float().clone()
                high = fp32_external_reference("mm", args, kwargs)
                delivered = high.to(actual.dtype)
                actual.copy_(delivered)
                delta = before - delivered.float()
                self.vector = delta.detach().cpu().reshape(-1).clone()
                self.summary = {"changed_coordinates": int(torch.count_nonzero(delta).item()), "l2": float(torch.linalg.vector_norm(delta.double()))}
                self.calls += 1
                return result
            namespace.mm = wrapped; self.restores.append((namespace, original))
        return self

    def __exit__(self, *unused):
        for namespace, original in self.restores: namespace.mm = original
        if self.calls != 1: raise RuntimeError(f"target Qwen MM executed {self.calls} times")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--release-dir", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("host GPU required")
    bank = json.loads((ROOT / "results/coverage/qwen_seq256_input_bank.json").read_text()); states = list(bank.get("states", bank.get("records")))
    if len(states) != 32: raise RuntimeError("Qwen reference trajectory must contain exactly 32 states")
    device = torch.device(args.device); configure_candidate_runtime(24000)
    model = load_model("qwen", Path("/data1/tzh/models/Qwen/Qwen3-1.7B"), device); model.eval()
    # qwen_seq256_r2 was frozen with the full-graph protocol; using a different
    # compile mode would make the wrapper release unverifiable.
    start = len(PyCodeCache.modules); step = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm = torch.tensor([states[0].get("token_ids", states[0].get("input_ids"))], dtype=torch.long, device=device); model.zero_grad(set_to_none=True); step(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:]); validate_release(wrapper_modules(modules), json.loads((args.release_dir / "capture.json").read_text()))
    parameter = dict(model.named_parameters())["model.norm.weight"]; reference = parameter.detach().float().cpu().clone()
    local=[]; gradient=[]; update=[]; rows=[]
    for i,state in enumerate(states):
        ids=torch.tensor([state.get("token_ids",state.get("input_ids"))],dtype=torch.long,device=device)
        with torch.no_grad(): parameter.copy_(reference.to(parameter.dtype))
        model.zero_grad(set_to_none=True); torch.manual_seed(24000+i); torch.cuda.manual_seed_all(24000+i); loss_c=step(ids); loss_c.backward(); torch.cuda.synchronize(device); gc=parameter.grad.detach().float().cpu().clone(); ld=tensor_digest(loss_c)
        with torch.no_grad(): parameter.copy_(reference.to(parameter.dtype))
        model.zero_grad(set_to_none=True); torch.manual_seed(24000+i); torch.cuda.manual_seed_all(24000+i); obs=ErrorObserver(modules)
        with obs: loss_r=step(ids); loss_r.backward()
        torch.cuda.synchronize(device)
        if tensor_digest(loss_r)!=ld or obs.vector is None: raise RuntimeError(f"Qwen repair/forward gate failed at state {i}")
        gr=parameter.grad.detach().float().cpu().clone(); e=obs.vector; g=gc-gr; u=-args.learning_rate*g; local.append(e); gradient.append(g); update.append(u); reference.add_((-args.learning_rate*gr).cpu())
        rows.append({"step":i+1,"state_id":str(state.get("state_id",state.get("sequence_id",i))),"local_l2":float(torch.linalg.vector_norm(e.double())),"gradient_l2":float(torch.linalg.vector_norm(g.double())),"effective_update_l2":float(torch.linalg.vector_norm(u.double())),"endpoint_changed_coordinates":obs.summary["changed_coordinates"]})
        print(json.dumps({"event":"QWEN_THREE_STAGE_STEP",**rows[-1]}),flush=True); del ids,loss_c,loss_r,gc,gr,e,g,u; torch.cuda.empty_cache()
    payload={"schema":"kernel-analyzer-qwen-three-stage-reference-v1","status":"COMPLETE_ORDERED_32_STATE_REFERENCE","case_id":"qwen_seq256_lmhead_dx","release_dir":str(args.release_dir),"state_count":32,"state_order":[str(x.get("state_id",x.get("sequence_id",i))) for i,x in enumerate(states)],"reference_trajectory":"repair-gradient SGD master; candidate and repair evaluated at identical pre-step norm state","learning_rate":args.learning_rate,"stages":{"operator_output_error":{"coherence_curve":curve(local)},"parameter_gradient_error":{"coherence_curve":curve(gradient)},"effective_update_error":{"coherence_curve":curve(update)}},"rows":rows,"claim_boundary":"One ordered reference trajectory and one norm carrier; not full-parameter training."}
    payload["result_sha256"]=sha(payload); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print(json.dumps({"status":payload["status"],"output":str(args.output)}))


if __name__ == "__main__": main()
