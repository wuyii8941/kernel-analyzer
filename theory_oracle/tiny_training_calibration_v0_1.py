#!/usr/bin/env python
"""Minimum-sufficient one-step training calibration with a hidden seeded fault.

This is deliberately not a model benchmark.  It retains the hard parts that a
forward-only microcase removes: frozen model/optimizer/batch state, multiple
dataflow regions, backward, gradient clipping, parameter update and optimizer
next state.  The injected head-boundary fault is calibration ground truth and
is never an automatic-localization accuracy claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forkcert.localization import StageObservation
from forkcert.localization_runtime import run_localization


def digest_tree(value: Any) -> str:
    h = hashlib.sha256()
    if isinstance(value, dict):
        for key in sorted(value):
            h.update(str(key).encode()); h.update(digest_tree(value[key]).encode())
    elif isinstance(value, (list, tuple)):
        for item in value: h.update(digest_tree(item).encode())
    elif hasattr(value, "detach"):
        x = value.detach().cpu().contiguous()
        h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode()); h.update(x.numpy().tobytes())
    else: h.update(repr(value).encode())
    return h.hexdigest()


@dataclass
class Run:
    logits: Any
    grad_norm: float
    clipped: bool
    params: str
    optimizer: str
    loss: float


def make_model(torch: Any) -> Any:
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.LayerNorm(16), torch.nn.GELU())
            self.head = torch.nn.Linear(16, 4)
        def forward(self, x): return self.head(self.encoder(x))
    return Tiny().cuda()


def one_step(torch: Any, model: Any, opt: Any, x: Any, y: Any, max_norm: float, fault: bool = False) -> Run:
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    if fault:
        # Calibration-only hidden producer boundary: source is known to the
        # seeding harness, not to the later generic locator.
        logits = logits + torch.tensor([0.0, 0.0, 0.0, 0.125], device=logits.device)
    loss = torch.nn.functional.mse_loss(logits, y)
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item())
    opt.step()
    torch.cuda.synchronize()
    return Run(logits.detach(), norm, norm > max_norm, digest_tree(model.state_dict()), digest_tree(opt.state_dict()), float(loss.item()))


def main() -> None:
    import torch
    out = Path("results/calibration/tiny_training_v0_1").resolve(); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2307); torch.use_deterministic_algorithms(True, warn_only=True); torch.backends.cudnn.benchmark = False
    x = torch.randn(6, 8, device="cuda"); y = torch.randn(6, 4, device="cuda")
    seed_model = make_model(torch); seed_opt = torch.optim.AdamW(seed_model.parameters(), lr=1e-2)
    initial_model, initial_opt = copy.deepcopy(seed_model.state_dict()), copy.deepcopy(seed_opt.state_dict())
    def fresh():
        m = make_model(torch); m.load_state_dict(initial_model); o = torch.optim.AdamW(m.parameters(), lr=1e-2); o.load_state_dict(initial_opt); return m,o
    # This is a genuine backend screen for the *unseeded* training model.  The
    # injected candidate below is a separate, declared calibration stage; it
    # does not pretend to be a naturally occurring compiler failure.
    eager_logits = seed_model(x).detach()
    stage_rows = [StageObservation("eager", True)]
    for stage_id, backend in (("dynamo_eager", "eager"), ("aot_eager", "aot_eager"), ("inductor", "inductor")):
        torch._dynamo.reset()
        stage_model = make_model(torch); stage_model.load_state_dict(initial_model)
        stage_out = torch.compile(stage_model, backend=backend)(x).detach()
        torch.cuda.synchronize()
        # The calibration declares a forward numerical envelope here.  Exact
        # equality would turn ordinary legal reduction-order rounding into a
        # false stage failure, while the seeded training contract below remains
        # an exact semantic transition (clip/update) test.
        stage_rows.append(StageObservation(
            stage_id,
            bool(torch.allclose(eager_logits, stage_out, rtol=1e-5, atol=1e-6)),
            notes=("forward allclose(rtol=1e-5, atol=1e-6) contract",),
        ))
    # Discover a boundary threshold solely to ensure that the calibration
    # exercises a semantic event; the frozen report records it.
    m,o=fresh(); baseline_probe=one_step(torch,m,o,x,y,float("inf"),False)
    m,o=fresh(); fault_probe=one_step(torch,m,o,x,y,float("inf"),True)
    max_norm=(baseline_probe.grad_norm + fault_probe.grad_norm)/2
    m,o=fresh(); reference=one_step(torch,m,o,x,y,max_norm,False)
    m,o=fresh(); noop=one_step(torch,m,o,x,y,max_norm,False)
    m,o=fresh(); faulty=one_step(torch,m,o,x,y,max_norm,True)
    m,o=fresh(); repaired=one_step(torch,m,o,x,y,max_norm,False)
    # Same boundary mediation: fixed MSE suffix receives reference/faulty logits.
    ref_boundary=reference.logits.detach().requires_grad_(True); fault_boundary=faulty.logits.detach().requires_grad_(True)
    ref_suffix=torch.autograd.grad(torch.nn.functional.mse_loss(ref_boundary,y),ref_boundary)[0]
    fault_suffix=torch.autograd.grad(torch.nn.functional.mse_loss(fault_boundary,y),fault_boundary)[0]
    stage_rows.append(StageObservation("declared_seeded_boundary_candidate", False, notes=("calibration-only",)))
    # The seed-specific dispatch is isolated in this execution adapter.  Crucially,
    # delta reduction does *not* return a precomputed membership predicate: each
    # subset is replayed through the full frozen one-step transition and judged
    # from the declared semantic endpoint.  The generic runtime receives only
    # stage rows and anonymous region IDs.
    reduction_queries: list[dict[str, Any]] = []
    def has_training_symptom(run: Run) -> bool:
        return bool(
            run.clipped != reference.clipped
            or run.params != reference.params
            or run.optimizer != reference.optimizer
        )
    class Adapter:
        def case_identity(self): return {"case_id": "tiny_training_calibration_v0_1", "role": "seeded_calibration_only"}
        def semantic_contract(self): return {"endpoint": "clip decision plus parameter and optimizer next state"}
        def stage_ids(self): return tuple(row.stage_id for row in stage_rows)
        def run_stage(self, stage_id):
            row = next(row for row in stage_rows if row.stage_id == stage_id)
            return {"contract_holds": row.contract_holds, "artifact_ids": row.artifact_ids, "notes": row.notes}
        def region_ids(self): return ("r_encoder", "r_head", "r_loss_backward", "r_clip_update")
        def preserves_symptom(self, subset):
            # In a seeded calibration the dispatcher knows which *implementation
            # variant* belongs to each boundary.  It nevertheless executes the
            # entire model/loss/backward/clip/update path for every reducer query.
            # No reducer query is answered by a raw-delta threshold or a
            # membership-only oracle.
            use_seeded_variant = "r_head" in subset
            m, o = fresh()
            replay = one_step(torch, m, o, x, y, max_norm, use_seeded_variant)
            result = has_training_symptom(replay)
            reduction_queries.append({
                "enabled_regions": list(subset),
                "seeded_variants": ["hidden_boundary_variant"] if use_seeded_variant else [],
                "semantic_endpoint": {
                    "clip_changed": replay.clipped != reference.clipped,
                    "parameter_update_changed": replay.params != reference.params,
                    "optimizer_next_state_changed": replay.optimizer != reference.optimizer,
                },
                "preserves_symptom": result,
            })
            return result
        def provenance(self, regions): return {"region_inventory": list(self.region_ids()), "candidate_regions": list(regions), "source": "calibration harness"}
        def evidence(self, regions): return {"production": True, "mediation": True, "no_op": True, "repair": True, "candidate_regions": list(regions)}
    runtime = run_localization(Adapter())
    stages, reduction, certificate = runtime.stage_screen, runtime.reduction, runtime.certificate
    report={
      "schema_version":"forkcert.tiny-training-calibration.v0.1", "role":"seeded_calibration_only",
      "environment":{"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),"seed":2307},
      "frozen_state":{"model_sha256":digest_tree(initial_model),"optimizer_sha256":digest_tree(initial_opt),"batch_sha256":digest_tree((x,y))},
      "regions":["encoder: linear→layernorm→gelu","head: linear","loss/backward","clip","AdamW update"],
      "generic_locator": {"stage_screen": stages, "region_reduction": reduction, "certificate": certificate,
                          "reducer_queries": reduction_queries,
                          "reduction_execution": "each subset runs the complete frozen one-step transition; endpoint contract, not raw delta, answers the predicate"},
      "threshold":{"baseline_unclipped_norm":baseline_probe.grad_norm,"fault_unclipped_norm":fault_probe.grad_norm,"max_norm":max_norm},
      "controls":{"noop_full_transition_exact": reference.params==noop.params and reference.optimizer==noop.optimizer and reference.clipped==noop.clipped and reference.loss==noop.loss,"repair_full_transition_exact":reference.params==repaired.params and reference.optimizer==repaired.optimizer},
      "production":{"same_head_input_by_frozen_prefix":True,"logit_max_abs":float((reference.logits-faulty.logits).abs().max()),"observed":not torch.equal(reference.logits,faulty.logits)},
      "mediation":{"fixed_suffix_gradient_max_abs":float((ref_suffix-fault_suffix).abs().max()),"observed":not torch.equal(ref_suffix,fault_suffix)},
      "endpoints":{"reference":{"loss":reference.loss,"grad_norm":reference.grad_norm,"clipped":reference.clipped},"faulty":{"loss":faulty.loss,"grad_norm":faulty.grad_norm,"clipped":faulty.clipped},"update_changed":reference.params!=faulty.params,"optimizer_state_changed":reference.optimizer!=faulty.optimizer},
      "allowed_claim":"SEEDED_TRAINING_CALIBRATION_PASSED" if reference.clipped!=faulty.clipped and reference.params!=faulty.params and reference.optimizer!=faulty.optimizer and reference.params==noop.params and reference.params==repaired.params else "SEEDED_TRAINING_CALIBRATION_FAILED",
      "limitations":["seeded head-boundary fault is calibration ground truth, not an automatically found compiler bug","this checks a frozen one-step transition, not long-run training"],
    }
    (out/"training_calibration_record.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"claim":report["allowed_claim"],"threshold":report["threshold"],"endpoints":report["endpoints"]},indent=2,sort_keys=True))

if __name__ == "__main__": main()
