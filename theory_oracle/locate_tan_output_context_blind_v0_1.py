#!/usr/bin/env python
"""Blind, pre-patch stage-to-FX localization for the #115260 candidate.

The case adapter supplies a program and declared *contextual equivalence*
contract.  It automatically derives candidate regions from the FX graph.  A
subset means: make those existing FX values additionally observable while
preserving the original final output.  The reducer repeatedly recompiles that
variant and tests whether the final output changes.  The generic core contains
no case, operator, or patch knowledge.

This script intentionally does not contain a fixed commit, patch URL, source
line, or proposed root cause.  It is to be run and hashed before any such
material is consulted.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from forkcert.localization_runtime import run_localization


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main() -> None:
    import torch
    import torch.fx

    torch.manual_seed(115260)
    p0 = torch.tensor([1.0879], dtype=torch.float16)
    args = (
        torch.randn((17, 5, 1, 7), dtype=torch.float16) * 0.1 + 5,
        torch.randn((17, 5, 1, 7), dtype=torch.float16) * 0.1 + 5,
        torch.randn((17, 5, 11, 7), dtype=torch.float16) * 0.1 + 5,
        torch.randn((17, 5, 1, 7), dtype=torch.float16) * 0.1 + 5,
        torch.tensor(4.39, dtype=torch.float16),
    )

    class Program(torch.nn.Module):
        def forward(self, *values):
            cat = torch.cat((values[3], values[2], values[1], values[0]), dim=2)
            mul = torch.mul(cat, torch.max(values[4], p0))
            return torch.tan(mul)

    base = torch.fx.symbolic_trace(Program())
    region_nodes = [node for node in base.graph.nodes if node.op not in ("placeholder", "output", "get_attr")]
    region_ids = tuple(node.name for node in region_nodes)
    if len(region_ids) < 2:
        raise RuntimeError("FX inventory unexpectedly has fewer than two natural regions")

    def exposed_variant(enabled: tuple[str, ...]):
        gm = copy.deepcopy(base)
        nodes = {node.name: node for node in gm.graph.nodes}
        output = next(node for node in gm.graph.nodes if node.op == "output")
        final = output.args[0]
        captures = tuple(nodes[name] for name in enabled)
        output.args = ((captures, final),)
        gm.graph.lint(); gm.recompile()
        return gm

    def execute_pair(stage: str, enabled: tuple[str, ...]):
        torch._dynamo.reset()
        hidden = Program()
        exposed = exposed_variant(enabled)
        if stage != "eager":
            hidden = torch.compile(hidden, backend=stage, fullgraph=True)
            exposed = torch.compile(exposed, backend=stage, fullgraph=True)
        baseline = hidden(*[x.clone() for x in args])
        captures, final = exposed(*[x.clone() for x in args])
        delta = (baseline - final).abs()
        return {
            "max_abs": float(delta.max().item()),
            "equal": bool(torch.equal(baseline, final)),
            "finite": bool(torch.isfinite(baseline).all().item() and torch.isfinite(final).all().item()),
            "capture_count": len(captures),
        }

    # A stage observation uses the complete automatically derived FX inventory;
    # it does not choose an operator in advance.
    stage_backend = {"eager": "eager", "dynamo_eager": "eager", "aot_eager": "aot_eager", "inductor": "inductor"}
    stage_rows: dict[str, dict[str, Any]] = {}
    for stage_id, backend in stage_backend.items():
        values = [execute_pair(backend, region_ids) for _ in range(2)]
        stage_rows[stage_id] = {
            "contract_holds": bool(values[0]["equal"] and values[1]["equal"]),
            "repeatable": values[0] == values[1],
            "measurements": values,
            "artifact_ids": (),
            "notes": ("contextual-equivalence: exposing FX values must not alter final output",),
        }

    reduction_queries: list[dict[str, Any]] = []
    query_cache: dict[tuple[str, ...], bool] = {}

    def preserves_inductor_symptom(enabled: tuple[str, ...]) -> bool:
        if enabled in query_cache:
            return query_cache[enabled]
        values = [execute_pair("inductor", enabled) for _ in range(2)]
        # Require both a violation and repeatability.  A nonrepeatable outcome
        # is not evidence for a candidate region.
        result = bool(not values[0]["equal"] and values[0] == values[1])
        reduction_queries.append({"enabled_regions": list(enabled), "measurements": values,
                                  "preserves_symptom": result})
        query_cache[enabled] = result
        return result

    class Adapter:
        def case_identity(self):
            return {"case_id": "pytorch_115260_tan_output_context", "role": "blind_pre_patch_historical_candidate",
                    "torch": torch.__version__, "input_seed": 115260}
        def semantic_contract(self):
            return {"kind": "contextual_equivalence", "endpoint": "final tan tensor",
                    "statement": "adding observability of existing FX values must not alter final output"}
        def stage_ids(self): return tuple(stage_rows)
        def run_stage(self, stage_id): return stage_rows[stage_id]
        def region_ids(self): return region_ids
        def preserves_symptom(self, enabled): return preserves_inductor_symptom(enabled)
        def provenance(self, candidates):
            node_map = {node.name: node for node in base.graph.nodes}
            return {"provenance_level": "FX",
                    "all_regions": [{"id": n.name, "op": n.op, "target": str(n.target)} for n in region_nodes],
                    "candidate_regions": [{"id": n, "op": node_map[n].op, "target": str(node_map[n].target)} for n in candidates],
                    "lower_level": "UNINSTANTIATED: no generated-code mapping is inferred from an FX candidate"}
        def evidence(self, candidates):
            return {"reducer_query_count": len(reduction_queries), "reducer_queries": reduction_queries,
                    "production": "UNINSTANTIATED: output observability changes compilation context; this is not same-input isolated-region replay",
                    "mediation": "UNINSTANTIATED: endpoint is the contextual-equivalence final output",
                    "intervention": "OBSERVABILITY_CONTEXT_INTERVENTION_ONLY",
                    "limitations": ["candidate is an FX context-exposure set, not a unique source",
                                    "no patch/fixed-revision information was used"]}

    result = run_localization(Adapter())
    certificate = result.certificate
    certificate["pre_reveal"] = {
        "patch_or_fixed_revision_accessed": False,
        "certificate_sha256": digest_json(certificate),
        "allowed_claim": "STAGE_FX_CANDIDATE_PRE_REVEAL" if result.stage_screen["failing_stages"] else "NO_OBSERVED_STAGE_FAILURE",
        "not_claimed": ["root cause", "source line", "generated kernel", "external patch agreement"],
    }
    out = Path("results/historical_blind/tan_output_context_115260_v0_1")
    out.mkdir(parents=True, exist_ok=True)
    (out / "pre_reveal_certificate.json").write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage_screen": result.stage_screen, "reduction": result.reduction,
                      "claim": certificate["pre_reveal"]["allowed_claim"],
                      "hash": certificate["pre_reveal"]["certificate_sha256"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
