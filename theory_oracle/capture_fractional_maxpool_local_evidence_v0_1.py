#!/usr/bin/env python
"""Same-input ATen-to-generated-code evidence for the #141538 witness.

The script intentionally records a *local injection* result only.  The
program has a single meaningful compute region and no fixed nontrivial suffix,
so it must not pretend to offer graph reduction, mediation, or a source-line
diagnosis.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from forkcert.operator_evidence import (
    EvidenceGates,
    allowed_claim_level,
    tensor_fingerprint,
    validate_evidence_report,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(141538)
    x = torch.randn(1, 1, 10, 10, device="cuda", dtype=torch.float32)
    samples = torch.rand(1, 1, 2, device="cuda", dtype=torch.float32)

    class Program(nn.Module):
        def forward(self, value, random_samples):
            return F.fractional_max_pool2d(
                value, kernel_size=(1, 1), output_ratio=(0.5, 0.5),
                _random_samples=random_samples,
            )

    program = Program().eval().cuda()
    exported = torch.export.export(program, (x.clone(), samples.clone()))
    aten_nodes = [
        {"id": node.name, "op": node.op, "target": str(node.target)}
        for node in exported.graph_module.graph.nodes
    ]

    eager = program(x.clone(), samples.clone())
    torch._dynamo.reset()
    compiled = torch.compile(program, backend="inductor", fullgraph=True)
    candidate = compiled(x.clone(), samples.clone())
    candidate_repeat = compiled(x.clone(), samples.clone())
    torch.cuda.synchronize()

    debug_root = Path(os.environ.get("TORCH_COMPILE_DEBUG_DIR", "torch_compile_debug"))
    cache_root_value = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    cache_root = Path(cache_root_value) if cache_root_value else None
    wrapper_set = set(debug_root.rglob("output_code.py"))
    if cache_root is not None:
        wrapper_set.update(
            path for path in cache_root.rglob("*.py")
            if "async_compile" in path.read_text(errors="replace")
        )
    wrappers = sorted(wrapper_set)
    wrapper_rows = []
    for wrapper in wrappers:
        text = wrapper.read_text(errors="replace")
        wrapper_rows.append({
            "path": str(wrapper), "sha256": sha256(wrapper),
            "kernel_symbols": sorted(set(re.findall(
                r"^([A-Za-z0-9_]+)\s*=\s*async_compile\.triton\(", text, re.MULTILINE
            ))),
            # Inductor emits both ``Source Nodes`` and ``Topologically Sorted
            # Source Nodes`` comments.  Treating only the former as provenance
            # silently drops the useful annotation in this case.
            "source_node_annotations": re.findall(
                r"# (?:Topologically Sorted )?Source Nodes: (.+)", text
            ),
            "original_aten_annotations": re.findall(r"Original ATen: (.+)", text),
        })
    source_node_ids = [row["id"] for row in aten_nodes if row["op"] == "call_function"]
    local_discrepancy = not bool(torch.equal(eager, candidate))
    provenance_complete = bool(wrapper_rows and any(row["kernel_symbols"] for row in wrapper_rows))
    gates = EvidenceGates(
        complete_witness=True,
        same_input_local_replay=True,
        local_discrepancy_reproducible=local_discrepancy and bool(torch.equal(candidate, candidate_repeat)),
        provenance_complete=provenance_complete,
        candidate_realization_preserved=False,
        intervention_executed=False,
        oracle_recomputed=False,
        non_target_context_invariant=False,
        lower_level_replay=False,
        first_bad_stage_isolated=False,
        null_controls_valid=True,
    )
    report = {
        "schema_version": "forkcert.fractional-maxpool-local-evidence.v0.1",
        "case_identity": {
            "case_id": "pytorch_fractional_maxpool_lowering",
            "role": "development_retrospective_case_not_blind_phase3_score",
            "torch": torch.__version__, "triton": __import__("triton").__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        },
        "region_inventory": [
            {"region_id": "functional_program", "module": "Program",
             "aten_call_function_ids": source_node_ids, "all_aten_nodes": aten_nodes}
        ],
        "local_replay": {
            "boundary_inputs": [tensor_fingerprint(x), tensor_fingerprint(samples)],
            "same_input": True,
            "eager_output": tensor_fingerprint(eager),
            "compiled_output": tensor_fingerprint(candidate),
            "compiled_repeat_exact": bool(torch.equal(candidate, candidate_repeat)),
            "production_observed": local_discrepancy,
            "max_abs": float((eager.float() - candidate.float()).abs().max().item()),
        },
        "provenance": {
            "exported_aten_nodes": aten_nodes,
            "debug_root": str(debug_root), "cache_root": str(cache_root) if cache_root else None,
            "generated_wrappers": wrapper_rows,
            "level": "ATEN_INVENTORY_TO_COMPILER_EMITTED_WRAPPER" if provenance_complete else "ATEN_ONLY",
        },
        "intervention": {
            "status": "UNINSTANTIATED",
            "reason": "single output region has no fixed nontrivial suffix; no-op/context-preserving repair has not yet been constructed",
        },
        "oracle": {
            "endpoint": "FractionalMaxPool output equality under fixed input and explicit random-sample boundary",
            "eager_compiled_equal": bool(torch.equal(eager, candidate)),
            "max_abs": float((eager.float() - candidate.float()).abs().max().item()),
        },
        "gates": gates.__dict__,
        "allowed_claim_level": allowed_claim_level(gates),
        "limitations": [
            "the case is a development/retrospective candidate because public history was consulted before this run",
            "the complete functional program is the local replay region; this is not a nontrivial region reduction",
            "no fixed-suffix mediation or context-preserving repair is instantiated",
            "backend observations do not establish a unique first-bad pass or source line",
        ],
    }
    errors = validate_evidence_report(report)
    report["verifier_errors"] = errors
    if errors:
        report["allowed_claim_level"] = "INVALID"
    output = Path("results/historical_blind/fractional_maxpool_141538_v0_1")
    output.mkdir(parents=True, exist_ok=True)
    (output / "local_evidence_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"allowed_claim_level": report["allowed_claim_level"],
                      "provenance_level": report["provenance"]["level"],
                      "oracle": report["oracle"], "verifier_errors": errors}, indent=2))


if __name__ == "__main__":
    main()
