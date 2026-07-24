#!/usr/bin/env python
"""Controlled replay/intervention for the #122260 emitted Triton wrapper.

The input wrapper must come from a provenance-capture artifact, so this script
never searches filenames or guesses a kernel.  It first checks baseline and
no-op replay, then changes exactly one compiler-emitted expression.  The
result is deliberately an intervention-dependent kernel-expression result,
not a root-cause conclusion.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from forkcert.operator_evidence import (
    EvidenceGates,
    allowed_claim_level,
    compare_non_target_context,
    tensor_fingerprint,
    validate_evidence_report,
)


TARGET = "tmp5 = tmp4 - tmp4"
REPAIR = "tmp5 = tl.zeros_like(tmp4)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generated wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke(torch: Any, module: ModuleType, x: Any, scale: Any) -> Any:
    result = module.call([x.clone(), scale.clone()])[0]
    torch.cuda.synchronize()
    return result


def delta(torch: Any, left: Any, right: Any) -> dict[str, Any]:
    diff = (left.detach().float() - right.detach().float()).abs()
    return {
        "equal": bool(torch.equal(left, right)),
        "max_abs": float(diff.max().item()),
        "finite_left": bool(torch.isfinite(left).all()),
        "finite_right": bool(torch.isfinite(right).all()),
    }


def context(text: str) -> dict[str, Any]:
    """Context invariant: replace only the declared target before hashing."""
    if text.count(TARGET) == 1:
        normalized = text.replace(TARGET, "<DECLARED_TARGET>")
    elif text.count(REPAIR) == 1:
        normalized = text.replace(REPAIR, "<DECLARED_TARGET>")
    else:
        raise RuntimeError("generated wrapper does not contain exactly one declared target")
    return {"artifacts": [{"target_id": "fma_expression", "non_target_sha256": hashlib.sha256(normalized.encode()).hexdigest()}],
            "graph_count": 1, "graphs": ["single_emitted_wrapper"],
            "shape_layout_contracts": ["scalar-fp32-inputs-output"],
            "compiler_config_digest": "captured-wrapper-replay", "autotuning": {"status": "UNOBSERVED"}}


def main() -> None:
    import torch

    provenance_path = Path("results/historical_blind/fma_context_122260_v0_1/post_certificate_provenance.json")
    provenance = json.loads(provenance_path.read_text())
    source = Path(provenance["generated_artifacts"][0]["path"])
    original_text = source.read_text()
    if original_text.count(TARGET) != 1:
        raise RuntimeError(f"expected one target expression, found {original_text.count(TARGET)}")

    out = Path("results/historical_blind/fma_context_122260_v0_1/generated_kernel_intervention")
    out.mkdir(parents=True, exist_ok=True)
    original_path = out / "generated_original.py"
    noop_path = out / "generated_noop.py"
    repaired_path = out / "generated_repaired.py"
    original_path.write_text(original_text)
    noop_path.write_text(original_text)
    repaired_text = original_text.replace(TARGET, REPAIR)
    repaired_path.write_text(repaired_text)

    x = torch.tensor(1134139801600.0, device="cuda", dtype=torch.float32)
    scale = torch.tensor(0.180336877703666687, device="cuda", dtype=torch.float32)
    reference = torch.exp((x * scale) - (x * scale))
    original = invoke(torch, load(original_path, "fma_original"), x, scale)
    noop = invoke(torch, load(noop_path, "fma_noop"), x, scale)
    repaired = invoke(torch, load(repaired_path, "fma_repaired"), x, scale)
    original_repeat = invoke(torch, load(original_path, "fma_original_repeat"), x, scale)

    input_fps = [tensor_fingerprint(x), tensor_fingerprint(scale)]
    same_input = input_fps == [tensor_fingerprint(x), tensor_fingerprint(scale)]
    non_target = compare_non_target_context(context(original_text), context(repaired_text), ignored_target_ids=("fma_expression",))
    # This case's intervention target is the *entire* emitted wrapper.  After
    # excluding it, the remaining artifact inventory is empty.  Exact equality
    # of an empty non-target context is a useful audit fact, but it is not the
    # non-target-context invariance required to upgrade an attribution to an
    # operator-level effect.
    non_target_context_instantiated = bool(non_target["baseline"]["artifacts"])
    no_op_exact = bool(torch.equal(original, noop))
    repair_matches_reference = bool(torch.equal(repaired, reference))
    baseline_matches_compiled = bool(torch.equal(original, torch.tensor(float("inf"), device="cuda")))
    local_discrepancy = not bool(torch.equal(original, reference))
    gates = EvidenceGates(
        complete_witness=baseline_matches_compiled,
        same_input_local_replay=same_input,
        local_discrepancy_reproducible=local_discrepancy and bool(torch.equal(original, original_repeat)),
        provenance_complete=bool(provenance["generated_artifacts"][0]["triton_symbols"]),
        candidate_realization_preserved=no_op_exact,
        intervention_executed=True,
        oracle_recomputed=True,
        non_target_context_invariant=bool(non_target["exact"] and non_target_context_instantiated),
        lower_level_replay=True,
        first_bad_stage_isolated=False,
        null_controls_valid=no_op_exact and repair_matches_reference,
    )
    report = {
        "schema_version": "forkcert.fma-generated-kernel-intervention.v0.1",
        "case_identity": {"case_id": "pytorch_fma_context_gpu", "role": "mechanism_case_not_blind_patch_score",
                          "torch": torch.__version__, "cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(0)},
        "region_inventory": [{"region_id": "triton_poi_fused_exp_mul_sub_0",
                              "generated_wrapper": str(source), "source_sha256": sha256(source),
                              "fx_source_nodes": ["mul", "mul_1", "sub", "exp"]}],
        "local_replay": {"input_fingerprints": input_fps, "same_input": same_input,
                         "reference": tensor_fingerprint(reference), "compiled_wrapper": tensor_fingerprint(original),
                         "repeat_exact": bool(torch.equal(original, original_repeat)),
                         "production_observed": local_discrepancy,
                         "delta": delta(torch, reference, original)},
        "provenance": {"capture_artifact": str(provenance_path), "generated_wrapper": str(source),
                       "kernel_symbols": provenance["generated_artifacts"][0]["triton_symbols"],
                       "claim": "compiler-emitted source-node annotations and invoked kernel symbol"},
        "intervention": {"type": "direct_generated_expression_repair", "target_before": TARGET,
                         "target_after": REPAIR, "original_sha256": sha256(original_path),
                         "noop_sha256": sha256(noop_path), "repaired_sha256": sha256(repaired_path),
                         "no_op_exact": no_op_exact, "non_target_context": non_target,
                         "non_target_context_gate_instantiated": non_target_context_instantiated,
                         "repair_matches_reference": repair_matches_reference,
                         "before_vs_after": delta(torch, original, repaired)},
        "oracle": {"endpoint": "scalar output must be finite one", "reference_value": float(reference.item()),
                   "baseline_value": float(original.item()), "noop_value": float(noop.item()),
                   "repaired_value": float(repaired.item()), "baseline_contract_holds": bool(torch.equal(original, reference)),
                   "repaired_contract_holds": repair_matches_reference},
        "gates": gates.__dict__,
        "allowed_claim_level": allowed_claim_level(gates),
        "limitations": [
            "the candidate is the whole emitted fused wrapper, so this is not a nontrivial FX region reduction or non-target-context gate",
            "the case was selected after reading public issue discussion and is not blind external-patch scoring",
            "backend observation does not establish a unique first-bad pass",
            "the repair demonstrates an intervention effect, not a unique root cause or an upstream source line",
        ],
    }
    errors = validate_evidence_report(report)
    report["verifier_errors"] = errors
    if errors:
        report["allowed_claim_level"] = "INVALID"
    (out / "intervention_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"allowed_claim_level": report["allowed_claim_level"], "oracle": report["oracle"],
                      "verifier_errors": errors}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
