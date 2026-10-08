#!/usr/bin/env python3
"""Declared-convention registry (protocol v3 section 7): the G4 candidate registration
(results/essential/phase2b/candidates.json) plus the closure's float64 counterparts, with two added columns per candidate:
the contract it is held to (contract v2 row of its family, spec file) and the conventions it declares for the C-labelled
items (ATT-C1/C2, ROPE-C1, MOE-C1..C4, OPT-A1, BASE-A3) and the observed implementation conventions for undefined cases
(ATT-A1 from the 2b runs).  A candidate is compared only against the convention it declares; differences between
conventions go to column 2.  Output: results/closure/candidate_conventions.json.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/closure/candidate_conventions.json"

SPEC = {"matmul_linear": "spec_base_ops", "reductions": "spec_base_ops", "activations": "spec_base_ops",
        "gather_layout": "spec_base_ops", "embedding": "spec_embedding", "attention": "spec_attention",
        "packing": "spec_attention (packed_mask)", "rope": "spec_attention (rope_apply)", "normalization": "spec_normalization",
        "optimizers": "spec_optimizers", "schedulers": "spec_training_program", "clip_amp": "spec_training_program",
        "training_program": "spec_training_program / spec_accumulation", "checkpoint": "spec_training_program (prop)",
        "moe": "spec_moe"}

F64_ADDED = {   # closure: same-device float64 / float32 counterparts for precision invariance (env ka_main_f64)
    "attention": ["sdpa_math_cuda_float64", "manual_eager_float64", "sdpa_math_cpu_float32"],
    "optimizers": ["torch_for_loop_cpu_float64", "torch_foreach_cpu_float64", "torch_for_loop_cuda_float64",
                   "torch_foreach_cuda_float64"],
    "packing": ["sdpa_math_block_mask_float64", "flex_eager_block_mask_float64"],
    "rope": ["ref_torch_cuda_float64"], "moe": ["ref_loop_cpu_float32", "vectorised_scatter_cuda_float64"]}


def declared(fam, cid):
    """conventions the candidate declares (documentation of its library / the harness call it is made through)."""
    d = {}
    if fam in ("attention", "packing"):
        if cid.startswith("hf_"):
            d["ATT-C1"] = "upper-left for q_len != k_len (HF eager builds the causal mask as SDPA documents)"
        elif cid.startswith("xformers"):
            d["ATT-C1"] = "explicit mask from the harness (no causal flag): upper-left as built"
        else:
            d["ATT-C1"] = "upper-left (SDPA documentation; flex / manual receive the same explicit mask)"
        d["ATT-C2"] = "query head h -> kv head h // (H_q / H_kv) (enable_gqa / repeat_interleave)"
    if fam == "rope":
        d["ROPE-C1"] = ("rotate-half and interleaved, offset as given (in-repo reference)" if cid.startswith("ref_")
                        else "rotate-half only (library documents rotate-half); interleaved conditions: unsupported, not a deviation")
    if fam == "moe":
        d.update({"MOE-C1": "renormalised top-k softmax weights (Mixtral)",
                  "MOE-C2": "ties broken by lower expert index (stable sort); the spec accepts any valid set",
                  "MOE-C3": "drop beyond capacity in order of appearance (token order kept by the stable sort)",
                  "MOE-C4": "aux = E * sum_e f_e P_e with f_e the fraction of tokens whose top-1 is e (Switch); spec counts "
                            "dispatched experts -> declared difference, column 2"})
    if fam == "optimizers":
        if cid.startswith(("torch_", "nightly_torch_")):
            d["OPT-A1"] = "main reading (max over the uncorrected v, then bias-correct): torch 2.10 documentation"
        elif cid.startswith(("bnb_", "torchao_")):
            d["OPT-A1"] = "amsgrad not exercised (AdamW without amsgrad)"
        elif cid.startswith("hf_adafactor"):
            d["contract"] = "X: Adafactor not named by contract v2"
    if fam == "activations":
        d["BASE-A3"] = "gelu_erf conditions -> erf formula; gelu_tanh -> tanh formula (approximate='tanh'); swiglu/geglu as written"
    if fam == "embedding":
        d["EMB-A1"] = "max_norm renormalisation divides by (norm + 1e-7) (observed in the F run, results/closure/f_eval/embedding.json)" \
            if not cid.startswith(("nightly_", "inductor_")) or True else ""
    if fam == "schedulers":
        d["SCH-A2"] = "step counting from 0 at construction (torch / HF documentation)"
    if fam == "clip_amp":
        d["CLIP-A1"] = "coefficient max_norm / (total_norm + 1e-6) (documented)"
    return d


def att_a1_observed():
    p = ROOT / "results/essential/phase2b/attention/analysis.json"
    d = json.loads(p.read_text())
    return {cid: v.get("conventions") for cid, v in d["candidates"].items()}


def main():
    g4 = json.loads((ROOT / "results/essential/phase2b/candidates.json").read_text())
    contract = json.loads((ROOT / "results/closure/contract_classification.json").read_text())
    att = att_a1_observed()
    out = {"source": "G4 registration results/essential/phase2b/candidates.json + closure float64 counterparts",
           "rule": "compare each candidate only against the convention it declares; differences between conventions -> column 2",
           "families": {}}
    for fam, cands in g4["families"].items():
        rows = []
        for c in cands:
            r = dict(c)
            r["adopted_contract"] = f"contract v2 row '{fam}' ({SPEC.get(fam, 'no spec')}); class counts " \
                                    f"{contract['families'].get(fam, {}).get('class_counts')}"
            r["declared_conventions"] = declared(fam, c["id"])
            if fam == "attention" and c["id"] in att:
                r["observed_ATT-A1_convention"] = att[c["id"]]
            rows.append(r)
        for cid in F64_ADDED.get(fam, []):
            rows.append({"id": cid, "added_by": "closure (precision invariance pair)", "adopted_contract":
                         f"contract v2 row '{fam}' ({SPEC.get(fam)})", "declared_conventions": declared(fam, cid)})
        out["families"][fam] = rows
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print({f: len(v) for f, v in out["families"].items()})


if __name__ == "__main__":
    main()
