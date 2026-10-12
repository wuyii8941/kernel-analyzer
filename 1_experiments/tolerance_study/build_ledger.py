#!/usr/bin/env python3
"""The machine-readable ledger of the tolerance study (protocol section 6): one row per
mechanism x implementation x configuration x input distribution x state x observation layer x reference x check, with
the result and the file it comes from.  Failures, budget exhaustion, reruns and post-hoc items are rows too.

    python build_ledger.py --out results/ledger.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
VERSIONS = {"torch": "2.10.0+cu128", "triton": "3.6.0", "transformers": "5.18.0 (latest_train env)",
            "torchao": "0.16.0", "timm": "1.0.30", "device": "NVIDIA RTX A6000 (sm_86)",
            "model": "Qwen/Qwen2.5-0.5B-Instruct", "data": "yahma/alpaca-cleaned"}
MECH = {"A": "low-precision storage drops increments (pure-bf16 AdamW parameters and states)",
        "B": "small parameter error, larger update error (RAdam rectification evaluated in fp32 on the compiled path)",
        "C": "matrix precision choice for fp32 inputs (TF32)"}
IMPL = {("A", "compiled"): "torch.compile(torch.optim.AdamW(foreach=True).step) -- Inductor Triton kernels",
        ("A", "fused"): "torch.optim.AdamW(fused=True) -- torchtune default, ATen fused kernel",
        ("B", "compiled"): "torch.compile(torch.optim.RAdam(foreach=True).step) -- Inductor Triton kernels",
        ("B", "eager"): "torch.optim.RAdam(foreach=True), eager -- ATen",
        ("C", "triton_default"): "Triton tutorial 03 matmul kernel, fp32 inputs, default input precision (tf32)",
        ("C", "triton_ieee"): "the same kernel with input_precision='ieee' (control)",
        ("C", "inductor_high"): "torch.compile(F.linear, mode='max-autotune-no-cudagraphs') under 'high' "
                                "(set inside the call)",
        ("C", "cublas_high"): "F.linear under 'high' (cuBLAS TF32; set inside the call)",
        ("C", "cublas_highest"): "F.linear under 'highest' (cuBLAS IEEE)"}
REF = {"A": "K_R (automatic reference, mode A)", "B": "K_R (automatic reference) and f (mode B)",
       "black-box": "f (exact specification, black box)"}


def rows_measure(group, path):
    s = json.loads(path.read_text())
    job = s["job"]
    impl = IMPL[(group, job["impl"])]
    stratum = job["stratum"]
    base = {"mechanism": group, "mechanism_text": MECH[group], "implementation": impl, "versions": VERSIONS,
            "state": stratum, "mode": job["mode"], "reference": REF.get(job["mode"]) if job["mode"] != "B" else REF["B"],
            "source": str(path.relative_to(HERE)), "attempt": "attempt1 (kept)" if "attempt1" in path.name else "final"}
    if group in ("A", "B"):
        base["configuration"] = ("torchtune qwen2_5/0.5B_full_single_device: pure bf16, AdamW(fused) lr 2e-5" if group == "A"
                                 else "fp32 parameters + bf16 autocast, RAdam default betas, lr 2e-5")
        base["input_distribution"] = (f"captured optimizer state before step {job['step']} of 32 independent fine-tuning "
                                      f"runs (seeds {'1000' if group == 'A' else '4000'}-..+31), 12 pre-declared tensors, "
                                      "pre-declared coordinates")
    else:
        base["configuration"] = "fp32 linear layer, real Qwen2.5-0.5B weight"
        base["input_distribution"] = (f"inputs of layer {job['layer']} {job['mat']} from fp32 forwards over held-out "
                                      "alpaca text, 32 units of 128 tokens")
    if s.get("status") != "ok":
        return [{**base, "observation_layer": "all", "check": "unified entry",
                 "result": {"status": s.get("status"), "reason": str(s.get("reason"))[:300]}}]
    out = []
    for name, o in s["outputs"].items():
        layer = "increment" if o.get("observation") else "output"
        r = {**base, "observation_layer": f"{name} {layer}" if layer == "increment" else name,
             "check": "bias query (statistics per declared rule class) and observation layer",
             "result": {"status": o.get("status"), "reference_scope": (o.get("reference") or {}).get("reference_scope"),
                        "complete_rate": (o.get("reference") or {}).get("complete_rate"),
                        "rules": {cls: {k: v["judgment"] for k, v in st["rules"].items()}
                                  for cls, st in (o.get("statistics") or {}).items()},
                        "observation": (o.get("observation") or {}).get("summary"),
                        "decomposition": (o.get("observation") or {}).get("decomposition"),
                        "semantic": o.get("semantic"), "scale": o.get("scale")}}
        out.append(r)
    return out


def rows_tolerance(group, stratum, path):
    t = json.loads(path.read_text())
    out = []
    for key, cell in t["summary"].items():
        parts = key.split("|")
        impl, ref = parts[0], parts[1]
        rest = parts[2:]
        out.append({"mechanism": group, "mechanism_text": MECH[group], "state": stratum, "versions": VERSIONS,
                    "implementation": impl, "reference": ref, "observation_layer": " ".join(rest) or "output",
                    "check": "question 1: element-wise tolerance checks (frozen)", "checks": t.get("checks"),
                    "result": cell, "source": str(path.relative_to(HERE))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for group in ("A", "B", "C"):
        for d in sorted((RES / group).glob("*")):
            for f in sorted(d.glob("*.json")):
                if f.name == "tolerance.json":
                    rows += rows_tolerance(group, d.name, f)
                else:
                    rows += rows_measure(group, f)
    for mech, name, path in (("A", "A-2/A-3 paired runs", RES / "A23/summary.json"),
                             ("B", "B-2 paired runs", RES / "B2/summary.json"),
                             ("C", "C-2 downstream", RES / "C2/summary.json"),
                             ("A,B", "post-hoc controls (added after the paired results)", RES / "posthoc/summary.json")):
        if path.exists():
            s = json.loads(path.read_text())
            rows.append({"mechanism": mech, "item": name, "versions": VERSIONS, "source": str(path.relative_to(HERE)),
                         "result": {k: s[k] for k in s if k not in ("units", "per_unit", "eval_per_seed")}})
    unify = RES / "evidence" / "unification_check.json"
    if unify.exists():
        rows.append({"mechanism": "A,B,C", "item": "rerun check after the code unification (deviation D-11)",
                     "versions": VERSIONS, "source": str(unify.relative_to(HERE)), "result": json.loads(unify.read_text())})
    upstream = HERE / "results" / "upstream_tests.json"
    if upstream.exists():
        rows.append({"mechanism": "A,B", "item": "question 1: PyTorch v2.10.0 upstream tests", "versions": VERSIONS,
                     "source": str(upstream.relative_to(HERE)), "result": json.loads(upstream.read_text())})
    Path(a.out).write_text(json.dumps({"schema": "tolerance-study-ledger-v1", "rows": rows}, indent=1,
                                      ensure_ascii=False) + "\n")
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
