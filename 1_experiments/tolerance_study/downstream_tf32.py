#!/usr/bin/env python3
"""Group C-2 (protocol appendix C; executed because C-1 found a nonzero alignment): how the TF32 shrink of one matrix
multiplication propagates through a whole forward and backward pass, and whether Adam's scale invariance removes a
uniform gradient shrink.

Units: 8 held-out batches (2 sequences each, the workload's held-out split, examples disjoint from C-1's: offset 192
of C-1's fixed permutation).  Variants, all fp32 parameters and activations, no autocast:
  eager_highest   eager, 'highest' (cuBLAS IEEE)
  eager_high      eager, 'high' (cuBLAS TF32)
  compiled_high   torch.compile(model, mode="max-autotune-no-cudagraphs") under 'high' (Triton TF32 templates where
                  autotuning selects them, cuBLAS TF32 elsewhere)
  compiled_highest the same compile under 'highest' (Triton IEEE templates / cuBLAS IEEE)
Reference: the same model in float64 (eager).  Per unit and variant: loss difference, logits relative RMS and alignment
<e, r>/<r, r>, and per parameter group (attention / MLP matrices, norms) the gradient relative RMS and alignment; for the
update, one AdamW step (torchtune hyper-parameters) from a zero state -- its direction cosine and scale against the
reference update.  The frame quantities (relative RMS, alignment, cosine, scale) are the tool's
(``observation.output_unit``), not a copy.

    python downstream_tf32.py --out results/C2/summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

from kernel_analyzer.reference_eval.observation import output_unit

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def batches(tok, n_units, hf_home):
    data = json.loads(Path(glob.glob(os.path.join(hf_home, "hub", "datasets--yahma--alpaca-cleaned", "snapshots", "*",
                                                  "alpaca_data_cleaned.json"))[0]).read_text())
    order = np.random.default_rng(1).permutation(len(data))
    held = [data[i] for i in order[-256:]]
    pick = np.random.default_rng(2026).permutation(len(held))
    out = []
    for u in range(n_units):
        texts = []
        for i in pick[192 + 2 * u:192 + 2 * u + 2]:
            e = held[i]
            texts.append(e["instruction"] + "\n" + e.get("input", "") + "\n" + e["output"])
        enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        out.append((enc.input_ids, enc.attention_mask))
    return out


def groups(model):
    g = {"attention": [], "mlp": [], "norms": [], "embedding": []}
    for n, p in model.named_parameters():
        if "self_attn" in n:
            g["attention"].append(n)
        elif "mlp" in n:
            g["mlp"].append(n)
        elif "norm" in n:
            g["norms"].append(n)
        else:
            g["embedding"].append(n)
    return g


def fwd_bwd(model, ids, att, fn=None):
    model.zero_grad(set_to_none=True)
    f = fn if fn is not None else model
    logits = f(input_ids=ids, attention_mask=att).logits
    lab = ids.clone()
    lab[att == 0] = -100
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                                             lab[:, 1:].reshape(-1), ignore_index=-100)
    loss.backward()
    # gradients and logits in host memory (8 units of float64 gradients do not fit on one GPU)
    grads = {n: p.grad.detach().double().cpu() for n, p in model.named_parameters() if p.grad is not None}
    return float(loss), logits.detach().double().cpu(), grads


def adam_update(grads, names):
    """One AdamW step (lr 2e-5, betas 0.9 / 0.999, eps 1e-8) from a zero state, as the exact formula in float64:
    -lr * g / (|g| + eps) (bias-corrected m / sqrt(v) of one step) -- the parameter's own decay term is common to all
    variants and left out."""
    return torch.cat([(-2e-5 * grads[n] / (grads[n].abs() + 1e-8)).reshape(-1) for n in names])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", type=int, default=8)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_home = os.environ["HF_HOME"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = batches(tok, a.units, hf_home)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float64).cuda().train()
    refs = []
    for ids, att in data:
        refs.append(fwd_bwd(ref_model, ids.cuda(), att.cuda()))
    del ref_model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().train()
    grp = groups(model)
    compiled = torch.compile(model, mode="max-autotune-no-cudagraphs")
    variants = {"eager_highest": ("highest", None), "eager_high": ("high", None),
                "compiled_high": ("high", compiled), "compiled_highest": ("highest", compiled)}
    res = {"units": a.units, "variants": {}}
    for vname, (prec, fn) in variants.items():
        torch.set_float32_matmul_precision(prec)
        rows = []
        for (ids, att), (l_ref, lg_ref, g_ref) in zip(data, refs):
            loss, lg, g = fwd_bwd(model, ids.cuda(), att.cuda(), fn)
            fl = output_unit(lg.numpy(), lg_ref.numpy())
            row = {"loss_diff": loss - l_ref, "logits_relative_rms": fl["relative_rms"],
                   "logits_alignment": fl["alignment"], "gradients": {}}
            for gname, names in grp.items():
                names = [n for n in names if n in g and n in g_ref]
                if not names:
                    continue
                fg = output_unit(torch.cat([g[n].reshape(-1) for n in names]).numpy(),
                                 torch.cat([g_ref[n].reshape(-1) for n in names]).numpy())
                fu = output_unit(adam_update(g, names).numpy(), adam_update(g_ref, names).numpy())
                row["gradients"][gname] = {"relative_rms": fg["relative_rms"], "alignment": fg["alignment"],
                                           "adam_update_cosine": fu["cosine"], "adam_update_scale": fu["scale_ratio"]}
            rows.append(row)
        agg = {"loss_diff": _d([r["loss_diff"] for r in rows]),
               "logits_relative_rms": _d([r["logits_relative_rms"] for r in rows]),
               "logits_alignment": _d([r["logits_alignment"] for r in rows]), "gradients": {}}
        for gname in rows[0]["gradients"]:
            agg["gradients"][gname] = {k: _d([r["gradients"][gname][k] for r in rows])
                                       for k in rows[0]["gradients"][gname]}
        res["variants"][vname] = {"precision": prec, "compiled": fn is not None, "summary": agg, "per_unit": rows}
        print(vname, json.dumps({k: v for k, v in agg.items() if k != "gradients"}), flush=True)
    torch.set_float32_matmul_precision("highest")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


def _d(v):
    v = np.asarray(v, np.float64)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else None, "min": float(v.min()),
            "max": float(v.max()), "n": int(v.size)}


if __name__ == "__main__":
    main()
