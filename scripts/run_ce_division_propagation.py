#!/usr/bin/env python3
"""Does the approximate division in Liger cross-entropy reach the parameter update? (round-2 item 4)

Qwen3-1.7B (FP32, body frozen) trains lm_head with AdamW on wikitext-103
windows.  Each of H independent histories (own seed for the data order)
trains T steps with the original kernel; at step T the real state (weights,
moments, step) and that step's batch are frozen, and four kernel variants
(exp in {tl.exp, libdevice.exp} x division in {approximate, correctly
rounded}) compute dlogits and dW on the same hidden states.  From the same
state each variant takes one AdamW step (real moments), one zero-moment AdamW
step and one SGD step.

Saved per history: sign fractions of the dlogits difference, scalar
projections on full coordinates (aligned with the reference update), and the
vectors on the declared rows D for the fixed-direction rule.  Runs in the
liger environment; statistics are computed by --stats.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

MODEL = Path("/data1/tzh/models/Qwen/Qwen3-1.7B")
WIKITEXT = Path("/data1/tzh/cache/huggingface/datasets/Salesforce___wikitext/wikitext-103-raw-v1/0.0.0/"
                "b08601e04326c79dfdd32d625aee71d232d685c3")
VARIANTS = ("original", "div_rn", "libdevice", "libdevice_div_rn")
LR, BETAS, EPS = 1e-4, (0.9, 0.95), 1e-8
ROW_SEED = 20261002
SEQ = 64


def windows(tokenizer, count, seed):
    import pyarrow.ipc as ipc

    path = str(WIKITEXT / "wikitext-train-00000-of-00002.arrow")
    try:
        table = ipc.open_file(path).read_all()
    except Exception:  # datasets caches arrow files in the streaming format
        table = ipc.open_stream(path).read_all()
    texts = [t for t in table.column("text").to_pylist()[:200000] if len(t) > 200]
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.choice(len(texts), size=count, replace=False):
        ids = tokenizer(texts[i], return_tensors="pt").input_ids[0]
        if ids.numel() < SEQ:
            continue
        start = int(rng.integers(0, ids.numel() - SEQ + 1))
        out.append(ids[start:start + SEQ])
    return out


def ce_gradient(fused, kernel, loss_mod, weight, hidden, labels):
    fused.liger_cross_entropy_kernel = kernel
    w = torch.nn.Parameter(weight.detach().clone())
    h = hidden.detach().clone().requires_grad_(True)  # Liger accumulates dW only when the input needs grad
    loss = loss_mod(w, h, labels)
    grad, _ = torch.autograd.grad(loss, (w, h))
    return grad.detach(), loss.detach()


def adamw_write(weight, grad, exp_avg, exp_avg_sq, step):
    w = torch.nn.Parameter(weight.detach().clone())
    opt = torch.optim.AdamW([w], lr=LR, betas=BETAS, eps=EPS, weight_decay=0.0, foreach=False, fused=False)
    if exp_avg is not None:
        opt.state[w] = {"step": torch.tensor(float(step)), "exp_avg": exp_avg.clone(), "exp_avg_sq": exp_avg_sq.clone()}
    w.grad = grad
    opt.step()
    return (w.detach() - weight).double()


def run(args):
    import liger_kernel.ops.fused_linear_cross_entropy as fused
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scripts.capture_exp_intervention import patched_kernel

    torch.backends.cuda.matmul.allow_tf32 = False
    workdir = args.out / "patched_sources"
    workdir.mkdir(parents=True, exist_ok=True)
    kernels = {v: patched_kernel(v, workdir) for v in VARIANTS}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager",
                                                 local_files_only=True).cuda().eval()
    model.config.use_cache = False
    weight0 = model.lm_head.weight.detach().clone()
    V, H = weight0.shape
    rows = np.sort(np.random.default_rng(ROW_SEED).choice(V, size=256, replace=False))
    row_idx = torch.as_tensor(rows, device="cuda")
    loss_mod = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean", accum_dtype=torch.float32)
    (args.out / "design.json").write_text(json.dumps({
        "histories": args.histories, "steps": args.steps, "seq": SEQ, "lr": LR, "betas": BETAS, "eps": EPS,
        "variants": VARIANTS, "rows": rows.tolist(), "row_seed": ROW_SEED,
        "data": "wikitext-103 train shard 0, windows of 64 tokens, history seed = 1000 + history"}) + "\n")
    for hist in range(args.start, args.histories):
        path = args.out / f"history{hist:03d}.npz"
        if path.exists():
            continue
        t0 = time.time()
        seqs = windows(tokenizer, args.steps + 8, 1000 + hist)[: args.steps + 1]
        w = weight0.clone()
        exp_avg = torch.zeros_like(w)
        exp_avg_sq = torch.zeros_like(w)
        # Train T steps with the original kernel (real trajectory).
        for step in range(1, args.steps + 1):
            ids = seqs[step - 1].unsqueeze(0).cuda()
            with torch.no_grad():
                hidden = model.model(input_ids=ids).last_hidden_state.reshape(-1, H)
            labels = torch.nn.functional.pad(ids, (0, 1), value=-100)[..., 1:].reshape(-1)
            grad, _ = ce_gradient(fused, kernels["original"], loss_mod, w, hidden, labels)
            wp = torch.nn.Parameter(w.clone())
            opt = torch.optim.AdamW([wp], lr=LR, betas=BETAS, eps=EPS, weight_decay=0.0, foreach=False, fused=False)
            if step > 1:
                opt.state[wp] = {"step": torch.tensor(float(step - 1)), "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
            wp.grad = grad
            opt.step()
            st = opt.state[wp]
            w, exp_avg, exp_avg_sq = wp.detach(), st["exp_avg"], st["exp_avg_sq"]
        # Frozen state at step T + 1 and its batch.
        ids = seqs[args.steps].unsqueeze(0).cuda()
        with torch.no_grad():
            hidden = model.model(input_ids=ids).last_hidden_state.reshape(-1, H)
        labels = torch.nn.functional.pad(ids, (0, 1), value=-100)[..., 1:].reshape(-1)
        grads, saved = {}, {}
        for v in VARIANTS:
            grads[v], _ = ce_gradient(fused, kernels[v], loss_mod, w, hidden, labels)
        ref = "div_rn"  # correctly rounded division, original exp
        scalars = {}

        def update(layer, v):
            if layer == "adam":
                return adamw_write(w, grads[v], exp_avg, exp_avg_sq, args.steps).float()
            if layer == "zero":
                return adamw_write(w, grads[v], None, None, 0).float()
            if layer == "sgd_write":  # the parameter difference actually written in FP32
                return (w - LR * grads[v]).float() - w
            return -LR * grads[v]  # "sgd": the formula update -lr * g, not written to the parameter

        def dot(a, b):
            step_ = 1 << 24
            return float(sum((a[i:i + step_].double() * b[i:i + step_].double()).sum()
                             for i in range(0, a.numel(), step_)))

        for layer in ("adam", "zero", "sgd", "sgd_write"):
            r = update(layer, ref).reshape(-1)
            rn = math.sqrt(dot(r, r))
            for v in VARIANTS:
                upd = update(layer, v).reshape(-1)
                saved[f"{layer}__{v}"] = upd.reshape(V, H)[row_idx].reshape(-1).double().cpu().numpy()
                u = upd - r if v != ref else torch.zeros_like(r)  # float32 updates: difference is exact in float64 below
                u64 = upd.double() - r.double() if v != ref else None
                scalars[f"{layer}_aligned_full_{v}"] = (float((u64 * r.double()).sum()) / rn) if (v != ref and rn > 0) else 0.0
                scalars[f"{layer}_norm_full_{v}"] = float(torch.linalg.vector_norm(u64)) if v != ref else 0.0
                del upd, u, u64
            scalars[f"{layer}_reference_norm_full"] = rn
            del r
            torch.cuda.empty_cache()
        g_ref = grads[ref]
        for v in VARIANTS:
            d = grads[v].double() - g_ref.double()
            scalars[f"grad_aligned_full_{v}"] = float((d * g_ref.double()).sum() / torch.linalg.vector_norm(g_ref.double()))
            scalars[f"grad_norm_full_{v}"] = float(torch.linalg.vector_norm(d))
            saved[f"grad__{v}"] = grads[v][row_idx].reshape(-1).double().cpu().numpy()
            del d
        scalars["grad_reference_norm_full"] = float(torch.linalg.vector_norm(g_ref))
        np.savez_compressed(path, **saved, scalars=json.dumps(scalars))
        print(json.dumps({"history": hist, "seconds": round(time.time() - t0, 1),
                          "adam_aligned_original": scalars["adam_aligned_full_original"],
                          "sgd_aligned_original": scalars["sgd_aligned_full_original"]}), flush=True)
        del grads
        torch.cuda.empty_cache()


LAYERS = ("grad", "adam", "zero", "sgd", "sgd_write")


def stats(args):
    sys.path.insert(0, str(ROOT / "src"))
    from kernel_analyzer.reference_eval.analysis import _holm, _summarize

    design = json.loads((args.out / "design.json").read_text())
    paths = sorted(args.out.glob("history*.npz"))
    n = len(paths)
    n_cal = n // 3
    data = [np.load(p) for p in paths]
    layers = [layer for layer in LAYERS if f"{layer}__original" in data[0]]  # older runs: no sgd_write
    sc = [json.loads(str(d["scalars"])) for d in data]
    results = []
    comparisons = {
        "division_effect (original - div_rn)": ("original", "div_rn"),
        "exp_effect (original - libdevice)": ("original", "libdevice"),
        "division_effect_under_libdevice (libdevice - libdevice_div_rn)": ("libdevice", "libdevice_div_rn"),
    }
    for layer in layers:
        for name, (a, b) in comparisons.items():
            vecs = np.stack([d[f"{layer}__{a}"] - d[f"{layer}__{b}"] for d in data])
            direction = vecs[:n_cal].mean(axis=0)
            if np.linalg.norm(direction) > 0:
                wdir = direction / np.linalg.norm(direction)
                proj = vecs[n_cal:] @ wdir
                r = _summarize(name, "fixed_direction_rows_D", proj, proj, 0.05)
            else:
                r = {"comparison": name, "rule": "fixed_direction_rows_D", "verdict": "UNRESOLVED_MEASUREMENT"}
            r["layer"] = layer
            results.append(r)
            # Aligned projections on full coordinates are stored relative to div_rn; differences give other pairs.
            al = np.array([s[f"{layer}_aligned_full_{a}"] - s[f"{layer}_aligned_full_{b}"] for s in sc])
            r2 = _summarize(name, "aligned_reference_update_full", al[n_cal:], al[n_cal:], 0.05)
            r2["layer"] = layer
            results.append(r2)
    # 2x2 interaction on the aligned full-coordinate projections.
    for layer in layers:
        inter = np.array([(s[f"{layer}_aligned_full_original"] - 0.0) - (s[f"{layer}_aligned_full_libdevice"]
                          - s[f"{layer}_aligned_full_libdevice_div_rn"]) for s in sc])
        r = _summarize("interaction (exp x division)", "aligned_reference_update_full", inter[n_cal:], inter[n_cal:], 0.05)
        r["layer"] = layer
        results.append(r)
    tested = [r for r in results if "p_value_two_sided_conservative" in r]
    for r, rej in zip(tested, _holm([r["p_value_two_sided_conservative"] for r in tested], 0.05)):
        r["holm_reject"] = bool(rej)
        r["final_verdict"] = r["verdict"] if rej else "NOT_CONFIRMED"
    norms = {layer: {v: float(np.median([s[f"{layer}_norm_full_{v}"] / max(s[f'{layer}_reference_norm_full'], 1e-300)
                                         for s in sc])) for v in VARIANTS}
             for layer in layers}
    report = {"schema": "kernel-analyzer-ce-division-propagation-v1", "design": design, "histories": n,
              "calibration": n_cal, "confirmation": n - n_cal, "relative_norm_of_difference_median": norms,
              "results": results,
              "notes": ["units are independent training histories (one frozen state each); not checkpoints of one run",
                        "reference implementation for differences: correctly rounded division, original exp (div_rn)",
                        "aligned projections use full lm_head coordinates; fixed directions use the declared rows D"]}
    (args.report).write_text(json.dumps(report, indent=2) + "\n")
    for r in results:
        print(f"{r['layer']:5s} {r['comparison'][:52]:52s} {r['rule'][:22]:22s} mean={r.get('mean_projection', float('nan')):+.3e} "
              f"ci={[round(x, 14) for x in r.get('t_interval', [])]} {r.get('final_verdict', r.get('verdict'))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--histories", type=int, default=48)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stats:
        stats(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
