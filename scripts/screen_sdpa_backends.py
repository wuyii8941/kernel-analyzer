#!/usr/bin/env python3
"""Screen: torch.nn.functional.scaled_dot_product_attention backends against float64 math semantics.

For each backend (MATH, EFFICIENT_ATTENTION, FLASH_ATTENTION, CUDNN_ATTENTION) that accepts a configuration, the
output and the q/k/v gradients are compared with the documented semantics evaluated in float64 on the same inputs
(softmax(q k^T * scale + mask) v; bool mask False = -inf; is_causal = top-left lower triangular; GQA = repeated
k/v heads; fully masked rows give NaN in the math definition and are compared as classes).  The error is reported
relative to a same-dtype baseline (the MATH backend in the same dtype), so precision effects cancel and semantic
outliers stand out.

    python scripts/screen_sdpa_backends.py --out results/screen/sdpa_backends.json
"""

import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def reference(q, k, v, mask, causal, scale, gqa):
    q, k, v = q.double(), k.double(), v.double()
    if gqa:
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    s = q @ k.transpose(-1, -2) * (scale if scale is not None else q.shape[-1] ** -0.5)
    L, S = q.shape[-2], k.shape[-2]
    if causal:
        s = s.masked_fill(~torch.ones(L, S, dtype=torch.bool, device=s.device).tril(), float("-inf"))
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else s + mask.double()
    return torch.softmax(s, -1) @ v


def run_case(cfg, backend, dtype, seed=0):
    B, H, Hkv, L, S, D, Dv = cfg["B"], cfg["H"], cfg["Hkv"], cfg["L"], cfg["S"], cfg["D"], cfg["Dv"]
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, H, L, D, device="cuda", generator=g)
    k = torch.randn(B, Hkv, S, D, device="cuda", generator=g)
    v = torch.randn(B, Hkv, S, Dv, device="cuda", generator=g)
    if cfg.get("layout") == "projected":  # (B, L, H, D) memory, transposed view, as after a q_proj reshape
        q = q.transpose(1, 2).contiguous().transpose(1, 2)
        k = k.transpose(1, 2).contiguous().transpose(1, 2)
        v = v.transpose(1, 2).contiguous().transpose(1, 2)
    mask = None
    if cfg["mask"] == "bool_broadcast":  # (B, 1, 1, S) padding mask expanded, HF style
        valid = torch.arange(S, device="cuda")[None, :] < torch.tensor([S, S - 7][:B] + [S] * max(0, B - 2), device="cuda")[:, None]
        mask = valid[:, None, None, :].expand(B, H, L, S)
    elif cfg["mask"] == "float_rows_masked":
        mask = torch.zeros(B, 1, L, S, device="cuda")
        mask[:, :, ::5, :] = float("-inf")  # fully masked query rows
        mask[:, :, 1::5, ::3] = float("-inf")
    elif cfg["mask"] == "bool_causal_padded":  # causal + left padding, HF generation style
        pad = torch.arange(S, device="cuda")[None, :] >= torch.tensor([0, 5][:B] + [0] * max(0, B - 2), device="cuda")[:, None]
        causal = torch.ones(L, S, dtype=torch.bool, device="cuda").tril(diagonal=S - L)
        mask = (causal[None] & pad[:, None, :])[:, None].expand(B, H, L, S)
    causal = cfg["mask"] == "causal"
    scale = cfg.get("scale")
    gqa = Hkv != H
    gq = torch.randn(B, H, L, Dv, device="cuda", generator=g)
    qq, kk, vv = (t.detach().to(dtype).requires_grad_(True) for t in (q, k, v))
    mm = mask if mask is None or mask.dtype == torch.bool else mask.to(dtype)
    try:
        with sdpa_kernel([backend]):
            o = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mm, is_causal=causal, scale=scale, enable_gqa=gqa)
            o.backward(gq.to(dtype))
    except Exception as exc:  # noqa: BLE001
        return {"status": "unsupported", "why": str(exc).splitlines()[0][:120]}
    qd, kd, vd = (t.detach().double().requires_grad_(True) for t in (qq, kk, vv))
    ref = reference(qd, kd, vd, mask, causal, scale, gqa)
    fin = torch.isfinite(ref)
    if fin.all():
        ref.backward(gq.double())
    res = {"status": "ok"}
    o64 = o.detach().double()
    res["out_nan_class_mismatch"] = int((torch.isnan(o64) != torch.isnan(ref)).sum())
    f = fin & torch.isfinite(o64)
    res["out_rel"] = float((o64[f] - ref.detach()[f]).norm() / ref.detach()[f].norm().clamp_min(1e-30))
    if fin.all():
        for name, a, b in (("dq", qq.grad, qd.grad), ("dk", kk.grad, kd.grad), ("dv", vv.grad, vd.grad)):
            res[f"{name}_rel"] = float((a.double() - b).norm() / b.norm().clamp_min(1e-30))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    base = dict(B=2, H=8, Hkv=8, L=128, S=128, D=64, Dv=64, mask="none")
    variants = []
    for D in (32, 40, 64, 72, 80, 96, 112, 128, 160, 192, 256):
        variants.append({**base, "D": D, "Dv": D, "name": f"d{D}"})
    for mask in ("causal", "bool_broadcast", "float_rows_masked", "bool_causal_padded"):
        for D in (64, 80, 128):
            variants.append({**base, "D": D, "Dv": D, "mask": mask, "name": f"{mask}_d{D}"})
    variants += [
        {**base, "Hkv": 2, "name": "gqa4"}, {**base, "Hkv": 2, "mask": "causal", "name": "gqa4_causal"},
        {**base, "L": 37, "S": 300, "name": "L37_S300"}, {**base, "L": 300, "S": 37, "name": "L300_S37"},
        {**base, "L": 37, "S": 300, "mask": "causal", "name": "L37_S300_causal"},
        {**base, "L": 1, "S": 513, "name": "decode_L1"}, {**base, "L": 1, "S": 513, "mask": "causal", "name": "decode_L1_causal"},
        {**base, "Dv": 32, "name": "dv32"}, {**base, "scale": 0.3, "name": "scale0.3"},
        {**base, "layout": "projected", "name": "projected_layout"},
        {**base, "layout": "projected", "mask": "bool_broadcast", "D": 80, "Dv": 80, "name": "projected_bool_d80"},
        {**base, "L": 1000, "S": 1000, "mask": "causal", "name": "L1000_causal"},
        {**base, "L": 129, "S": 129, "mask": "causal", "D": 96, "Dv": 96, "name": "L129_causal_d96"},
    ]
    backends = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]
    rows = []
    for cfg, dtype in itertools.product(variants, (torch.float32, torch.float16, torch.bfloat16)):
        per = {}
        for be in backends:
            per[be.name] = run_case(cfg, be, dtype)
        baseline = per["MATH"]
        line = []
        for be, r in per.items():
            if r["status"] != "ok":
                continue
            flags = []
            for key in ("out_rel", "dq_rel", "dk_rel", "dv_rel"):
                if key in r and key in baseline and baseline["status"] == "ok":
                    ratio = r[key] / max(baseline[key], 1e-12)
                    r[key + "_vs_math"] = ratio
                    if r[key] > 10 * baseline[key] and r[key] > 1e-3:
                        flags.append(f"{key}={r[key]:.1e} ({ratio:.0f}x math)")
            if r.get("out_nan_class_mismatch"):
                flags.append(f"NaN-class mismatch {r['out_nan_class_mismatch']}")
            if flags:
                line.append(f"{be}: " + ", ".join(flags))
        rows.append({"cfg": cfg, "dtype": str(dtype), "results": per})
        status = " | ".join(line) if line else "ok"
        sup = ",".join(b for b, r in per.items() if r["status"] == "ok")
        print(f"{cfg['name']:24s} {str(dtype)[6:]:9s} [{sup}] {status}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
