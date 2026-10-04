#!/usr/bin/env python3
"""Real-gradient check of the bf16 AdamW second-moment drift.

A model held in bf16 is fine-tuned on wikitext-103 with torch.optim.AdamW (states follow the parameter dtype, so
exp_avg / exp_avg_sq are bf16).  Alongside, a shadow FP32 second moment is updated with exactly the same gradients
(v = b2 v + (1 - b2) g^2 in FP32), so the comparison is free of trajectory divergence.  Logged every few steps over
all parameters: the median and mean of v_bf16 / v_fp32, and the resulting ratio of the Adam step magnitude
|m| / (sqrt(v_hat) + eps) (both with the same bf16 m, isolating the second moment).

    python scripts/bf16_adam_state_real.py --model /data1/tzh/models/EleutherAI/gpt-neo-125m --steps 3000 --out results/bf16_adam/gptneo.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

WIKITEXT = Path("/data1/tzh/cache/huggingface/datasets/Salesforce___wikitext/wikitext-103-raw-v1/0.0.0/")


def token_stream(tok, seq, batch, seed=0):
    import pyarrow as pa

    path = next(WIKITEXT.glob("*/wikitext-train-00000-of-00002.arrow"))
    with pa.memory_map(str(path)) as src:
        table = pa.ipc.open_stream(src).read_all()
    texts = [t for t in table.column("text").to_pylist() if t.strip()]
    rng = np.random.default_rng(seed)
    buf = []
    while True:
        for i in rng.permutation(len(texts)):
            buf.extend(tok(texts[i])["input_ids"])
            while len(buf) >= seq * batch:
                x = torch.tensor(buf[: seq * batch]).view(batch, seq)
                buf = buf[seq * batch:]
                yield x


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--impl", choices=("foreach", "fused"), default="foreach")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
    if args.checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    kw = {"foreach": True} if args.impl == "foreach" else {"fused": True}
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, args.beta2), eps=1e-8, weight_decay=0.0, **kw)
    shadow = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    data = token_stream(tok, args.seq, args.batch)
    log, t0 = [], time.time()
    checks = sorted({10, 30, 100, 300, 1000} | set(range(500, args.steps + 1, 500)))
    for step in range(1, args.steps + 1):
        x = next(data).cuda()
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        with torch.no_grad():
            for s, p in zip(shadow, params):
                g = p.grad.float()
                s.mul_(args.beta2).addcmul_(g, g, value=1 - args.beta2)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step in checks:
            with torch.no_grad():
                ratios, dirs = [], []
                bc2 = 1 - args.beta2 ** step
                for s, p in zip(shadow, params):
                    st = opt.state[p]
                    v16 = st["exp_avg_sq"].float().flatten()
                    v32 = s.flatten()
                    keep = v32 > 0
                    idx = torch.nonzero(keep).flatten()
                    if idx.numel() > 20000:
                        idx = idx[torch.randperm(idx.numel(), device=idx.device)[:20000]]
                    r = v16[idx] / v32[idx]
                    ratios.append(r)
                    m = st["exp_avg"].float().flatten()[idx].abs()
                    d16 = m / ((v16[idx] / bc2).sqrt() + 1e-8)
                    d32 = m / ((v32[idx] / bc2).sqrt() + 1e-8)
                    dirs.append(torch.stack([d16.sum(), d32.sum()]))
                r = torch.cat(ratios)
                d = torch.stack(dirs).sum(0)
                row = {"step": step, "loss": float(loss), "v_ratio_median": float(r.median()),
                       "v_ratio_mean": float(r.mean()), "step_magnitude_ratio": float(d[0] / d[1]),
                       "seconds": round(time.time() - t0, 1)}
                log.append(row)
                print(json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model, "steps": args.steps, "batch": args.batch, "seq": args.seq,
                                    "lr": args.lr, "beta2": args.beta2, "impl": args.impl, "log": log}, indent=2) + "\n")


if __name__ == "__main__":
    main()
