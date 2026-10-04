#!/usr/bin/env python3
"""Real-gradient check of torchao low-bit AdamW second moments, and a training comparison.

gpt-neo-125m (fp32 parameters, bf16 autocast forward) fine-tuned on wikitext-103 with one optimizer:
  fp32   torch.optim.AdamW
  8bit   torchao.optim.AdamW8bit
  4bit   torchao.optim.AdamW4bit
For the low-bit runs a shadow FP32 Adam state (m, v) is updated with exactly the same gradients, so the comparison
of the dequantized states with the shadow is free of trajectory divergence.  Logged: held-out loss, quantiles of
v_dequant / v_fp32 over all parameters (sampled), and the ratio of Adam step magnitudes |m_hat|/(sqrt(v_hat)+eps)
of the optimizer's own states over the shadow's.

    python scripts/lowbit_adam_real.py --opt 4bit --out results/lowbit_adam/gptneo_4bit.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bf16_adam_four_arm import batches  # noqa: E402


def dq(x):
    return x.dequantize().float() if hasattr(x, "dequantize") else x.float()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", choices=("fp32", "8bit", "4bit"), required=True)
    parser.add_argument("--model", default="/data1/tzh/models/EleutherAI/gpt-neo-125m")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).cuda()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    kw = dict(lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    if args.opt == "fp32":
        opt = torch.optim.AdamW(params, **kw)
    else:
        import torchao.optim as ao

        opt = (ao.AdamW8bit if args.opt == "8bit" else ao.AdamW4bit)(params, **kw)
    shadow = None if args.opt == "fp32" else [(torch.zeros_like(p), torch.zeros_like(p)) for p in params]
    train = batches(tok, "train", args.seq, args.batch, seed=0)
    val_iter = batches(tok, "validation", args.seq, args.batch, seed=1)
    val = [next(val_iter) for _ in range(16)]

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = []
        for x in val:
            x = x.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(float(model(input_ids=x, labels=x).loss))
        model.train()
        return float(np.mean(out))

    log, t0 = [{"step": 0, "val_loss": evaluate()}], time.time()
    for step in range(1, args.steps + 1):
        x = next(train).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=x).loss
        loss.backward()
        if shadow is not None:
            with torch.no_grad():
                for (m, v), p in zip(shadow, params):
                    g = p.grad.float()
                    m.mul_(0.9).add_(g, alpha=0.1)
                    v.mul_(0.999).addcmul_(g, g, value=0.001)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % args.eval_every == 0 or step in (10, 100):
            row = {"step": step, "train_loss": float(loss), "val_loss": evaluate(), "seconds": round(time.time() - t0, 1)}
            if shadow is not None:
                with torch.no_grad():
                    rs, d_q, d_s = [], 0.0, 0.0
                    bc1, bc2 = 1 - 0.9 ** step, 1 - 0.999 ** step
                    for (ms, vs), p in zip(shadow, params):
                        st = opt.state[p]
                        vq, mq = dq(st["exp_avg_sq"]).flatten(), dq(st["exp_avg"]).flatten()
                        vf, mf = vs.flatten(), ms.flatten()
                        keep = torch.nonzero(vf > 0).flatten()
                        if keep.numel() > 20000:
                            keep = keep[torch.randperm(keep.numel(), device=keep.device)[:20000]]
                        rs.append(vq[keep] / vf[keep])
                        d_q += float(((mq / bc1).abs() / ((vq / bc2).sqrt() + 1e-8)).sum())
                        d_s += float(((mf / bc1).abs() / ((vf / bc2).sqrt() + 1e-8)).sum())
                    r = torch.cat(rs)
                    q = torch.quantile(r[torch.randperm(r.numel(), device=r.device)[:200000]],
                                       torch.tensor([0.1, 0.5, 0.9], device=r.device))
                    row.update(v_ratio_p10=float(q[0]), v_ratio_median=float(q[1]), v_ratio_p90=float(q[2]),
                               step_magnitude_ratio=d_q / d_s)
            log.append(row)
            print(json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"opt": args.opt, "model": args.model, "lr": args.lr, "steps": args.steps,
                                    "log": log}, indent=2) + "\n")


if __name__ == "__main__":
    main()
