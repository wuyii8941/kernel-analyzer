#!/usr/bin/env python3
"""Four-arm fine-tuning control: parameter storage (fp32 / bf16) x Adam state storage (fp32 / bf16).

All arms: the same model (gpt-neo-125m by default), the same data order, forward / backward under bf16 autocast,
gradients rounded to bf16 before the optimizer (as in pure-bf16 training), AdamW (beta2 = 0.999 by default).  Only
the storage of the parameters and of the states differs:

    A  P32 S32   reference
    B  P32 S16   second-moment drift only (states rounded to bf16 as torch's foreach does: v*b2 rounded, then
                 + (1 - b2) g^2 rounded; m likewise)
    C  P16 S32   stale weights only (parameter written back in bf16)
    D  P16 S16   both: the pure-bf16 path of torch.optim.AdamW (foreach); checked against torch on the first steps
    E  P32 S32   with the learning rate multiplied by the step-magnitude ratio measured in arm B (--lr-scale-from)

Logged: training loss, held-out loss on fixed wikitext-103 validation batches, the fraction of parameter updates
that are lost (P16 arms: the parameter did not change although the update was nonzero), and for arms with bf16
states the median v / v_fp32-shadow and the step-magnitude ratio (same gradients, same m).

    python scripts/bf16_adam_four_arm.py --arm A --out results/bf16_adam/four_arm_A.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

WIKITEXT = Path("/data1/tzh/cache/huggingface/datasets/Salesforce___wikitext/wikitext-103-raw-v1/0.0.0/")


def batches(tok, split, seq, batch, seed):
    import pyarrow as pa

    path = sorted(WIKITEXT.glob(f"*/wikitext-{split}*.arrow"))[0]
    with pa.memory_map(str(path)) as src:
        table = pa.ipc.open_stream(src).read_all()
    texts = [t for t in table.column("text").to_pylist() if t.strip()]
    rng = np.random.default_rng(seed)
    buf = []
    while True:
        for i in rng.permutation(len(texts)):
            buf.extend(tok(texts[i])["input_ids"])
            while len(buf) >= seq * batch:
                yield torch.tensor(buf[: seq * batch]).view(batch, seq)
                buf = buf[seq * batch:]


class StorageAdamW:
    """AdamW with chosen parameter / state storage; math in fp32, each stored tensor rounded on write.  bf16 states
    are rounded the way torch's foreach implementation rounds them (two in-place ops per moment)."""

    def __init__(self, params, lr, betas, eps, state_bf16, param_bf16, lr_scale=None):
        self.params = list(params)
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.state_bf16, self.param_bf16, self.lr_scale = state_bf16, param_bf16, lr_scale
        sd = torch.bfloat16 if state_bf16 else torch.float32
        self.m = [torch.zeros_like(p, dtype=sd) for p in self.params]
        self.v = [torch.zeros_like(p, dtype=sd) for p in self.params]
        self.shadow_v = [torch.zeros_like(p, dtype=torch.float32) for p in self.params] if state_bf16 else None
        self.t = 0

    @torch.no_grad()
    def step(self, grads):
        self.t += 1
        lr = self.lr * (self.lr_scale(self.t) if self.lr_scale else 1.0)
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        lost = total = 0
        for i, (p, g) in enumerate(zip(self.params, grads)):
            g = g.float()
            m, v = self.m[i], self.v[i]
            if self.state_bf16:  # torch foreach: m.lerp_(g, 1-b1) ; v.mul_(b2).addcmul_(g, g, value=1-b2), in bf16
                m.copy_((m.float() + (1 - self.b1) * (g - m.float())).to(torch.bfloat16))
                v.copy_((v.float() * self.b2).to(torch.bfloat16))
                v.copy_((v.float() + (1 - self.b2) * g * g).to(torch.bfloat16))
                self.shadow_v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            else:
                m.mul_(self.b1).add_(g, alpha=1 - self.b1)
                v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            upd = (m.float() / bc1) / ((v.float() / bc2).sqrt() + self.eps)
            new = p.float() - lr * upd
            if self.param_bf16:
                new16 = new.to(torch.bfloat16)
                nz = upd != 0
                lost += int(((new16 == p) & nz).sum())
                total += int(nz.sum())
                p.copy_(new16)
            else:
                p.copy_(new)
        return lost / max(total, 1)

    @torch.no_grad()
    def drift(self):
        if not self.state_bf16:
            return None
        bc2 = 1 - self.b2 ** self.t
        rs, d16, d32 = [], 0.0, 0.0
        for m, v, s in zip(self.m, self.v, self.shadow_v):
            v16, v32, mm = v.float().flatten(), s.flatten(), m.float().abs().flatten()
            keep = torch.nonzero(v32 > 0).flatten()[:20000]
            rs.append(v16[keep] / v32[keep])
            d16 += float((mm / ((v16 / bc2).sqrt() + self.eps)).sum())
            d32 += float((mm / ((v32 / bc2).sqrt() + self.eps)).sum())
        return float(torch.cat(rs).median()), d16 / d32


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple("ABCDE"), required=True)
    parser.add_argument("--model", default="/data1/tzh/models/EleutherAI/gpt-neo-125m")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--lr-scale-from", type=Path, default=None, help="arm B log (for arm E)")
    parser.add_argument("--check-torch", action="store_true", help="arm D: compare with torch.optim.AdamW (foreach)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    p16 = args.arm in "CD"
    s16 = args.arm in "BD"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if p16 else torch.float32).cuda()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    lr_scale = None
    if args.arm == "E":
        log_b = json.loads(args.lr_scale_from.read_text())["log"]
        pts = sorted((r["step"], r["step_magnitude_ratio"]) for r in log_b if r.get("step_magnitude_ratio"))
        xs, ys = np.array([0] + [s for s, _ in pts]), np.array([1.0] + [r for _, r in pts])
        lr_scale = lambda t: float(np.interp(t, xs, ys))  # noqa: E731
    opt = StorageAdamW(params, args.lr, (0.9, args.beta2), 1e-8, s16, p16, lr_scale)
    ref = None
    if args.check_torch and args.arm == "D":
        ref_params = [p.detach().clone().requires_grad_(True) for p in params]
        ref = (ref_params, torch.optim.AdamW(ref_params, lr=args.lr, betas=(0.9, args.beta2), eps=1e-8,
                                             weight_decay=0.0, foreach=True))
    train = batches(tok, "train", args.seq, args.batch, seed=0)
    val_iter = batches(tok, "validation", args.seq, args.batch, seed=1)
    val = [next(val_iter) for _ in range(16)]

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for x in val:
            x = x.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses.append(float(model(input_ids=x, labels=x).loss))
        model.train()
        return float(np.mean(losses))

    log, t0 = [], time.time()
    log.append({"step": 0, "val_loss": evaluate()})
    print(json.dumps(log[-1]), flush=True)
    for step in range(1, args.steps + 1):
        x = next(train).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=x).loss
        loss.backward()
        grads = [p.grad.to(torch.bfloat16) for p in params]
        if ref is not None and step <= 20:
            for rp, g in zip(ref[0], grads):
                rp.grad = g.clone()
            ref[1].step()
        lost = opt.step(grads)
        for p in params:
            p.grad = None
        if ref is not None and step == 20:
            same = all(torch.equal(a, b) for a, b in zip(params, ref[0]))
            st = [ref[1].state[rp] for rp in ref[0]]
            same_v = all(torch.equal(a, s["exp_avg_sq"]) for a, s in zip(opt.v, st))
            print(json.dumps({"torch_check_step": 20, "params_identical": same, "exp_avg_sq_identical": same_v}), flush=True)
            log.append({"torch_check_step": 20, "params_identical": same, "exp_avg_sq_identical": same_v})
            ref = None
        if step % args.eval_every == 0 or step in (10, 100):
            row = {"step": step, "train_loss": float(loss), "val_loss": evaluate(), "lost_update_frac": lost,
                   "seconds": round(time.time() - t0, 1)}
            dr = opt.drift()
            if dr:
                row["v_ratio_median"], row["step_magnitude_ratio"] = dr
            log.append(row)
            print(json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"arm": args.arm, "model": args.model, "lr": args.lr, "beta2": args.beta2,
                                    "steps": args.steps, "batch": args.batch, "seq": args.seq, "log": log}, indent=2) + "\n")


if __name__ == "__main__":
    main()
