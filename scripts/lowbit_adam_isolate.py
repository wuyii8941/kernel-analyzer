#!/usr/bin/env python3
"""Isolation: which 4-bit state causes torchao AdamW4bit's fine-tuning gap?  AdamW in fp32 with only one moment
stored through torchao's own 4-bit quantizer (block 128; m: signed dynamic map, v: unsigned linear map without
zero), dequantized every step, everything else as torch AdamW.  Same model, data and schedule as
lowbit_adam_real.py.

    python scripts/lowbit_adam_isolate.py --quantize v --out results/lowbit_adam/gptneo_isolate_v.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bf16_adam_four_arm import batches  # noqa: E402


def qdq(x, signed, block=128):
    from torchao.optim.quant_utils import dequant_with_qmap, quantize_4bit_with_qmap, scale_tensor
    from torchao.optim.subclass_4bit import get_qmap_signed, get_qmap_unsigned

    if x.numel() < 4096 or x.numel() % block:
        return x  # torchao keeps such tensors in full precision
    qmap = torch.tensor(get_qmap_signed() if signed else get_qmap_unsigned(), device=x.device)
    xn, scale = scale_tensor(x.detach().float().reshape(-1), block)
    codes = quantize_4bit_with_qmap(xn, qmap)
    return dequant_with_qmap(codes, qmap, scale).view(x.shape)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", choices=("m", "v"), required=True)
    parser.add_argument("--model", default="/data1/tzh/models/EleutherAI/gpt-neo-125m")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).cuda()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    state = [(torch.zeros_like(p), torch.zeros_like(p)) for p in params]
    train = batches(tok, "train", 512, 8, seed=0)
    val_iter = batches(tok, "validation", 512, 8, seed=1)
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
    for t in range(1, args.steps + 1):
        x = next(train).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=x).loss
        loss.backward()
        with torch.no_grad():
            bc1, bc2 = 1 - 0.9 ** t, 1 - 0.999 ** t
            for (m, v), p in zip(state, params):
                g = p.grad.float()
                m.mul_(0.9).add_(g, alpha=0.1)
                v.mul_(0.999).addcmul_(g, g, value=0.001)
                if args.quantize == "m":
                    m.copy_(qdq(m, True))
                else:
                    v.copy_(qdq(v, False))
                p.add_(-(args.lr / bc1) * m / ((v / bc2).sqrt() + 1e-8))
                p.grad = None
        if t % 250 == 0:
            log.append({"step": t, "val_loss": evaluate(), "seconds": round(time.time() - t0, 1)})
            print(json.dumps(log[-1]), flush=True)
    args.out.write_text(json.dumps({"quantize": args.quantize, "lr": args.lr, "log": log}, indent=2) + "\n")


if __name__ == "__main__":
    main()
