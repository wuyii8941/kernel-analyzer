#!/usr/bin/env python3
"""End-to-end training run with selected operators captured for the tool (liger env).

Qwen3-1.7B, full fine-tuning on wikitext-103: FP32 weights under bf16 autocast,
Liger kernels for RMSNorm, RoPE, SwiGLU and the fused linear cross-entropy,
torchao AdamW8bit (its update is compiled by Inductor into Triton kernels).
Batches are variable-length text samples padded to the longest one, so the
number of non-ignored tokens N changes from step to step.

At the declared capture steps, sampled launches of the chosen operators are
captured with operand copies (capture packages for the tool):

* liger_cross_entropy_kernel            every 8th chunk launch of the step
* (_block)_rms_norm_forward / _backward the first, a middle and the last launch
* _swiglu_forward / _backward_kernel    the first launch
* _triton_rope                          the first and the last launch
* torchao AdamW8bit (Inductor triton_*) occurrences 1-3 of each compiled kernel (first-layer parameters)

    python scripts/run_e2e_training_capture.py --out .cache/e2e --log results/reference_eval/e2e/training_log.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, save_launch  # noqa: E402

MODEL = Path("/data1/tzh/models/Qwen/Qwen3-1.7B")
WIKITEXT = Path("/data1/tzh/cache/huggingface/datasets/Salesforce___wikitext/wikitext-103-raw-v1/0.0.0/"
                "b08601e04326c79dfdd32d625aee71d232d685c3")


def samples(tokenizer, count, seed, min_len=64, max_len=512):
    """Variable-length samples of consecutive wikitext paragraphs."""

    import pyarrow.ipc as ipc

    path = str(WIKITEXT / "wikitext-train-00000-of-00002.arrow")
    try:
        table = ipc.open_file(path).read_all()
    except Exception:
        table = ipc.open_stream(path).read_all()
    texts = [t for t in table.column("text").to_pylist()[:400000] if len(t.strip()) > 0]
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        start = int(rng.integers(0, len(texts) - 64))
        target = int(rng.integers(min_len, max_len + 1))
        ids = []
        for t in texts[start:start + 64]:
            ids.extend(tokenizer(t).input_ids)
            if len(ids) >= target:
                break
        if len(ids) >= min_len:
            out.append(ids[:target])
    return out


class Sampler:
    """select(kernel_name, launch_index): per-kernel occurrence counters within one captured step."""

    def __init__(self):
        self.count = collections.Counter()
        self.totals = None  # occurrences per kernel in a full step (measured on the step before)

    def __call__(self, name, index):
        k = self.count[name]
        self.count[name] += 1
        total = (self.totals or {}).get(name, 0)
        if name == "liger_cross_entropy_kernel":
            return k % 8 == 0
        if name in ("_rms_norm_forward_kernel", "_rms_norm_backward_kernel", "_block_rms_norm_forward_kernel",
                    "_block_rms_norm_backward_kernel"):
            return k in (0, total // 2, total - 1)
        if name in ("_swiglu_forward_kernel", "_swiglu_backward_kernel", "_triton_rope"):
            return k in (0, total - 1) if name == "_triton_rope" else k == 0
        if name.startswith("triton_"):  # torchao AdamW8bit, compiled by Inductor
            return k in (1, 2, 3)  # early parameters of the first decoder layer (the embedding comes first)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--capture-steps", default="5,100,200,300")
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    import torch._inductor.config as inductor_config

    inductor_config.use_static_cuda_launcher = False  # every Triton launch goes through CompiledKernel.run
    TritonLaunchRecorder.install_hook()  # before the first launch: Inductor caches launchers
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    from torchao.optim import AdamW8bit
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scripts.resource_preflight import resource_report

    torch.manual_seed(args.seed)
    preflight = resource_report(2_031_739_904, 512, 4, 500)
    if not preflight["launch_allowed"]:
        raise SystemExit(f"preflight failed: {preflight['failures']}")
    apply_liger_kernel_to_qwen3(rope=True, rms_norm=True, swiglu=True, cross_entropy=False,
                                fused_linear_cross_entropy=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa",
                                                 local_files_only=True).cuda()
    model.config.use_cache = False
    model.train()
    opt = AdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    data = samples(tokenizer, args.steps * args.batch, args.seed)
    pad = tokenizer.pad_token_id
    capture_steps = {int(s) for s in args.capture_steps.split(",")}
    log = {"model": str(MODEL), "precision": "fp32 weights, bf16 autocast", "optimizer": "torchao AdamW8bit",
           "lr": args.lr, "batch": args.batch, "liger": "rope, rms_norm, swiglu, fused_linear_cross_entropy",
           "data": "wikitext-103 train shard 0, consecutive paragraphs, 64..512 tokens, right-padded",
           "seed": args.seed, "preflight": preflight, "steps": []}
    sampler = Sampler()
    totals_step = collections.Counter()
    t_start = time.time()
    for step in range(1, args.steps + 1):
        batch = data[(step - 1) * args.batch: step * args.batch]
        length = max(len(x) for x in batch)
        ids = torch.full((len(batch), length), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, x in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
        labels = ids.masked_fill(mask == 0, -100)
        ids, mask, labels = ids.cuda(), mask.cuda(), labels.cuda()
        n_non_ignore = int((labels[:, 1:] != -100).sum())
        recorder = None
        if step in capture_steps:
            sampler.count.clear()
            sampler.totals = dict(totals_step)
            recorder = TritonLaunchRecorder(select=sampler)
            recorder.__enter__()
        elif step + 1 in capture_steps:  # count launches per kernel without copying
            counter = TritonLaunchRecorder(select=lambda n, i: totals_step.update([n]) and False)
            counter.__enter__()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        seconds = time.time() - t0
        if step + 1 in capture_steps and step not in capture_steps:
            counter.__exit__(None, None, None)
        if recorder is not None:
            recorder.__exit__(None, None, None)
            target = args.out / f"step{step:04d}"
            kept = collections.Counter()
            for launch in recorder.launches:
                save_launch(launch, target / f"{launch.kernel_name[:60]}__{kept[launch.kernel_name]:03d}")
                kept[launch.kernel_name] += 1
            (target / "batch.json").write_text(json.dumps({"step": step, "n_non_ignore": n_non_ignore,
                                                           "lengths": [len(x) for x in batch]}) + "\n")
            totals_step.clear()
        log["steps"].append({"step": step, "loss": float(out.loss), "n_non_ignore": n_non_ignore,
                             "max_length": length, "seconds": round(seconds, 3),
                             "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                             "captured": dict(kept) if recorder is not None else None})
        if step % 10 == 0 or recorder is not None:
            print(json.dumps(log["steps"][-1]), flush=True)
    log["wall_seconds"] = round(time.time() - t_start, 1)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(log, indent=1) + "\n")


if __name__ == "__main__":
    main()
