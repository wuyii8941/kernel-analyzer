#!/usr/bin/env python3
"""Record one path A workload (docs/OPERATOR_ENTRY_PROTOCOL.md sections 5 and 6) with FlagGems 5.0.2 enabled.

    record_workload.py --model MODEL_DIR --config fp32|bf16 --out NEW_DIR      (W1: one eager training step)
    record_workload.py --trigger D3|D5|D18|D20|D8 --out NEW_DIR               (W2: one known-defect trigger)

T1: warm-up step, then one step without the recorder (launch sequence A) and one with it (sequence B, the recorder's
own copies excluded); run.json records whether the kernel-name sequences agree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import record as R  # noqa: E402

R.install_launch_hook()                 # before FlagGems' first launch (launchers are cached at first use)
import torch  # noqa: E402
import flag_gems  # noqa: E402
from instantiate import FLAGGEMS_EXCLUDE  # noqa: E402

CAP_BYTES = 200 * 2**20


def model_step(model_dir, config):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_dir)
    text_cfg = getattr(cfg, "text_config", None) or cfg
    for key in ("num_hidden_layers", "n_layer", "num_layers", "num_decoder_layers", "n_layers"):
        if isinstance(getattr(text_cfg, key, None), int):
            setattr(text_cfg, key, min(getattr(text_cfg, key), 2))
    dtype = torch.bfloat16 if config == "bf16" else torch.float32
    for cls in (AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM):
        try:
            model = cls.from_pretrained(model_dir, config=cfg, dtype=dtype)
            break
        except (ValueError, KeyError):
            continue
    else:
        raise SystemExit(f"no model class for {cfg.model_type}")
    model = model.cuda().train()
    tok = AutoTokenizer.from_pretrained(model_dir)
    text = Path("/data1/tzh/kernel-analyzer/1_experiments/three_way/data/wikitext103_validation.txt")
    if not text.exists():
        text = Path("/data1/tzh/kernel-analyzer/.cache/three_way/data/wikitext103_validation.txt")
    ids = tok(text.read_text()[:20000], return_tensors="pt").input_ids[:, :128].cuda()

    def step():
        model.zero_grad(set_to_none=True)
        torch.manual_seed(20261020)
        model(input_ids=ids, labels=ids).loss.backward()
    return step


def trigger_step(tid):
    import torch.nn.functional as F
    g = torch.Generator(device="cuda").manual_seed(20261020)
    if tid == "D3":
        inp = torch.randn(8, 16, device="cuda", generator=g)
        inp[0, 0] = float("nan")
        a = torch.randn(8, 4, device="cuda", generator=g)
        b = torch.randn(4, 16, device="cuda", generator=g)
        return lambda: torch.addmm(inp, a, b, beta=0)
    if tid == "D5":
        x = torch.randn(4, 64, device="cuda", generator=g)
        w = torch.randn(64, device="cuda", generator=g)
        return lambda: F.rms_norm(x, (64,), w, eps=None)
    if tid == "D18":
        x = torch.randn(2, 3, 8, 10, device="cuda", generator=g).transpose(2, 3)
        return lambda: F.interpolate(x, scale_factor=2, mode="nearest")
    if tid == "D20":
        x = torch.randn(16, 12, device="cuda", generator=g).t()
        return lambda: torch.argmax(x)
    if tid == "D8":
        x = torch.randn(1, 1, 4, 4, device="cuda", generator=g)

        def step():
            xx = x.detach().requires_grad_(True)
            y = F.avg_pool2d(xx, 3, 2, ceil_mode=True)
            y.backward(torch.ones_like(y))
        return step
    if tid == "SMOKE":                    # harness smoke test only (the development calls; not a registered workload)
        x = torch.randn(64, 32, device="cuda", generator=g)
        w = torch.randn(32, 16, device="cuda", generator=g)
        b = torch.randn(16, device="cuda", generator=g)

        def step():
            torch.exp(x)
            torch.addmm(b, x, w)
            torch.argmax(x.t(), dim=1)
            (x * 2).sum()
        return step
    raise SystemExit(f"unknown trigger {tid}")


def names(launches):
    return [n for n, _, _ in launches]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--config", default="fp32")
    ap.add_argument("--trigger")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    flag_gems.enable(unused=list(FLAGGEMS_EXCLUDE))
    step = trigger_step(a.trigger) if a.trigger else model_step(a.model, a.config)
    step()
    torch.cuda.synchronize()
    R.reset_launches()
    step()
    torch.cuda.synchronize()
    seq_a = names(R.all_launches())
    R.reset_launches()
    with R.recording() as rec:
        step()
        torch.cuda.synchronize()
    seq_b = names(R.all_launches())
    first_diff = next((i for i, (x, y) in enumerate(zip(seq_a, seq_b)) if x != y), None)
    if first_diff is None and len(seq_a) != len(seq_b):
        first_diff = min(len(seq_a), len(seq_b))
    unattributed = sum(1 for *_, idx in R.all_launches() if idx is None)
    # capture size limit: drop the values of keys above the cap (kept in the account by the runner)
    over_cap = []
    for k, (args, kwargs) in list(rec.values.items()):
        size = 0

        def add(t):
            nonlocal size
            size += t.untyped_storage().nbytes() if isinstance(t, torch.Tensor) else 0
            return t
        R._walk(args, add)
        R._walk(kwargs, add)
        if size > CAP_BYTES:
            del rec.values[k]
            over_cap.append(k)
    import importlib.metadata as md
    R.save(rec, a.out, extra={
        "model": a.model, "config": a.config, "trigger": a.trigger,
        "t1_launch_sequences_agree": first_diff is None, "t1_first_difference": first_diff,
        "launches_without_recorder": len(seq_a), "launches_with_recorder": len(seq_b),
        "launches_outside_recorded_calls": unattributed, "calls": len(rec.calls), "keys": len(rec.keys),
        "keys_over_capture_cap": over_cap, "flag_gems": md.version("flag_gems"), "torch": torch.__version__,
        "exclude": list(FLAGGEMS_EXCLUDE)})
    print(json.dumps({"calls": len(rec.calls), "keys": len(rec.keys), "t1": first_diff is None,
                      "unattributed": unattributed, "over_cap": len(over_cap)}))


if __name__ == "__main__":
    main()
