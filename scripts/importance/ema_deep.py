#!/usr/bin/env python3
"""Deep case C9 (docs/directed_search_protocol_20261007.md section 9): weight EMA of a pure-bf16 model.

One pure-bf16 training run (configuration P, decay 0.999) maintains three EMAs of the same parameter sequence:
  real_api_bf16  torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999)): the averaged
                 copy inherits the model's bf16 dtype (the stock API rejects an fp32 copy of a bf16 model)
  reference_fp32 a hand-written fp32 EMA buffer
  fix_fp32       AveragedModel of an fp32 copy with a per-parameter avg_fn that lerps in fp32 (the deployable change)
and reports the validation loss of each next to the trained model's.

    python scripts/importance/ema_deep.py --seed 0     # -> results/directed_search/ema_deep_s0.json
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import small_lm as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DECAY = 0.999


def evaluate_with(model, params, data, c):
    saved = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for p, q in zip(model.parameters(), params):
            p.copy_(q.to(p.dtype))
    v = S.evaluate(model, data, c)
    with torch.no_grad():
        for p, s in zip(model.parameters(), saved):
            p.copy_(s)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / "results/directed_search" / f"ema_deep_s{a.seed}.json"
    if out.exists():
        return
    c = S.Config(seed=a.seed, precision="bf16", optimizer="adamw_bf16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    S.setup_precision(c)
    model, opt = S.build(c)
    data = S.Data(c, "cuda")
    # the RoPE tables are filled lazily at the first forward; create them now so the averaged copies carry the same
    # buffers (AveragedModel.update_parameters zips model and copy buffers strictly)
    model.cos, model.sin = S.rope_tables(c.seq, c.d // c.heads, c.rope_theta, "cuda")
    real = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(DECAY))
    fix = AveragedModel(copy.deepcopy(model).float(), avg_fn=lambda e, p, n: e.lerp(p.float(), 1 - DECAY))
    ref = None  # starts where AveragedModel starts: a copy at the first update (after step 0)
    t0 = time.time()
    for step in range(c.steps):
        S.train_step(model, opt, data, step, c)
        real.update_parameters(model)
        fix.update_parameters(model)
        with torch.no_grad():
            if ref is None:
                ref = [p.detach().clone().float() for p in model.parameters()]
            else:
                for e, p in zip(ref, model.parameters()):
                    e.lerp_(p.detach().float(), 1 - DECAY)
    res = {"seed": a.seed, "config": "pure bf16 (P), EMA decay 0.999", "seconds": round(time.time() - t0, 1),
           "val_loss_model": S.evaluate(model, data, c),
           "val_loss_real_api_bf16": evaluate_with(model, list(real.module.parameters()), data, c),
           "val_loss_reference_fp32": evaluate_with(model, ref, data, c),
           "val_loss_fix_fp32": evaluate_with(model, list(fix.module.parameters()), data, c),
           "real_api_dtype": str(next(real.module.parameters()).dtype)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
