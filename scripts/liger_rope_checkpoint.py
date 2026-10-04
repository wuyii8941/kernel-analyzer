#!/usr/bin/env python3
"""Real checkpoint with Liger's RoPE patch only: held-out loss and the RoPE output difference on one layer.

Used for two cases found by ``census_liger_integration.py``:

* Phi-4-mini (model_type phi3, partial_rotary_factor 0.75): HF rotates the first ``rotary_dim = 96`` of 128 dims and
  passes the rest; Liger <= 0.8.4 rotates the whole head.  Fixed on Liger main by PR #1451.
* gpt-oss (model_type gpt_oss): HF's rotary embedding returns cos/sin of width ``head_dim / 2`` (no duplication) and
  rotates (x[:d/2], x[d/2:]).  Liger <= 0.8.4 reads cos/sin of width d/2 and is correct; Liger main (#1451) infers
  ``rotary_dim = min(cos.shape[-1], head_dim) = d/2`` and rotates only the first half of the head.

Measured: wikitext-103 validation loss of the unpatched model, then of the same instance after
``apply_liger_kernel_to_<type>(model=..., rope=True)`` with every other option off (the instance path HF Trainer's
``use_liger_kernel=True`` takes).

    python scripts/liger_rope_checkpoint.py --model /data1/tzh/models/microsoft/Phi-4-mini-instruct \
        --out results/census/phi4mini_liger_rope_0.8.4.json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bf16_adam_four_arm import batches  # noqa: E402


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--device-map", default=None, help="e.g. auto for models that do not fit one GPU")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import liger_kernel
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN

    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {"dtype": torch.bfloat16}
    if args.device_map:
        kw["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if not args.device_map:
        model = model.cuda()
    model.eval()
    first = next(model.parameters()).device
    it = batches(tok, "validation", args.seq, 2, seed=1)
    data = [next(it) for _ in range(args.batches)]

    @torch.no_grad()
    def evaluate():
        return float(np.mean([float(model(input_ids=x.to(first), labels=x.to(first)).loss) for x in data]))

    modeling = importlib.import_module(type(model).__module__)
    orig_apply = modeling.apply_rotary_pos_emb
    captured = {}

    def capture(q, k, cos, sin, *a, **kwargs):
        captured.setdefault("args", (q.detach().clone(), k.detach().clone(), cos.detach().clone(),
                                     sin.detach().clone()))
        return orig_apply(q, k, cos, sin, *a, **kwargs)

    modeling.apply_rotary_pos_emb = capture
    with torch.no_grad():
        model(input_ids=data[0][:1, :64].to(first))
    modeling.apply_rotary_pos_emb = orig_apply

    loss_ref = evaluate()
    fn = MODEL_TYPE_TO_APPLY_LIGER_FN[model.config.model_type]
    flags = [p.name for p in inspect.signature(fn).parameters.values() if isinstance(p.default, bool)]
    fn(model=model, **{f: f == "rope" for f in flags})
    patched = modeling.apply_rotary_pos_emb
    loss_liger = evaluate()

    q, k, cos, sin = (t.float() for t in captured["args"])
    q_ref, _ = orig_apply(q, k, cos, sin)
    q_lig, _ = patched(q.clone(), k.clone(), cos, sin)
    diff = (q_lig - q_ref).abs()
    rec = {"liger_kernel": md.version("liger-kernel"), "liger_path": liger_kernel.__file__,
           "transformers": md.version("transformers"), "model": args.model, "model_type": model.config.model_type,
           "head_dim": q.shape[-1], "cos_width": cos.shape[-1], "patched_fn": f"{patched.__module__}.{patched.__name__}",
           "tokens": args.batches * 2 * args.seq, "val_loss_hf": loss_ref, "val_loss_liger_rope": loss_liger,
           "q_rope_max_abs_diff": float(diff.max()), "q_rope_rel_diff": float(diff.norm() / q_ref.norm())}
    print(json.dumps(rec, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()
