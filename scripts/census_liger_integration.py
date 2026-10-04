#!/usr/bin/env python3
"""Census: Liger model integrations against the unpatched HF model (fp32, TF32 off).

For every text model type Liger patches, a tiny random model is built; the reference loss, logits and parameter
gradients are taken from the unpatched model, then one Liger option is applied to the same instance (``model=``) and
everything is recomputed.  In fp32 an honest kernel agrees with the reference to ~1e-5 relative; a larger gradient
difference is a candidate semantic deviation (formula, casting, offset, rope variant) or an input-source deviation
(in-place reuse of a buffer another node still reads).  One subprocess per (model type, option), because Liger
patches module globals.

    python scripts/census_liger_integration.py --out results/census/liger_integration.jsonl
    python scripts/census_liger_integration.py --one llama rms_norm      # single job, prints the record
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = dict(pad_token_id=0, bos_token_id=1, eos_token_id=2, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, vocab_size=512, max_position_embeddings=256, tie_word_embeddings=False)

# (model_type, variant name, config overrides).  Overrides follow the architecture features real checkpoints use.
MODELS = [
    ("llama", "base", {}),
    ("llama", "llama3_rope_scaling", {"rope_parameters": {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0,
                                                          "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                                          "original_max_position_embeddings": 64}}),
    ("granite", "multipliers", {"embedding_multiplier": 12.0, "residual_multiplier": 0.22, "logits_scaling": 8.0,
                                "attention_multiplier": 0.03125}),
    ("smollm3", "base", {"num_hidden_layers": 4, "no_rope_layer_interval": 2}),
    ("mistral", "sliding", {"sliding_window": 32}),
    ("mixtral", "base", {"num_local_experts": 4, "num_experts_per_tok": 2}),
    ("gemma", "base", {"head_dim": 32}),
    ("gemma2", "base", {"sliding_window": 32, "final_logit_softcapping": 30.0, "attn_logit_softcapping": 50.0}),
    ("gemma3_text", "base", {"sliding_window": 32, "num_hidden_layers": 6}),
    ("qwen2", "base", {}),
    ("qwen3", "base", {}),
    ("qwen3_moe", "base", {"num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64}),
    ("gpt_oss", "base", {"num_local_experts": 4, "num_experts_per_tok": 2, "sliding_window": 32}),
    ("phi3", "base", {}),
    ("phi3", "partial_rotary_0.75", {"partial_rotary_factor": 0.75}),
    ("olmo2", "base", {}),
    ("olmo3", "base", {"sliding_window": 32, "num_hidden_layers": 4}),
    ("glm4", "base", {}),
    ("falcon_h1", "base", {"mamba_d_ssm": 128, "mamba_n_heads": 4, "mamba_d_head": 32, "mamba_n_groups": 1,
                           "mamba_d_state": 16, "mamba_chunk_size": 16}),
    ("qwen3_next", "base", {"num_hidden_layers": 4, "num_experts": 4, "num_experts_per_tok": 2,
                            "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64,
                            "linear_key_head_dim": 32, "linear_value_head_dim": 32, "linear_num_key_heads": 2,
                            "linear_num_value_heads": 4}),
    ("hunyuan_v1_dense", "base", {}),
    ("hunyuan_v1_moe", "base", {"num_experts": 4, "moe_topk": 2}),
    ("exaone4", "base", {"sliding_window": 32, "num_hidden_layers": 4}),
    ("ministral", "sliding", {"sliding_window": 32}),
    ("nemotron", "base", {}),
    ("deepseek_v3", "base", {"moe_intermediate_size": 64, "n_routed_experts": 8, "n_shared_experts": 1,
                             "num_experts_per_tok": 2, "n_group": 2, "topk_group": 1, "kv_lora_rank": 32,
                             "q_lora_rank": 48, "qk_rope_head_dim": 16, "qk_nope_head_dim": 16, "v_head_dim": 32,
                             "first_k_dense_replace": 1}),
    ("deepseek_v4", "base", {"moe_intermediate_size": 64, "n_routed_experts": 8, "n_shared_experts": 1,
                             "num_experts_per_tok": 2, "q_lora_rank": 48, "o_lora_rank": 32, "sliding_window": 32,
                             "index_n_heads": 2, "index_head_dim": 32, "index_topk": 16, "num_nextn_predict_layers": 0}),
    ("gemma4_text", "base", {"num_hidden_layers": 6, "sliding_window": 32, "hidden_size_per_layer_input": 16,
                             "vocab_size_per_layer_input": 512}),
    ("qwen3_5_text", "base", {"num_hidden_layers": 4, "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                              "linear_num_key_heads": 2, "linear_num_value_heads": 4}),
    ("qwen3_5_moe_text", "base", {"num_hidden_layers": 4, "num_experts": 4, "num_experts_per_tok": 2,
                                  "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64,
                                  "linear_key_head_dim": 32, "linear_value_head_dim": 32, "linear_num_key_heads": 2,
                                  "linear_num_value_heads": 4}),
    ("llama4_text", "base", {"num_hidden_layers": 4, "num_local_experts": 4, "num_experts_per_tok": 1,
                             "intermediate_size_mlp": 256, "attention_chunk_size": 32}),
]


def rel(a, b):
    import torch

    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def job(model_type, variant, option):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    overrides = dict(next(o for m, v, o in MODELS if m == model_type and v == variant))
    kw = {**BASE, **overrides}
    cfg_cls = type(AutoConfig.for_model(model_type))
    accepted = set(inspect.signature(cfg_cls.__init__).parameters)
    cfg = cfg_cls(**{k: v for k, v in kw.items() if k in accepted or k in overrides})
    for k in ("attention_dropout", "hidden_dropout", "resid_pdrop", "embd_pdrop", "router_jitter_noise"):
        if hasattr(cfg, k):
            setattr(cfg, k, 0.0)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager", dtype=torch.float32).cuda()
    model.train()
    g = torch.Generator(device="cpu").manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (2, 64), generator=g).cuda()
    labels = ids.clone()
    labels[:, :5] = -100

    def run():
        model.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=labels)
        logits = None if out.logits is None else out.logits.detach().clone()
        out.loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        return float(out.loss), logits, grads

    loss_r, logits_r, grads_r = run()
    # Conditioning floor: the same unpatched model with every parameter moved by one random ulp (relative 2^-23).
    # A rounding-level kernel difference propagates like this; a semantic one does not stay within a few floors.
    saved = {n: p.detach().clone() for n, p in model.named_parameters()}
    gen = torch.Generator(device="cuda").manual_seed(2)
    with torch.no_grad():
        for p in model.parameters():
            sign = torch.randint(0, 2, p.shape, generator=gen, device=p.device, dtype=p.dtype) * 2 - 1
            p.mul_(1 + sign * 2.0 ** -23)
    _, _, grads_n = run()
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(saved[n])
    floor = {n: rel(grads_n[n], grads_r[n]) for n in grads_r if n in grads_n}
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN

    fn = MODEL_TYPE_TO_APPLY_LIGER_FN[model_type]
    flags = [p.name for p in inspect.signature(fn).parameters.values() if isinstance(p.default, bool)]
    if option == "defaults":
        fn(model=model)
    else:
        fn(model=model, **{f: (f == option) for f in flags})
    loss_l, logits_l, grads_l = run()
    per = {n: rel(grads_l[n], grads_r[n]) for n in grads_r if n in grads_l}
    missing = sorted(set(grads_r) ^ set(grads_l))
    ratio = {n: per[n] / max(floor.get(n, 0.0), 1e-7) for n in per}
    worst = sorted(((n, per[n], floor.get(n), ratio[n]) for n in per), key=lambda t: -t[3])[:4]
    import importlib.metadata as md

    rec = {"versions": {k: md.version(k) for k in ("liger-kernel", "transformers", "torch")},
           "model": model_type, "variant": variant, "option": option, "loss_ref": loss_r,
           "rel_loss": abs(loss_l - loss_r) / abs(loss_r),
           "rel_logits": None if logits_l is None or logits_r is None else rel(logits_l, logits_r),
           "rel_grad_max": max(per.values()) if per else None, "ratio_to_floor_max": max(ratio.values()) if ratio else None,
           "worst_params": worst, "grad_set_mismatch": missing}
    return rec


def options_for(model_type):
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN

    fn = MODEL_TYPE_TO_APPLY_LIGER_FN[model_type]
    return [p.name for p in inspect.signature(fn).parameters.values() if isinstance(p.default, bool)] + ["defaults"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--one", nargs="+", metavar=("MODEL", "OPTION"))
    parser.add_argument("--variant", default="base")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()
    if args.one:
        model_type, option = args.one
        try:
            rec = job(model_type, args.variant, option)
        except Exception as exc:  # noqa: BLE001
            rec = {"model": model_type, "variant": args.variant, "option": option,
                   "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        print("RECORD " + json.dumps(rec), flush=True)
        return
    jobs = [(m, v, o) for m, v, _ in MODELS if not args.models or m in args.models for o in options_for(m)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.out, "w")

    def launch(i_job):
        i, (m, v, o) = i_job
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % 4))
        p = subprocess.run([sys.executable, __file__, "--one", m, o, "--variant", v], env=env, capture_output=True,
                           text=True, timeout=900)
        line = next((l[7:] for l in p.stdout.splitlines() if l.startswith("RECORD ")), None)
        rec = json.loads(line) if line else {"model": m, "variant": v, "option": o, "error": p.stderr[-400:]}
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        g, r = rec.get("rel_grad_max"), rec.get("ratio_to_floor_max")
        flag = "  <== CHECK" if (r is not None and r > 30) or rec.get("error") or rec.get("grad_set_mismatch") else ""
        print(f"{m:18s} {v:22s} {o:28s} loss {rec.get('rel_loss', float('nan')):.1e} "
              f"grad {g if g is not None else float('nan'):.1e} x{r if r is not None else float('nan'):.0f} floor "
              f"{rec.get('error', '')[:120]}{flag}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(launch, enumerate(jobs)))


if __name__ == "__main__":
    main()
