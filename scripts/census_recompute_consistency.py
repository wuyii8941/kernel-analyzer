#!/usr/bin/env python3
"""Census of input-source consistency (class 2) under activation checkpointing, across HF causal-LM architectures.

With deterministic algorithms in fp32, the recomputation inside a checkpointed layer must reproduce the values the
original forward used, so gradients with gradient checkpointing must equal gradients without it bit for bit.  Any
difference means the backward reads values that the forward did not use (grad-mode-dependent branches, state mutated
during the forward, RNG not restored, ...).  For each architecture a tiny random model is built and compared in
three runs: no checkpointing, ``use_reentrant=True`` and ``use_reentrant=False``; dropout stays at the config
default and the RNG is reseeded before each run.

    python scripts/census_recompute_consistency.py --out results/census/recompute_consistency.jsonl
    python scripts/census_recompute_consistency.py --one llama
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

SIZES = dict(hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
             head_dim=32, vocab_size=512, max_position_embeddings=256, pad_token_id=0, bos_token_id=1, eos_token_id=2,
             num_experts=4, num_local_experts=4, n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
             shared_expert_intermediate_size=64, n_shared_experts=1, n_group=1, topk_group=1, sliding_window=32,
             linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
             kv_lora_rank=32, q_lora_rank=48, qk_rope_head_dim=16, qk_nope_head_dim=16, v_head_dim=32,
             first_k_dense_replace=1, attention_chunk_size=32, d_model=128, n_layer=2, n_head=4, n_embd=128,
             num_layers=2, num_heads=4, ffn_hidden_size=256, mamba_chunk_size=16, chunk_size=16, mamba_d_state=16,
             state_size=16, mamba_n_heads=4, mamba_d_head=32, mamba_d_ssm=128, mamba_n_groups=1, n_groups=1,
             mamba_head_dim=32, intermediate_size_mlp=256, ssm_state_size=16, hidden_size_per_layer_input=16,
             vocab_size_per_layer_input=512, index_n_heads=2, index_head_dim=32, index_topk=16, o_lora_rank=32,
             num_nextn_predict_layers=0, num_mtp_layers=0)


def rel(a, b):
    a, b = a.double(), b.double()
    d = (a - b).norm()
    return float(d / b.norm()) if float(b.norm()) > 0 else float(d)


def build(model_type, extra=None):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg_cls = type(AutoConfig.for_model(model_type))
    accepted = set(inspect.signature(cfg_cls.__init__).parameters)
    kw = {k: v for k, v in {**SIZES, **(extra or {})}.items() if k in accepted}
    cfg = cfg_cls(**kw)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager", dtype=torch.float32).cuda()
    return cfg, model


def job(model_type):
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg, model = build(model_type)
    model.train()
    vocab = getattr(cfg.get_text_config(), "vocab_size", 512)
    g = torch.Generator(device="cpu").manual_seed(1)
    ids = torch.randint(3, min(vocab, 512), (2, 64), generator=g).cuda()
    labels = ids.clone()
    labels[:, :5] = -100

    def run():
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)
        model.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=labels, use_cache=False)
        out.loss.backward()
        return float(out.loss.detach()), {n: p.grad.detach().clone() for n, p in model.named_parameters()
                                           if p.grad is not None}

    loss0, g0 = run()
    loss0b, g0b = run()  # run-to-run repeatability of the plain model
    rec = {"model": model_type, "loss": loss0,
           "repeat_rel_grad_max": max((rel(g0b[n], g0[n]) for n in g0 if n in g0b), default=None),
           "repeat_loss_equal": loss0b == loss0}
    for reentrant in (True, False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
        if reentrant:
            model.enable_input_require_grads()
        loss1, g1 = run()
        model.gradient_checkpointing_disable()
        per = {n: rel(g1[n], g0[n]) for n in g0 if n in g1}
        worst = sorted(per.items(), key=lambda kv: -kv[1])[:4]
        key = "reentrant" if reentrant else "nonreentrant"
        rec[key] = {"loss_equal": loss1 == loss0, "rel_loss": abs(loss1 - loss0) / max(abs(loss0), 1e-30),
                    "rel_grad_max": max(per.values()) if per else None, "worst": worst,
                    "grad_set_mismatch": sorted(set(g0) ^ set(g1))[:8]}
    return rec


def causal_lm_types():
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    return sorted(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--one")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()
    if args.one:
        try:
            rec = job(args.one)
        except Exception as exc:  # noqa: BLE001
            rec = {"model": args.one, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        print("RECORD " + json.dumps(rec), flush=True)
        return
    types = args.models or causal_lm_types()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.out, "w")

    def launch(i_m):
        i, m = i_m
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % 4))
        try:
            p = subprocess.run([sys.executable, __file__, "--one", m], env=env, capture_output=True, text=True,
                               timeout=600)
            line = next((l[7:] for l in p.stdout.splitlines() if l.startswith("RECORD ")), None)
            rec = json.loads(line) if line else {"model": m, "error": p.stderr[-300:]}
        except subprocess.TimeoutExpired:
            rec = {"model": m, "error": "timeout"}
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        if "error" in rec:
            print(f"{m:28s} ERROR {rec['error'][-150:]!r}", flush=True)
            return
        parts = []
        for key in ("reentrant", "nonreentrant"):
            r = rec[key]
            parts.append(f"{key} loss_eq={r['loss_equal']} grad={r['rel_grad_max']:.1e}")
        flag = "  <== CHECK" if any(not rec[k]["loss_equal"] or (rec[k]["rel_grad_max"] or 0) > 0
                                    for k in ("reentrant", "nonreentrant")) else ""
        print(f"{m:28s} repeat={rec['repeat_rel_grad_max']:.1e} " + " | ".join(parts) + flag, flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(launch, enumerate(types)))


if __name__ == "__main__":
    main()
