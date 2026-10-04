#!/usr/bin/env python3
"""Census: Unsloth's patched models against plain HF / PEFT (fp32, TF32 off).

For each architecture a tiny random checkpoint is written to ``.cache/tiny_models`` (TinyLlama tokenizer).  Stage
``ref`` (no unsloth import) records loss and gradients of the HF model (full fine-tuning) and of a PEFT LoRA model
whose adapters are filled with seeded nonzero values; stage ``unsloth`` loads the same checkpoint through
``FastLanguageModel`` (full fine-tuning, and LoRA via ``get_peft_model`` with the same adapter values) and compares.
A conditioning floor (one-ulp parameter perturbation of the HF model) is recorded in ``ref`` so that a deviation is
judged in multiples of what rounding-level differences produce.

Run in the ``liger`` env (transformers 4.57.3, trl 0.24.0) with ``PYTHONPATH=.cache/pylibs/unsloth_latest``:

    python scripts/census_unsloth.py --out results/census/unsloth.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / ".cache" / "tiny_models"
TOKENIZER = "/data1/tzh/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BASE = dict(hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, vocab_size=32000, max_position_embeddings=512, pad_token_id=0, bos_token_id=1, eos_token_id=2,
            tie_word_embeddings=False)
MODELS = [
    ("llama", "base", {}),
    ("mistral", "base", {"sliding_window": 4096}),
    ("qwen2", "base", {}),
    ("qwen3", "base", {}),
    ("qwen3_moe", "base", {"num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64}),
    ("gemma", "base", {}),
    ("gemma2", "base", {"final_logit_softcapping": 30.0, "attn_logit_softcapping": 50.0, "sliding_window": 4096}),
    ("gemma3_text", "base", {"sliding_window": 4096, "num_hidden_layers": 6}),
    ("phi3", "base", {}),
    ("phi3", "partial_rotary_0.75", {"partial_rotary_factor": 0.75}),
    ("granite", "multipliers", {"embedding_multiplier": 12.0, "residual_multiplier": 0.22, "logits_scaling": 8.0,
                                "attention_multiplier": 0.03125}),
    ("cohere", "base", {"logit_scale": 0.0625, "use_qk_norm": False}),
    ("olmo2", "base", {}),
    ("glm4", "base", {}),
    ("smollm3", "base", {"num_hidden_layers": 4, "no_rope_layer_interval": 2}),
    ("gpt_oss", "base", {"num_local_experts": 4, "num_experts_per_tok": 2, "sliding_window": 32}),
    ("falcon_h1", "base", {"mamba_d_ssm": 128, "mamba_n_heads": 4, "mamba_d_head": 32, "mamba_n_groups": 1,
                           "mamba_d_state": 16, "mamba_chunk_size": 16}),
]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tiny_dir(model_type, variant):
    return TINY / f"{model_type}_{variant}"


def setup_torch():
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def inputs(torch):
    g = torch.Generator(device="cpu").manual_seed(1)
    ids = torch.randint(3, 32000, (2, 64), generator=g).cuda()
    labels = ids.clone()
    labels[:, :5] = -100
    return ids, labels


def fill_lora(model, torch):
    """Seeded nonzero adapter values keyed by a canonical name, identical in PEFT and Unsloth."""
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            key = canonical(name)
            g = torch.Generator(device="cpu").manual_seed(zlib.crc32(key.encode()))
            with torch.no_grad():
                p.copy_((torch.randn(p.shape, generator=g) * 0.02).to(p.device, p.dtype))


def canonical(name):
    """Parameter name with wrapper prefixes removed (base_model.model., _orig_mod., default.)."""
    for s in ("base_model.model.", "_orig_mod."):
        name = name.replace(s, "")
    return name


def grads_of(model, ids, labels, only_lora=False):
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    return float(out.loss.detach()), {canonical(n): p.grad.detach().float().cpu().clone()
                                      for n, p in model.named_parameters()
                                      if p.grad is not None and (not only_lora or "lora_" in n)}


def stage_ref(model_type, variant):
    torch = setup_torch()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import inspect

    overrides = next(o for m, v, o in MODELS if m == model_type and v == variant)
    cfg_cls = type(AutoConfig.for_model(model_type))
    accepted = set(inspect.signature(cfg_cls.__init__).parameters)
    kw = {k: v for k, v in {**BASE, **overrides}.items() if k in accepted or k in overrides}
    cfg = cfg_cls(**kw)
    for k in ("attention_dropout", "hidden_dropout", "resid_pdrop", "embd_pdrop", "router_jitter_noise"):
        if hasattr(cfg, k):
            setattr(cfg, k, 0.0)
    d = tiny_dir(model_type, variant)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    AutoTokenizer.from_pretrained(TOKENIZER).save_pretrained(d)
    model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32, attn_implementation="eager").cuda()
    model.train()
    ids, labels = inputs(torch)
    loss, g = grads_of(model, ids, labels)
    saved = {n: p.detach().clone() for n, p in model.named_parameters()}
    gen = torch.Generator(device="cuda").manual_seed(2)
    with torch.no_grad():
        for p in model.parameters():
            sign = torch.randint(0, 2, p.shape, generator=gen, device=p.device, dtype=p.dtype) * 2 - 1
            p.mul_(1 + sign * 2.0 ** -23)
    _, g_floor = grads_of(model, ids, labels)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(saved[n])
    from peft import LoraConfig, get_peft_model

    targets = [t for t in LORA_TARGETS if any(n.endswith(t) for n, _ in model.named_modules())]
    pm = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=targets, bias="none"))
    fill_lora(pm, torch)
    loss_lora, g_lora = grads_of(pm, ids, labels, only_lora=True)
    torch.save({"loss": loss, "grads": g, "floor": g_floor, "loss_lora": loss_lora, "grads_lora": g_lora,
                "targets": targets}, d / "ref.pt")
    return {"model": model_type, "variant": variant, "stage": "ref", "loss": loss, "loss_lora": loss_lora,
            "n_lora": len(g_lora)}


def rel(a, b):
    d = float((a.double() - b.double()).norm())
    n = float(b.double().norm())
    return d / n if n > 0 else d


def compare(got, ref, floor=None):
    per = {n: rel(got[n], ref[n]) for n in ref if n in got}
    out = {"rel_grad_max": max(per.values()) if per else None, "missing": sorted(set(ref) - set(got))[:6],
           "extra": sorted(set(got) - set(ref))[:6]}
    if floor is not None:
        fl = {n: rel(floor[n], ref[n]) for n in ref if n in floor}
        ratio = {n: per[n] / max(fl.get(n, 0.0), 1e-7) for n in per}
        out["ratio_to_floor_max"] = max(ratio.values()) if ratio else None
        out["worst"] = sorted(((n, per[n], fl.get(n), ratio[n]) for n in per), key=lambda t: -t[3])[:4]
    else:
        out["worst"] = sorted(per.items(), key=lambda kv: -kv[1])[:4]
    return out


def stage_unsloth(model_type, variant, mode):
    import unsloth  # noqa: F401  (must precede transformers)
    from unsloth import FastLanguageModel

    torch = setup_torch()
    d = tiny_dir(model_type, variant)
    ref = torch.load(d / "ref.pt")
    ids, labels = inputs(torch)
    model, _ = FastLanguageModel.from_pretrained(str(d), max_seq_length=128, dtype=torch.float32, load_in_4bit=False,
                                                 full_finetuning=(mode == "full"), use_gradient_checkpointing=False)
    rec = {"model": model_type, "variant": variant, "stage": "unsloth", "mode": mode,
           "unsloth_class": type(model).__name__}
    if mode == "full":
        model.train()
        loss, g = grads_of(model, ids, labels)
        rec.update(rel_loss=abs(loss - ref["loss"]) / abs(ref["loss"]), **compare(g, ref["grads"], ref["floor"]))
    else:
        pm = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                                              target_modules=ref["targets"], use_gradient_checkpointing=False,
                                              random_state=0)
        fill_lora(pm, torch)
        pm.train()
        loss, g = grads_of(pm, ids, labels, only_lora=True)
        rec.update(rel_loss=abs(loss - ref["loss_lora"]) / abs(ref["loss_lora"]), **compare(g, ref["grads_lora"]))
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--one", nargs=3, metavar=("STAGE", "MODEL", "VARIANT"))
    parser.add_argument("--mode", default="full", choices=("full", "lora"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()
    if args.one:
        stage, m, v = args.one
        try:
            rec = stage_ref(m, v) if stage == "ref" else stage_unsloth(m, v, args.mode)
        except Exception as exc:  # noqa: BLE001
            import traceback

            rec = {"model": m, "variant": v, "stage": stage, "mode": args.mode,
                   "error": f"{type(exc).__name__}: {str(exc)[:300]}", "tb": traceback.format_exc()[-800:]}
        print("RECORD " + json.dumps(rec), flush=True)
        return
    todo = [(m, v) for m, v, _ in MODELS if not args.models or m in args.models]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.out, "w")
    env0 = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    unsloth_path = str(ROOT / ".cache" / "pylibs" / "unsloth_latest")

    def call(i, argv, with_unsloth):
        env = dict(env0, CUDA_VISIBLE_DEVICES=str(i % 4))
        if with_unsloth:
            env["PYTHONPATH"] = unsloth_path
        cwd = ROOT / ".cache" / "tmp" / f"unsloth_w{i}"
        cwd.mkdir(parents=True, exist_ok=True)
        try:
            p = subprocess.run([sys.executable, str(Path(__file__).resolve())] + argv, env=env, cwd=cwd,
                               capture_output=True, text=True, timeout=1200)
            line = next((l[7:] for l in p.stdout.splitlines() if l.startswith("RECORD ")), None)
            return json.loads(line) if line else {"argv": argv, "error": p.stderr[-500:]}
        except subprocess.TimeoutExpired:
            return {"argv": argv, "error": "timeout"}

    def launch(i_mv):
        i, (m, v) = i_mv
        recs = [call(i, ["--one", "ref", m, v], False)]
        if "error" not in recs[0]:
            for mode in ("full", "lora"):
                recs.append(call(i, ["--one", "unsloth", m, v, "--mode", mode], True))
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if rec.get("stage") == "unsloth" and "error" not in rec:
                r = rec.get("ratio_to_floor_max")
                print(f"{m:12s} {v:20s} {rec['mode']:5s} {rec['unsloth_class']:28s} loss {rec['rel_loss']:.1e} "
                      f"grad {rec['rel_grad_max'] if rec['rel_grad_max'] is not None else float('nan'):.1e} "
                      f"floorx {r if r is not None else float('nan'):.0f} missing={len(rec['missing'])}", flush=True)
            elif "error" in rec:
                print(f"{m:12s} {v:20s} {rec.get('stage', '?')} {rec.get('mode', '')} ERROR {rec['error'][-200:]!r}",
                      flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(launch, enumerate(todo)))


if __name__ == "__main__":
    main()
