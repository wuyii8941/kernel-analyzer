#!/usr/bin/env python3
"""Minimal reproducer: Liger's RoPE against HF's GPT-OSS ``apply_rotary_pos_emb`` (the function Liger replaces).

GPT-OSS builds cos/sin of width head_dim / 2 (``emb = freqs``, no duplication) and rotates (x[:d/2], x[d/2:]).
Prints the max abs and relative difference of the rotated q and k, and of dq/dk for a random upstream gradient.

    python scripts/repro_liger_gpt_oss_rope.py
"""

import importlib.metadata as md

import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding, apply_rotary_pos_emb

import liger_kernel
from liger_kernel.transformers.rope import liger_rotary_pos_emb

torch.manual_seed(0)
cfg = GptOssConfig()  # gpt-oss defaults: head_dim 64, yarn rope
B, Hq, Hk, T, D = 2, 8, 2, 256, cfg.head_dim
q = torch.randn(B, Hq, T, D, device="cuda", requires_grad=True)
k = torch.randn(B, Hk, T, D, device="cuda", requires_grad=True)
pos = torch.arange(T, device="cuda")[None].expand(B, T)
cos, sin = GptOssRotaryEmbedding(cfg).cuda()(q, pos)
print(f"liger-kernel {md.version('liger-kernel')} from {liger_kernel.__file__}")
print(f"head_dim {D}, cos width {cos.shape[-1]}")

q_ref, k_ref = apply_rotary_pos_emb(q, k, cos, sin)
gq, gk = torch.randn_like(q_ref), torch.randn_like(k_ref)
dq_ref, dk_ref = torch.autograd.grad((q_ref * gq).sum() + (k_ref * gk).sum(), (q, k))

q_l, k_l = liger_rotary_pos_emb(q.clone(), k.clone(), cos, sin)
dq_l, dk_l = torch.autograd.grad((q_l * gq).sum() + (k_l * gk).sum(), (q, k))

for name, a, b in (("q", q_l, q_ref), ("k", k_l, k_ref), ("dq", dq_l, dq_ref), ("dk", dk_l, dk_ref)):
    d = (a - b).abs()
    print(f"{name:3s} max|diff| {d.max().item():.3e}   rel {(d.norm() / b.norm()).item():.3e}")
