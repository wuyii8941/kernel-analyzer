"""Group C, one matrix multiplication (protocol appendix C): calls, input sources and the specification.

Inputs of a unit: the input of a linear layer of Qwen2.5-0.5B-Instruct (``capture_inputs`` below: an fp32 forward with
'highest' precision over held-out alpaca text, 128 tokens per unit, hooks on q_proj / down_proj of layers 2, 12, 22);
the weight is the layer's real weight (constant per stratum).  Calls:

* ``triton_default``: the matrix multiplication kernel of Triton's official tutorial 03 (python/tutorials/
  03-matrix-multiplication.py, MIT licence), called with fp32 inputs: ``tl.dot`` uses Triton's default input precision
  for fp32 (tf32);
* ``triton_ieee``: the same kernel with input_precision="ieee" (control);
* ``inductor_high``: torch.compile(F.linear, mode="max-autotune-no-cudagraphs") under
  torch.set_float32_matmul_precision('high') (PyTorch's own recommendation); the backend actually selected is recorded
  (a Triton template or cuBLAS);
* ``cublas_high`` / ``cublas_highest``: F.linear under 'high' / 'highest' (ATen; black box).
Specification ``linear_spec``: the exact product, enclosed as the float64 product +- gamma_k * (|X| |W|^T) (k the inner
dimension, gamma_k = k u / (1 - k u), u = 2^-53), outward rounded.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

STRATA = [(layer, mat) for layer in (2, 12, 22) for mat in ("q_proj", "down_proj")]
_CACHE = {}


def _root():
    return Path(os.environ.get("KA_TOL_CAPTURES_C", "/data1/tzh/kernel-analyzer/.cache/round2/captures_c"))


def _load(seed, layer, mat):
    key = ("x", seed, layer, mat)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = torch.load(_root() / f"unit{seed:04d}.pt", weights_only=False)[f"{layer}.{mat}"]
    return _CACHE[key]


def weight(layer, mat):
    key = ("w", layer, mat)
    if key not in _CACHE:
        _CACHE[key] = torch.load(_root() / "weights.pt", weights_only=False)[f"{layer}.{mat}"].cuda()
    return _CACHE[key]


def _source(layer, mat):
    def src(seed, shape, dtype):
        return _load(int(seed), layer, mat).cuda().float().clone()
    return src


# --- Triton tutorial 03 kernel (verbatim structure; fp32 inputs) -------------------------------------------------
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr, IEEE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        if IEEE:
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        else:
            accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _triton(x, w, ieee):
    wt = w.t()
    M, K = x.shape
    N = wt.shape[1]
    c = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
    matmul_kernel[grid](x, wt, c, M, N, K, x.stride(0), x.stride(1), wt.stride(0), wt.stride(1), c.stride(0),
                        c.stride(1), BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, IEEE=ieee)
    return c


def _calls(layer, mat):
    def triton_default(inp):
        return {"y": _triton(inp["x"], weight(layer, mat), False)}

    def triton_ieee(inp):
        return {"y": _triton(inp["x"], weight(layer, mat), True)}

    def inductor_high(inp):
        key = ("inductor", layer, mat)
        if key not in _CACHE:
            _CACHE[key] = torch.compile(lambda a, b: torch.nn.functional.linear(a, b),
                                        mode="max-autotune-no-cudagraphs")
        old = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")
        try:
            return {"y": _CACHE[key](inp["x"], weight(layer, mat))}
        finally:
            torch.set_float32_matmul_precision(old)

    def cublas(prec):
        def call(inp):
            old = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision(prec)
            try:
                return {"y": torch.nn.functional.linear(inp["x"], weight(layer, mat))}
            finally:
                torch.set_float32_matmul_precision(old)
        return call

    def spec(inp):
        x = inp["x"].double()
        w = weight(layer, mat).double()
        k = x.shape[1]
        u = 2.0 ** -53
        g = k * u / (1 - k * u)
        y = (x @ w.t()).cpu().numpy()
        b = (g * (x.abs() @ w.abs().t())).cpu().numpy() * (1 + 2.0 ** -50)
        return {"y": (np.nextafter(y - b, -np.inf), np.nextafter(y + b, np.inf))}

    return {"triton_default": triton_default, "triton_ieee": triton_ieee, "inductor_high": inductor_high,
            "cublas_high": cublas("high"), "cublas_highest": cublas("highest"), "spec": spec}


for _layer, _mat in STRATA:
    globals()[f"x_{_layer}_{_mat}"] = _source(_layer, _mat)
    for _name, _fn in _calls(_layer, _mat).items():
        globals()[f"{_name}_{_layer}_{_mat}"] = _fn


def declaration(layer, mat, impl, mode, units, seed_offset, k):
    cmp_ = {"mode": mode, "measure": ["y"]}
    if mode in ("B", "black-box"):
        cmp_["spec"] = f"calls_tf32.py:spec_{layer}_{mat}"
    return {"call": f"calls_tf32.py:{impl}_{layer}_{mat}",
            "inputs": {"x": {"state": f"calls_tf32.py:x_{layer}_{mat}", "shape": [128, k], "dtype": "float32"}},
            "compare": cmp_,
            "budget": {"cpu_seconds": 7200, "gpu_seconds": 7200, "case_timeout": 3600,
                       "max_units": units["development"] + units["confirmation"]},
            "units": {**units, "seed_offset": seed_offset}, "alpha": 0.05,
            "resolution": {"ulp_fraction": 0.125, "max_level": 1}}


def capture_inputs(out_dir, n_units, seed, hf_home, offset=0):
    """fp32 forward ('highest') over held-out alpaca text; per unit the inputs of q_proj / down_proj of layers 2, 12,
    22 for 128 consecutive tokens (four held-out examples per unit, concatenated).  Confirmation: offset 0 (examples
    pick[0:128]); development: offset 128 (disjoint)."""
    import json
    import glob
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_float32_matmul_precision("highest")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).cuda().eval()
    data = json.loads(Path(glob.glob(os.path.join(hf_home, "hub", "datasets--yahma--alpaca-cleaned", "snapshots", "*",
                                                  "alpaca_data_cleaned.json"))[0]).read_text())
    order = np.random.default_rng(1).permutation(len(data))
    held = [data[i] for i in order[-256:]]                     # the same held-out split as the workload
    pick = np.random.default_rng(seed).permutation(len(held))
    grabbed = {}
    hooks = []
    for layer in (2, 12, 22):
        for mat, mod in (("q_proj", model.model.layers[layer].self_attn.q_proj),
                         ("down_proj", model.model.layers[layer].mlp.down_proj)):
            hooks.append(mod.register_forward_hook(lambda m, i, o, key=f"{layer}.{mat}": grabbed.__setitem__(
                key, i[0].detach()[0, :128].cpu())))
    torch.save({f"{layer}.{mat}": getattr(model.model.layers[layer], "self_attn" if mat == "q_proj" else "mlp")
                .__getattr__(mat).weight.detach().cpu() for layer in (2, 12, 22) for mat in ("q_proj", "down_proj")},
               out_dir / "weights.pt")
    for u in range(n_units):
        text = "\n\n".join(e["instruction"] + "\n" + e.get("input", "") + "\n" + e["output"]
                           for e in (held[i] for i in pick[offset + 4 * u:offset + 4 * u + 4]))
        ids = tok(text, return_tensors="pt").input_ids[:, :128].cuda()
        if ids.shape[1] < 128:
            raise ValueError(f"unit {u}: {ids.shape[1]} tokens")
        with torch.no_grad():
            model(ids)
        torch.save(dict(grabbed), out_dir / f"unit{u:04d}.pt")
    for h in hooks:
        h.remove()
