"""Independent implementations for the Correctness Illusion corpus (docs/external_eval_protocol_20261006.md), written
from the kernel sources and the operator names; nothing here calls the tool's code.

- ``declared(entry, inputs)``: the real-number program of each Triton kernel in float64 (its compile-time constants at
  the float32 values the kernel uses, the planted change included), for the independent recomputation of K_R (§7).
- ``trusted32(op, inputs)``: a torch float32 eager implementation of the operator (the TTrace-style threshold, §5 B3).
- ``spec64(op, inputs)``: our own torch float64 formulation of the operator, a cross-check of the authors' f (§3).

``inputs`` are the NumPy float32 arrays of one draw; outputs are flattened in row-major order of the kernel's output.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def c32(v):
    return float(np.float32(v))


def _rows(x):
    return x.reshape(-1, x.shape[-1])


def _softmax_rows(x, pad=0):
    """Softmax over the last axis; ``pad`` lanes holding 0.0 join the max and the denominator (the tail bug)."""
    m = x.max(axis=-1, keepdims=True)
    if pad:
        m = np.maximum(m, 0.0)
    e = np.exp(x - m)
    return e / (e.sum(axis=-1, keepdims=True) + pad * np.exp(-m))


def _attention(q, k, v, scale):
    s = (q @ k.T) * scale
    return _softmax_rows(s) @ v


def _flash_buggy(q, k, v, scale):
    """The online softmax of flash_attention_triton_buggy in real arithmetic: l is rescaled, acc is not."""
    n = k.shape[0]
    block_n = max(8, 1 << (min(n, 32) - 1).bit_length())
    m = np.full(q.shape[0], -np.inf)
    l = np.zeros(q.shape[0])
    acc = np.zeros((q.shape[0], v.shape[1]))
    for start in range(0, n, block_n):
        s = (q @ k[start:start + block_n].T) * scale
        m_new = np.maximum(m, s.max(axis=1))
        alpha = np.exp(m - m_new)
        p = np.exp(s - m_new[:, None])
        l = l * alpha + p.sum(axis=1)
        acc = acc + p @ v[start:start + block_n]
        m = m_new
    return acc / l[:, None]


def declared(entry, inputs):
    a = {k: v.astype(np.float64) for k, v in inputs.items()}
    if entry.startswith(("attention", "flash_attention")):
        q, k, v = a["q"], a["k"], a["v"]
        scale = c32(1.0 / math.sqrt(q.shape[1]))  # a run-time float32 argument
        if entry == "attention_triton_buggy":
            return _attention(q, k, v, 1.0).reshape(-1)
        if entry == "flash_attention_triton_buggy":
            return _flash_buggy(q, k, v, scale).reshape(-1)
        return _attention(q, k, v, scale).reshape(-1)
    if entry.startswith("matmul_triton"):
        x, y = a["a"], a["b"]
        if entry.endswith("buggy"):
            return np.outer(x[:, -1], y[-1, :]).reshape(-1)
        return (x @ y).reshape(-1)
    x = a["input"]
    if entry.startswith("softmax_triton"):
        r = _rows(x)
        pad = (1 << (r.shape[1] - 1).bit_length()) - r.shape[1] if entry.endswith("buggy") else 0
        return _softmax_rows(r, pad).reshape(-1)
    if entry.startswith("rmsnorm_triton"):
        r = _rows(x)
        d = (r * r).sum(axis=1, keepdims=True) / r.shape[1] + c32(1e-5)
        return (r / (d if entry.endswith("buggy") else np.sqrt(d))).reshape(-1)
    if entry.startswith("l2norm_triton"):
        r = _rows(x)
        d = (r * r).sum(axis=1, keepdims=True) + c32(1e-12)
        return (r / (d if entry.endswith("buggy") else np.sqrt(d))).reshape(-1)
    x = x.reshape(-1)
    if entry == "relu_triton":
        return np.where(x > 0, x, 0.0)
    if entry.startswith("leaky_relu_triton"):
        return np.where(x > 0, x, c32(0.1 if entry.endswith("buggy") else 0.01) * x)
    if entry == "elu_triton":
        return np.where(x >= 0, x, np.expm1(x))
    if entry.startswith("gelu_triton"):
        inner = c32(0.7978845608028654) * (x + c32(0.044715) * x ** 3)
        y = x * (1.0 + np.tanh(inner))  # sign * (1 - 2 / (exp(2|t|) + 1)) is tanh(t) in real arithmetic
        return y if entry.endswith("buggy") else 0.5 * y
    if entry == "sigmoid_triton":
        return 1.0 / (1.0 + np.exp(-x))
    if entry.startswith("silu_triton"):
        return x / (1.0 + np.exp(-(2.0 if entry.endswith("buggy") else 1.0) * x))
    if entry == "tanh_triton":
        return np.tanh(x)
    raise KeyError(entry)


def _torch_op(op, t, dtype):
    t = {k: torch.from_numpy(v).to(dtype) for k, v in t.items()}
    if op in ("attention", "flash_attention"):
        q, k, v = t["q"], t["k"], t["v"]
        return torch.softmax(q @ k.T / math.sqrt(q.shape[1]), -1) @ v
    if op == "matmul":
        return t["a"] @ t["b"]
    x = t["input"]
    if op == "softmax":
        return torch.softmax(x, -1)
    if op == "rmsnorm":
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5)
    if op == "l2norm":
        return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-12)
    if op == "layernorm":
        return F.layer_norm(x, (x.shape[-1],), eps=1e-5)
    return {"relu": torch.relu, "leaky_relu": lambda z: F.leaky_relu(z, 0.01), "elu": F.elu,
            "gelu": lambda z: F.gelu(z, approximate="tanh"), "sigmoid": torch.sigmoid, "silu": F.silu,
            "tanh": torch.tanh}[op](x)


def trusted32(op, inputs):
    with torch.no_grad():  # CPU float32: deterministic, no TF32
        return _torch_op(op, inputs, torch.float32).double().numpy().reshape(-1)


def spec64(op, inputs):
    with torch.no_grad():
        return _torch_op(op, inputs, torch.float64).numpy().reshape(-1)


def op_of(entry):
    """The operator family of an entry (the authors' reference it shares)."""
    for op in ("flash_attention", "attention", "leaky_relu", "matmul", "softmax", "rmsnorm", "l2norm", "layernorm",
               "relu", "elu", "gelu", "sigmoid", "silu", "tanh"):
        if entry.startswith(op):
            return op
    raise KeyError(entry)
