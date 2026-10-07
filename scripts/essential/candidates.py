"""Phase-1 candidates (PyTorch's own implementations; protocol section 3 with section 12 entry 5).

``CANDIDATES[name] = (device, dtype, compiled)``.  ``run(family, cond, inp, name)`` returns
{"status": "ok", "received": {...}, "upstream": ..., "outputs": {name: float64 array}, ...} or a status
"unsupported" / "error" with the reason.  Inputs are rounded to the candidate's dtype first; ``received`` holds the
values the candidate actually got (float64), on which the specification is evaluated.

Compiled candidates: the forward is compiled with ``fullgraph=True`` after ``torch._dynamo.reset()`` (so neither a
graph break nor the recompile limit can silently fall back to eager); the backward is AOTAutograd's compiled backward.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

CANDIDATES = {
    "eager_cpu64": ("cpu", torch.float64, False),
    "eager_cpu32": ("cpu", torch.float32, False),
    "eager_cuda32": ("cuda", torch.float32, False),
    "eager_cuda_bf16": ("cuda", torch.bfloat16, False),
    "inductor_cuda32": ("cuda", torch.float32, True),
    "inductor_cuda_bf16": ("cuda", torch.bfloat16, True),
    "inductor_cpu32": ("cpu", torch.float32, True),      # 2b G8 discovery queue (Inductor CPU / C++ backend)
}
FAMILY_CANDIDATES = {
    "ce": list(CANDIDATES),
    "pool": ["eager_cpu64", "eager_cpu32", "eager_cuda32", "inductor_cuda32"],
    "index": ["eager_cpu64", "eager_cpu32", "eager_cuda32", "inductor_cuda32"],
}
DTYPE_NAME = {torch.float64: "float64", torch.float32: "float32", torch.bfloat16: "bfloat16"}


def _t(a, device, dtype, layout="contiguous"):
    """values rounded to dtype on the CPU, placed with the requested storage layout (phase-1 supplement S3):
    "contiguous"; "strided" (every second element of a buffer twice as long in the last dimension);
    "offset" (a view starting 3 elements into a larger buffer: non-zero storage offset);
    "transposed" (the last two dimensions swapped in storage, then viewed back)."""
    t = torch.as_tensor(np.asarray(a, dtype=np.float64)).to(dtype)
    if layout == "contiguous":
        return t.to(device)
    if layout == "strided":
        buf = torch.zeros(tuple(t.shape[:-1]) + (2 * t.shape[-1],), dtype=dtype)
        buf[..., ::2] = t
        return buf.to(device)[..., ::2]
    if layout == "offset":
        flat = torch.zeros(t.numel() + 3, dtype=dtype)
        flat[3:] = t.reshape(-1)
        return flat.to(device)[3:].view(t.shape)
    if layout == "transposed":
        if t.dim() < 2:
            return _t(a, device, dtype, "strided")
        return t.transpose(-1, -2).contiguous().to(device).transpose(-1, -2)
    raise KeyError(layout)


def _np(t):
    return t.detach().double().cpu().numpy()


def _compiled(fn, compiled):
    if not compiled:
        return fn
    torch._dynamo.reset()
    return torch.compile(fn, fullgraph=True, dynamic=False)


def ce_upstream(cond, seed):
    import hashlib                             # deterministic across processes (Python's hash() is salted)
    h = int.from_bytes(hashlib.sha256(f"{cond['id']}/{seed}/upstream".encode()).digest()[:8], "little")
    rng = np.random.default_rng(h)
    if cond["reduction"] == "none":
        v = rng.integers(1, 4, cond["N"]) * rng.choice([-1, 1], cond["N"])
        return v.astype(np.float64)
    return np.array(float(rng.choice([-2, -1, 1, 2, 3])))


def pool_upstream(cond, inp, out_shape):
    rng = np.random.default_rng(inp["upstream_seed"])
    return rng.integers(-3, 4, out_shape).astype(np.float64)


def index_upstream(inp, shape):
    rng = np.random.default_rng(inp["upstream_seed"])
    return rng.integers(-3, 4, shape).astype(np.float64)


# ------------------------------------------------------------------------------------------------ cross-entropy

def run_ce(cond, inp, name, seed, fn_cache=None):
    device, dtype, compiled = CANDIDATES[name]
    x = inp["logits"]
    if cond["layout"] == "transposed":
        base = _t(x.T, device, dtype).contiguous().requires_grad_(True)          # storage (C, N)
        logits_of = lambda b: b.t()                                               # noqa: E731
    else:
        base = _t(x, device, dtype).requires_grad_(True)
        logits_of = lambda b: b                                                   # noqa: E731
    if cond["target"] == "prob":
        target = _t(inp["target"], device, dtype)
    else:
        target = torch.as_tensor(inp["target"], dtype=torch.long, device=device)
    weight = None if inp["weights"] is None else _t(inp["weights"], device, dtype)
    eps = float(cond["eps"])
    kw = dict(ignore_index=cond["ignore_index"], reduction=cond["reduction"], label_smoothing=eps)

    def fwd(logits, target, weight):
        return F.cross_entropy(logits, target, weight=weight, **kw)

    v = ce_upstream(cond, seed)
    try:
        f = _compiled(fwd, compiled)
        loss = f(logits_of(base), target, weight)
        vt = torch.as_tensor(v, dtype=loss.dtype, device=device)
        (loss * vt).sum().backward()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"}
    grad = base.grad.t() if cond["layout"] == "transposed" else base.grad
    received = {"logits": _np(logits_of(base)), "target": inp["target"] if cond["target"] == "index" else _np(target),
                "weights": None if weight is None else _np(weight).tolist(),
                # the scalar as the implementation receives it: ATen takes a double; Inductor bakes it into the
                # generated Triton as a float32 constant (protocol section 12 entry 6)
                "eps": float(np.float32(eps)) if compiled else eps}
    return {"status": "ok", "received": received, "upstream": v, "dtype": DTYPE_NAME[dtype],
            "outputs": {"loss": _np(loss).reshape(-1), "grad": _np(grad)}}


# ------------------------------------------------------------------------------------------------ pooling

def run_pool(cond, inp, name):
    device, dtype, compiled = CANDIDATES[name]
    nd = cond["nd"]
    x = _t(inp["x"][None], device, dtype, cond.get("layout", "contiguous")).requires_grad_(True)   # batch of 1
    if cond["op"] == "avg_pool":
        if nd == 1 and cond["divisor_override"] is not None:
            return {"status": "unsupported", "reason": "avg_pool1d has no divisor_override"}
        pool = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}[nd]
        kw = dict(kernel_size=cond["kernel"], stride=cond["stride"], padding=cond["padding"],
                  ceil_mode=cond["ceil_mode"], count_include_pad=cond["count_include_pad"])
        if nd > 1:
            kw["divisor_override"] = cond["divisor_override"]

        def fwd(x):
            return pool(x, **kw)
    else:
        pool = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}[nd]
        kw = dict(kernel_size=cond["kernel"], stride=cond["stride"], padding=cond["padding"],
                  dilation=cond["dilation"], ceil_mode=cond["ceil_mode"], return_indices=True)

        def fwd(x):
            return pool(x, **kw)
    try:
        f = _compiled(fwd, compiled)
        res = f(x)
        y, ind = (res if isinstance(res, tuple) else (res, None))
        v = pool_upstream(cond, inp, tuple(y.shape[1:]))
        (y * torch.as_tensor(v[None], dtype=y.dtype, device=device)).sum().backward()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"}
    out = {"out": _np(y[0]), "grad": _np(x.grad[0])}
    if ind is not None:
        out["indices"] = ind[0].detach().cpu().numpy().astype(np.int64)
    return {"status": "ok", "received": {"x": _np(x[0])}, "upstream": v, "dtype": DTYPE_NAME[dtype], "outputs": out}


# ------------------------------------------------------------------------------------------------ index / scatter

def run_index(cond, inp, name):
    device, dtype, compiled = CANDIDATES[name]
    lay = cond.get("layout", "contiguous")
    s = _t(inp["self"], device, dtype, lay).requires_grad_(True)
    src = _t(inp["source"], device, dtype, lay).requires_grad_(True)
    idx = torch.as_tensor(inp["index"], dtype=torch.long, device=device)
    dim = inp["dim"]
    op = cond["op"]
    if op == "index_add":
        alpha = cond["alpha"]

        def fwd(s, idx, src):
            return torch.index_add(s, dim, idx, src, alpha=alpha)
    elif op == "index_reduce":
        red, inc = cond["reduce"], cond["include_self"]

        def fwd(s, idx, src):
            return torch.index_reduce(s, dim, idx, src, red, include_self=inc)
    else:
        red, inc = cond["reduce"], cond["include_self"]

        def fwd(s, idx, src):
            return torch.scatter_reduce(s, dim, idx, src, red, include_self=inc)
    v = index_upstream(inp, tuple(s.shape))
    try:
        f = _compiled(fwd, compiled)
        y = f(s, idx, src)
        (y * torch.as_tensor(v, dtype=y.dtype, device=device)).sum().backward()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"}
    return {"status": "ok", "received": {"self": _np(s), "source": _np(src)}, "upstream": v, "dtype": DTYPE_NAME[dtype],
            "outputs": {"out": _np(y), "grad_self": _np(s.grad), "grad_source": _np(src.grad)}}


def run(family, cond, inp, name, seed):
    if family == "ce":
        return run_ce(cond, inp, name, seed)
    if family == "pool":
        return run_pool(cond, inp, name)
    return run_index(cond, inp, name)
