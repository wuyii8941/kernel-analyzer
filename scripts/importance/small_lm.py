"""Small Llama-style language model and training loop for the importance calibration (stage B of
docs/next_phase_plan_20261006.md).

Everything that the calibration varies is a switch here:
- precision: "fp32" (TF32 off), "tf32" (TF32 matmuls), "bf16" (bf16 autocast, fp32 master weights);
- component implementations (``impl``): rmsnorm / rope / swiglu / ce, each "torch" (fp32 reference math) or an
  alternative (e.g. "hf_bf16" for the HF-style intermediate bf16 cast, "unsloth" for the Unsloth Triton kernels);
- optimizer: "adamw" (fp32 states) or "adamw_bf16" (bf16 parameters and states, pure bf16 training);
- injection after every optimizer step: u = gamma * r (r = the step's update) or u = b * |r| * nu (nu a fixed unit
  direction per parameter tensor, drawn once from ``inject_seed``).

Data: GPT-2 tokens of wikitext-103 (``.cache/data/wikitext103_gpt2_{train,val}.npy``); the order of training
blocks is a permutation drawn from the run's data seed, so two runs with the same seed see the same batches.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / ".cache/data"


@dataclass
class Config:
    vocab: int = 50257
    d: int = 256
    layers: int = 4
    heads: int = 4
    ffn: int = 688
    seq: int = 256
    batch: int = 32
    steps: int = 2000
    lr: float = 1e-3
    warmup: int = 100
    min_lr_frac: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    clip: float = 1.0
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    precision: str = "fp32"
    optimizer: str = "adamw"
    impl: dict = field(default_factory=lambda: {"rmsnorm": "torch", "rope": "torch", "swiglu": "torch", "ce": "torch"})
    seed: int = 0               # init and data order
    inject: str = "none"        # "none" | "gamma" | "direction"
    dose: float = 0.0
    inject_seed: int = 12345
    eval_batches: int = 32
    checkpoints: tuple = ()     # steps after which the full training state is saved
    log_every: int = 50
    compile: bool = False       # torch.compile the model (Inductor)


# ------------------------------------------------------------------------------------------------ components

def rmsnorm(x, w, eps, impl):
    if impl == "torch":
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)
    if impl == "hf_bf16":  # transformers LlamaRMSNorm: normalise in fp32, cast back, then multiply by the weight
        xf = x.float()
        h = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return w * h.to(x.dtype)
    if impl == "compiled_hf":  # the HF formula compiled by Inductor (which drops the bf16 round trip, pytorch#181568)
        return _compiled_hf_rmsnorm(x, w, eps)
    raise KeyError(impl)


_COMPILED = {}


def _compiled_hf_rmsnorm(x, w, eps):
    if "rms" not in _COMPILED:
        _COMPILED["rms"] = torch.compile(lambda x, w, eps: rmsnorm(x, w, eps, "hf_bf16"), dynamic=False)
    return _COMPILED["rms"](x, w, eps)


def rope_tables(seq, dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    ang = torch.arange(seq, dtype=torch.float32, device=device)[:, None] * inv[None, :]
    ang = torch.cat([ang, ang], -1)
    return torch.cos(ang), torch.sin(ang)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def apply_rope(q, k, cos, sin, impl):
    """q, k: (batch, heads, seq, dim)."""
    if impl == "torch":
        c, s = cos.to(q.dtype), sin.to(q.dtype)
        return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s
    if impl == "unsloth":
        from unsloth.kernels.rope_embedding import fast_rope_embedding

        return fast_rope_embedding(q, k, cos.to(q.dtype), sin.to(q.dtype), None)
    raise KeyError(impl)


def swiglu(gate, up, impl):
    if impl == "torch":
        return F.silu(gate) * up
    if impl == "hf_bf16":  # silu in the activation dtype, product in the activation dtype
        return (F.silu(gate.float()).to(gate.dtype)) * up
    if impl == "unsloth":
        from unsloth.kernels.swiglu import swiglu_fg_kernel

        return _UnslothSwiGLU.apply(gate, up)
    raise KeyError(impl)


class _UnslothSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e, g):
        from unsloth.kernels.swiglu import swiglu_fg_kernel

        ctx.save_for_backward(e, g)
        return swiglu_fg_kernel(e, g)

    @staticmethod
    def backward(ctx, dh):
        from unsloth.kernels.swiglu import swiglu_DWf_DW_dfg_kernel

        e, g = ctx.saved_tensors
        shape = e.shape
        DW = dh.reshape(-1, shape[-1]).contiguous().clone()
        e2 = e.reshape(-1, shape[-1]).contiguous().clone()
        g2 = g.reshape(-1, shape[-1]).contiguous().clone()
        swiglu_DWf_DW_dfg_kernel(DW, e2, g2)  # writes (h, df, de) over (DW, e, g)
        return g2.view(shape), e2.view(shape)


def cross_entropy(logits, targets, impl):
    if impl == "torch":
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
    if impl == "unsloth":
        from unsloth.kernels.cross_entropy_loss import Fast_CrossEntropyLoss

        losses = Fast_CrossEntropyLoss.apply(logits.reshape(-1, logits.shape[-1]).contiguous(), targets.reshape(-1), 0, 0)
        return losses.mean()
    raise KeyError(impl)


# ------------------------------------------------------------------------------------------------ model

class Block(torch.nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.n1 = torch.nn.Parameter(torch.ones(c.d))
        self.n2 = torch.nn.Parameter(torch.ones(c.d))
        self.qkv = torch.nn.Linear(c.d, 3 * c.d, bias=False)
        self.o = torch.nn.Linear(c.d, c.d, bias=False)
        self.gate_up = torch.nn.Linear(c.d, 2 * c.ffn, bias=False)
        self.down = torch.nn.Linear(c.ffn, c.d, bias=False)

    def forward(self, x, cos, sin):
        c = self.c
        B, T, _ = x.shape
        h = rmsnorm(x, self.n1, c.norm_eps, c.impl["rmsnorm"])
        q, k, v = self.qkv(h).view(B, T, 3, c.heads, c.d // c.heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, k, cos[:T], sin[:T], c.impl["rope"])
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(att.transpose(1, 2).reshape(B, T, c.d))
        h = rmsnorm(x, self.n2, c.norm_eps, c.impl["rmsnorm"])
        g, u = self.gate_up(h).chunk(2, dim=-1)
        return x + self.down(swiglu(g.contiguous(), u.contiguous(), c.impl["swiglu"]))


class SmallLM(torch.nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.emb = torch.nn.Embedding(c.vocab, c.d)
        self.blocks = torch.nn.ModuleList(Block(c) for _ in range(c.layers))
        self.nf = torch.nn.Parameter(torch.ones(c.d))
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                torch.nn.init.normal_(p, std=0.02)
        for b in self.blocks:
            torch.nn.init.normal_(b.o.weight, std=0.02 / math.sqrt(2 * c.layers))
            torch.nn.init.normal_(b.down.weight, std=0.02 / math.sqrt(2 * c.layers))

    def forward(self, idx, targets):
        c = self.c
        if self.cos is None or self.cos.device != idx.device:
            self.cos, self.sin = rope_tables(c.seq, c.d // c.heads, c.rope_theta, idx.device)
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        x = rmsnorm(x, self.nf, c.norm_eps, c.impl["rmsnorm"])
        logits = x @ self.emb.weight.to(x.dtype).T
        return cross_entropy(logits, targets, c.impl["ce"])


# ------------------------------------------------------------------------------------------------ data

class Data:
    def __init__(self, c: Config, device):
        self.c, self.device = c, device
        self.train = np.load(DATA / "wikitext103_gpt2_train.npy", mmap_mode="r")
        self.val = np.load(DATA / "wikitext103_gpt2_val.npy", mmap_mode="r")
        n_blocks = (self.train.size - 1) // c.seq
        rng = np.random.default_rng(1_000_000 + c.seed)
        self.order = rng.permutation(n_blocks)

    def _blocks(self, arr, starts):
        x = np.stack([np.asarray(arr[s:s + self.c.seq + 1], dtype=np.int64) for s in starts])
        t = torch.from_numpy(x).to(self.device)
        return t[:, :-1], t[:, 1:]

    def batch(self, step, batch=None):
        b = batch or self.c.batch
        idx = self.order[(step * b) % self.order.size: (step * b) % self.order.size + b]
        return self._blocks(self.train, idx * self.c.seq)

    def random_batch(self, rng):
        idx = rng.integers(0, self.order.size, self.c.batch)
        return self._blocks(self.train, idx * self.c.seq)

    def val_batches(self):
        n = self.c.eval_batches * self.c.batch
        starts = np.arange(n) * self.c.seq
        starts = starts[starts + self.c.seq + 1 <= self.val.size]
        for i in range(0, starts.size, self.c.batch):
            yield self._blocks(self.val, starts[i:i + self.c.batch])


# ------------------------------------------------------------------------------------------------ training

def setup_precision(c: Config):
    torch.backends.cuda.matmul.allow_tf32 = c.precision == "tf32"
    torch.backends.cudnn.allow_tf32 = c.precision == "tf32"
    torch.set_float32_matmul_precision("high" if c.precision == "tf32" else "highest")


def autocast(c: Config):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=c.precision == "bf16")


def make_optimizer(model, c: Config):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": c.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=c.lr, betas=c.betas, eps=c.eps, foreach=False)


def lr_at(step, c: Config):
    if step < c.warmup:
        return c.lr * (step + 1) / c.warmup
    frac = (step - c.warmup) / max(1, c.steps - c.warmup)
    return c.lr * (c.min_lr_frac + (1 - c.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac)))


def directions(model, seed):
    """A fixed direction nu over the whole parameter vector (unit norm overall), as per-tensor pieces."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = {n: torch.randn(p.shape, generator=g, dtype=torch.float64) for n, p in model.named_parameters()}
    norm = math.sqrt(sum(float((v * v).sum()) for v in raw.values()))
    return {n: (v / norm).to(p.device, p.dtype) for (n, v), p in zip(raw.items(), model.parameters())}


@torch.no_grad()
def evaluate(model, data, c: Config):
    model.eval()
    tot, n = 0.0, 0
    for x, y in data.val_batches():
        with autocast(c):
            tot += float(model(x, y)) * x.shape[0]
        n += x.shape[0]
    model.train()
    return tot / n


def build(c: Config, device="cuda"):
    torch.manual_seed(c.seed)
    model = SmallLM(c).to(device)
    if c.optimizer == "adamw_bf16":
        model = model.to(torch.bfloat16)
    opt = make_optimizer(model, c)
    return model, opt


def forward(model, x, y, c: Config):
    if c.compile:
        if "model" not in _COMPILED or _COMPILED["model"][0] is not model:
            _COMPILED["model"] = (model, torch.compile(model, dynamic=False))
        return _COMPILED["model"][1](x, y)
    return model(x, y)


def train_step(model, opt, data, step, c: Config, nu=None):
    for g in opt.param_groups:
        g["lr"] = lr_at(step, c)
    x, y = data.batch(step)
    with autocast(c):
        loss = forward(model, x, y, c)
    loss.backward()
    if c.clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip)
    prev = {n: p.detach().clone() for n, p in model.named_parameters()} if c.inject != "none" else None
    opt.step()
    opt.zero_grad(set_to_none=True)
    if c.inject != "none":
        with torch.no_grad():
            if c.inject == "gamma":  # u = gamma r
                for n, p in model.named_parameters():
                    p.add_(p - prev[n], alpha=c.dose)
            else:  # u = b |r| nu, |r| over the whole parameter vector
                rnorm = math.sqrt(sum(float(((p.double() - prev[n].double()) ** 2).sum()) for n, p in model.named_parameters()))
                for n, p in model.named_parameters():
                    p.add_(nu[n], alpha=c.dose * rnorm)
    return float(loss.detach())


def train(c: Config, out_dir: Path | None = None, device="cuda"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    setup_precision(c)
    model, opt = build(c, device)
    data = Data(c, device)
    nu = directions(model, c.inject_seed) if c.inject == "direction" else None
    log, t0 = [], time.time()
    ckpts = set(c.checkpoints)
    for step in range(c.steps):
        loss = train_step(model, opt, data, step, c, nu)
        if step % c.log_every == 0 or step == c.steps - 1:
            log.append({"step": step, "train_loss": loss})
        if out_dir is not None and (step + 1) in ckpts:
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1, "config": asdict(c)},
                       out_dir / f"ckpt_{step + 1:05d}.pt")
    val = evaluate(model, data, c)
    return {"config": asdict(c), "val_loss": val, "log": log, "seconds": round(time.time() - t0, 1),
            "final_train_loss": log[-1]["train_loss"]}
