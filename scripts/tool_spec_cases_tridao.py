"""FlashAttention-repo and mamba_ssm Triton kernels (Dao-AILab / state-spaces, main branches) for tool_spec_check.

Imported from .cache/pylibs/shim (the Triton modules only; the CUDA extensions are not built).  Specifications are
the documented formulas, written out in float64 eager PyTorch on the same inputs (declared bound 2^-40 * max|f|):

* rotary (flash_attn.ops.triton.rotary.apply_rotary): out[..., :r] = x_r cos + rotate_half(x_r) sin at position
  t + seqlen_offset (positions restart per sequence with cu_seqlens), out[..., r:] = x[..., r:]; interleaved pairs
  (2i, 2i+1); conjugate uses -sin.  Reference: flash_attn.layers.rotary.apply_rotary_emb_torch.
* layer_norm_fn / rms_norm_fn: flash_attn.ops.triton.layer_norm.layer_norm_ref / rms_norm_ref (the authors'
  reference) in float64.
* cross_entropy_loss: per row (1-e)(lse - s_y) + e(lse - mean s) + z lse^2 on s = logit_scale * x, 0 for
  ignore_index; z_losses = z lse^2 (tests/losses/test_cross_entropy.py).
* mamba_ssm rmsnorm_fn (gated): mamba_ssm.ops.triton.layernorm_gated.rms_norm_ref in float64.
* mamba_chunk_scan_combined: the SSM recurrence h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t^T, y_t = C_t h_t + D x_t,
  y *= silu(z); dt = clamp(softplus(dt + dt_bias)) when requested; state reset where seq_idx changes.
* selective_state_update: one step of the same recurrence (selective_state_update_ref).

Environment: ka_main + PYTHONPATH=.cache/pylibs/shim.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tool_spec_check import Case, f64_point_spec


def P(t):
    return f64_point_spec(t.detach().cpu().numpy())


class FnCase(Case):
    """Generic: ``make(g) -> inputs``, ``run(inp) -> {name: tensor}``, ``ref(inp64) -> {name: tensor}``; with
    ``grad_of`` the backward is checked (upstream gradient ``_g`` on output ``loss_of``)."""

    spec_bound = "float64 evaluation of the documented formula, declared bound 2^-40 * max|f|"

    def __init__(self, name, impl, spec, make, run, ref, grad_of=(), loss_of=None):
        self.name, self.implementation, self.specification = name, impl, spec
        self.make, self.run_fn, self.ref_fn, self.grad_of, self.loss_of = make, run, ref, list(grad_of), loss_of

    def setup(self):
        self.launch(self.inputs(10_000))
        torch.cuda.synchronize()

    def inputs(self, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        inp = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in self.make(g).items()}
        if self.grad_of:
            with torch.no_grad():
                y = self.ref_fn(self._cast(inp, torch.float64))[self.loss_of]
            inp["_g"] = torch.randn(y.shape, generator=g).cuda()
        return inp

    def _cast(self, inp, dtype, leaves=None):
        out = {}
        for k, v in inp.items():
            if k == "_g":
                continue
            if torch.is_tensor(v) and v.is_floating_point():
                v = v.to(dtype) if dtype is not None else v.clone()
                if leaves is not None and k in self.grad_of:
                    v.requires_grad_(True)
                    leaves[k] = v
            out[k] = v
        return out

    def launch(self, inp):
        if not self.grad_of:
            return self.run_fn(self._cast(inp, None))
        leaves = {}
        y = self.run_fn(self._cast(inp, None, leaves))[self.loss_of]
        y.backward(inp["_g"])
        return {f"d{k}": v.grad for k, v in leaves.items()}

    def spec(self, inp):
        if not self.grad_of:
            with torch.no_grad():
                return {k: P(v) for k, v in self.ref_fn(self._cast(inp, torch.float64)).items()}
        leaves = {}
        y = self.ref_fn(self._cast(inp, torch.float64, leaves))[self.loss_of]
        grads = torch.autograd.grad(y, list(leaves.values()), inp["_g"].double())
        return {f"d{k}": P(gr) for k, gr in zip(leaves, grads)}


def rn(g, *s, scale=1.0):
    return torch.randn(*s, generator=g) * scale


CASES = []

# ---------------------------------------------------------------------------------------------------------------
# flash_attn rotary
# ---------------------------------------------------------------------------------------------------------------


def _rot_tables(g, seqlen_ro, ro_dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, ro_dim, 2, dtype=torch.float64) / ro_dim))
    t = torch.arange(seqlen_ro, dtype=torch.float64)
    f = torch.outer(t, inv)
    return f.cos().float(), f.sin().float()


def _rotary_ref(x, cos, sin, offsets, cu_seqlens, interleaved, conjugate):
    """x: (B, T, H, D) or (total, H, D) with cu_seqlens; cos/sin: (seqlen_ro, r/2)."""
    if conjugate:
        sin = -sin
    r = cos.shape[-1] * 2

    def rot(xs, pos):  # xs (T, H, D), pos (T,)
        c, s = cos[pos], sin[pos]  # (T, r/2)
        if interleaved:
            c, s = c.repeat_interleave(2, -1), s.repeat_interleave(2, -1)
            x1, x2 = xs[..., :r:2], xs[..., 1:r:2]
            rh = torch.stack((-x2, x1), -1).reshape(*xs.shape[:-1], r)
        else:
            c, s = torch.cat([c, c], -1), torch.cat([s, s], -1)
            x1, x2 = xs[..., : r // 2], xs[..., r // 2: r]
            rh = torch.cat([-x2, x1], -1)
        out = xs[..., :r] * c[:, None, :] + rh * s[:, None, :]
        return torch.cat([out, xs[..., r:]], -1)

    if cu_seqlens is None:
        B, T = x.shape[:2]
        outs = []
        for b in range(B):
            off = int(offsets[b]) if torch.is_tensor(offsets) else int(offsets)
            outs.append(rot(x[b], torch.arange(T, device=x.device) + off))
        return torch.stack(outs)
    outs = []
    for b in range(len(cu_seqlens) - 1):
        s0, s1 = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        off = int(offsets[b]) if torch.is_tensor(offsets) else int(offsets)
        outs.append(rot(x[s0:s1], torch.arange(s1 - s0, device=x.device) + off))
    return torch.cat(outs)


def _rot_case(name, B=2, T=128, H=4, D=64, ro=64, interleaved=False, offsets=0, varlen=None, conjugate=False,
              inplace=False):
    def make(g):
        total = varlen[-1] if varlen else None
        x = rn(g, total, H, D) if varlen else rn(g, B, T, H, D)
        maxoff = int(max(offsets)) if isinstance(offsets, (list, tuple)) else int(offsets)
        seqlen_ro = (max(b - a for a, b in zip(varlen[:-1], varlen[1:])) if varlen else T) + maxoff
        cos, sin = _rot_tables(g, seqlen_ro, ro)
        d = {"x": x, "cos": cos, "sin": sin}
        if isinstance(offsets, (list, tuple)):
            d["off"] = torch.tensor(offsets, dtype=torch.int32)
        if varlen:
            d["cu"] = torch.tensor(varlen, dtype=torch.int32)
        return d

    def run(inp):
        from flash_attn.ops.triton.rotary import apply_rotary

        off = inp.get("off", offsets if not isinstance(offsets, (list, tuple)) else 0)
        cu = inp.get("cu")
        mx = max(b - a for a, b in zip(varlen[:-1], varlen[1:])) if varlen else None
        y = apply_rotary(inp["x"], inp["cos"], inp["sin"], seqlen_offsets=off, cu_seqlens=cu, max_seqlen=mx,
                         interleaved=interleaved, inplace=inplace, conjugate=conjugate)
        return {"out": y}

    def ref(inp):
        off = inp.get("off", offsets if not isinstance(offsets, (list, tuple)) else 0)
        return {"out": _rotary_ref(inp["x"], inp["cos"], inp["sin"], off, inp.get("cu"), interleaved, conjugate)}

    return FnCase(f"fa_rotary_{name}", "flash_attn.ops.triton.rotary.apply_rotary (main)",
                  f"rotary embedding, documented formula: B={B} T={T} H={H} D={D} rotary_dim={ro} "
                  f"interleaved={interleaved} offsets={offsets} varlen={varlen} conjugate={conjugate} inplace={inplace}",
                  make, run, ref)


CASES += [
    _rot_case("basic"),
    _rot_case("partial32", ro=32),
    _rot_case("interleaved", interleaved=True),
    _rot_case("interleaved_partial", interleaved=True, ro=48),
    _rot_case("offset_int", offsets=37),
    _rot_case("offset_tensor", offsets=[5, 60]),
    _rot_case("varlen", varlen=[0, 50, 128, 200]),
    _rot_case("varlen_offsets", varlen=[0, 50, 128, 200], offsets=[3, 0, 17]),
    _rot_case("conjugate", conjugate=True),
    _rot_case("d80_ro48", D=80, ro=48),
    _rot_case("inplace", inplace=True),
]

# ---------------------------------------------------------------------------------------------------------------
# flash_attn layer_norm_fn / rms_norm_fn
# ---------------------------------------------------------------------------------------------------------------
R, N = 64, 384


def _ln_case(name, rms, residual=False, prenorm=False, zero_centered=False, parallel=False, rowscale=False,
             bias=True, backward=False):
    def make(g):
        d = {"x": rn(g, R, N), "w": 0.1 * rn(g, N) + (0.0 if zero_centered else 1.0)}
        if bias:
            d["b"] = 0.1 * rn(g, N)
        if residual:
            d["res"] = rn(g, R, N)
        if parallel:
            d["x1"] = rn(g, R, N)
            d["w1"] = 0.1 * rn(g, N) + (0.0 if zero_centered else 1.0)
            if bias:
                d["b1"] = 0.1 * rn(g, N)
        if rowscale:
            d["rs"] = torch.rand(R, generator=g) + 0.5
        return d

    def args(inp):
        return dict(residual=inp.get("res"), x1=inp.get("x1"), weight1=inp.get("w1"), bias1=inp.get("b1"),
                    eps=1e-6, rowscale=inp.get("rs"), prenorm=prenorm, zero_centered_weight=zero_centered)

    def pack(res):
        res = res if isinstance(res, tuple) else (res,)
        names = ["out"] + (["out1"] if parallel else []) + (["residual_out"] if prenorm else [])
        return dict(zip(names, res))

    def run(inp):
        from flash_attn.ops.triton.layer_norm import layer_norm_fn, rms_norm_fn

        fn = rms_norm_fn if rms else layer_norm_fn
        return pack(fn(inp["x"], inp["w"], inp.get("b"), **args(inp)))

    def ref(inp):
        from flash_attn.ops.triton.layer_norm import layer_norm_ref, rms_norm_ref

        fn = rms_norm_ref if rms else layer_norm_ref
        return pack(fn(inp["x"], inp["w"], inp.get("b"), **args(inp)))

    grads = ()
    if backward:
        grads = ["x", "w"] + (["b"] if bias else []) + (["res"] if residual else []) + (["x1"] if parallel else [])
    kind = "rms_norm_fn" if rms else "layer_norm_fn"
    return FnCase(f"fa_{'rms' if rms else 'ln'}_{name}" + ("_bwd" if backward else ""),
                  f"flash_attn.ops.triton.layer_norm.{kind} (main)",
                  f"{kind} == {'rms_norm_ref' if rms else 'layer_norm_ref'} (float64): residual={residual} "
                  f"prenorm={prenorm} zero_centered_weight={zero_centered} parallel={parallel} rowscale={rowscale}",
                  make, run, ref, grad_of=grads, loss_of="out")


for _bwd in (False, True):
    CASES += [
        _ln_case("plain", rms=True, backward=_bwd),
        _ln_case("residual_prenorm", rms=True, residual=True, prenorm=True, backward=_bwd),
        _ln_case("zero_centered", rms=True, zero_centered=True, backward=_bwd),
        _ln_case("parallel", rms=True, parallel=True, residual=True, backward=_bwd),
        _ln_case("rowscale", rms=True, rowscale=True, residual=True, backward=_bwd),
        _ln_case("plain", rms=False, backward=_bwd),
        _ln_case("residual_prenorm", rms=False, residual=True, prenorm=True, backward=_bwd),
        _ln_case("parallel", rms=False, parallel=True, backward=_bwd),
    ]

# ---------------------------------------------------------------------------------------------------------------
# flash_attn cross_entropy_loss
# ---------------------------------------------------------------------------------------------------------------
V = 1000


def _ce_case(name, smoothing=0.0, scale=1.0, zsq=0.0, backward=False):
    def make(g):
        t = torch.randint(0, V, (R,), generator=g)
        t[::7] = -100
        return {"x": rn(g, R, V, scale=3), "t": t}

    def run(inp):
        from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

        loss, z = cross_entropy_loss(inp["x"], inp["t"], label_smoothing=smoothing, logit_scale=scale,
                                     lse_square_scale=zsq)
        return {"loss": loss} if backward else {"loss": loss, "z_loss": z}

    def ref(inp):
        s = inp["x"] * scale
        t = inp["t"]
        valid = t != -100
        lse = torch.logsumexp(s, -1)
        ce = F.cross_entropy(s, t, label_smoothing=smoothing, reduction="none")
        z = torch.where(valid, zsq * lse ** 2, torch.zeros_like(lse))
        return {"loss": ce + z} if backward else {"loss": ce + z, "z_loss": z}

    return FnCase(f"fa_ce_{name}" + ("_bwd" if backward else ""), "flash_attn.ops.triton.cross_entropy (main)",
                  f"per-row cross entropy, label_smoothing={smoothing} logit_scale={scale} lse_square_scale={zsq}",
                  make, run, ref, grad_of=(["x"] if backward else ()), loss_of="loss")


for _bwd in (False, True):
    CASES += [_ce_case("plain", backward=_bwd), _ce_case("smooth", smoothing=0.1, backward=_bwd),
              _ce_case("scale_zloss", scale=0.7, zsq=1e-2, backward=_bwd),
              _ce_case("all", smoothing=0.1, scale=0.7, zsq=1e-2, backward=_bwd)]

# ---------------------------------------------------------------------------------------------------------------
# mamba_ssm gated RMSNorm
# ---------------------------------------------------------------------------------------------------------------


def _gn_case(name, z=True, group=None, before=True, bias=False, backward=False):
    def make(g):
        d = {"x": rn(g, R, N), "w": 1 + 0.1 * rn(g, N)}
        if bias:
            d["b"] = 0.1 * rn(g, N)
        if z:
            d["z"] = rn(g, R, N, scale=2)
        return d

    def run(inp):
        from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn

        return {"out": rmsnorm_fn(inp["x"], inp["w"], inp.get("b"), z=inp.get("z"), eps=1e-5, group_size=group,
                                  norm_before_gate=before)}

    def ref(inp):
        from mamba_ssm.ops.triton.layernorm_gated import rms_norm_ref

        return {"out": rms_norm_ref(inp["x"], inp["w"], inp.get("b"), z=inp.get("z"), eps=1e-5, group_size=group,
                                    norm_before_gate=before, upcast=False)}

    grads = (["x", "w"] + (["z"] if z else []) + (["b"] if bias else [])) if backward else ()
    return FnCase(f"mamba_gatednorm_{name}" + ("_bwd" if backward else ""),
                  "mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn (main)",
                  f"rms_norm_ref (float64): z={z} group_size={group} norm_before_gate={before} bias={bias}",
                  make, run, ref, grad_of=grads, loss_of="out")


for _bwd in (False, True):
    CASES += [_gn_case("before", backward=_bwd), _gn_case("after", before=False, backward=_bwd),
              _gn_case("group96_before", group=96, backward=_bwd),
              _gn_case("group96_after_bias", group=96, before=False, bias=True, backward=_bwd),
              _gn_case("noz", z=False, backward=_bwd)]

# ---------------------------------------------------------------------------------------------------------------
# mamba_ssm SSD (mamba_chunk_scan_combined) and selective_state_update
# ---------------------------------------------------------------------------------------------------------------


def _ssm_ref(x, dt, A, B, C, D=None, z=None, dt_bias=None, softplus=False, dt_limit=(0.0, float("inf")),
             h0=None, seq_idx=None):
    """x (b,l,h,p), dt (b,l,h), A (h), B/C (b,l,g,n) -> y (b,l,h,p), final state (b,h,p,n)."""
    b, l, h, p = x.shape
    g, n = B.shape[2], B.shape[3]
    if dt_bias is not None:
        dt = dt + dt_bias
    if softplus:
        dt = F.softplus(dt)
    dt = dt.clamp(dt_limit[0], dt_limit[1])
    rep = h // g
    Bh, Ch = B.repeat_interleave(rep, 2), C.repeat_interleave(rep, 2)  # (b,l,h,n)
    state = torch.zeros(b, h, p, n, dtype=x.dtype, device=x.device) if h0 is None else h0.clone()
    ys = []
    for t in range(l):
        if seq_idx is not None and t > 0:
            reset = (seq_idx[:, t] != seq_idx[:, t - 1]).to(x.dtype)[:, None, None, None]
            state = state * (1 - reset)
        decay = torch.exp(dt[:, t] * A)[:, :, None, None]
        state = state * decay + (dt[:, t][:, :, None, None] * x[:, t][:, :, :, None] * Bh[:, t][:, :, None, :])
        y = (state * Ch[:, t][:, :, None, :]).sum(-1)
        if D is not None:
            y = y + (D[None, :, None] if D.dim() == 1 else D[None]) * x[:, t]
        ys.append(y)
    y = torch.stack(ys, 1)
    if z is not None:
        y = y * F.silu(z)
    return y, state


def _ssd_case(name, b=1, l=128, h=4, p=32, g=1, n=16, chunk=32, D=False, z=False, dt_bias=False, softplus=False,
              dt_limit=(0.0, float("inf")), h0=False, final=False, seq_idx=None):
    def make(gen):
        d = {"x": rn(gen, b, l, h, p), "dt": 0.1 * torch.rand(b, l, h, generator=gen) + 0.01,
             "A": -torch.rand(h, generator=gen) - 0.5, "B": rn(gen, b, l, g, n), "C": rn(gen, b, l, g, n)}
        if D:
            d["D"] = rn(gen, h)
        if z:
            d["z"] = rn(gen, b, l, h, p)
        if dt_bias:
            d["dtb"] = 0.5 * rn(gen, h)
            d["dt"] = rn(gen, b, l, h) - 2.0
        if h0:
            d["h0"] = 0.5 * rn(gen, b, h, p, n)
        if seq_idx is not None:
            d["sidx"] = torch.tensor([seq_idx], dtype=torch.int32).expand(b, l).contiguous()
        return d

    def run(inp):
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        res = mamba_chunk_scan_combined(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], chunk, D=inp.get("D"),
                                        z=inp.get("z"), dt_bias=inp.get("dtb"), initial_states=inp.get("h0"),
                                        seq_idx=inp.get("sidx"), dt_softplus=softplus, dt_limit=dt_limit,
                                        return_final_states=final)
        return {"y": res[0], "final_state": res[1]} if final else {"y": res}

    def ref(inp):
        y, st = _ssm_ref(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], inp.get("D"), inp.get("z"),
                         inp.get("dtb"), softplus, dt_limit, inp.get("h0"), inp.get("sidx"))
        return {"y": y, "final_state": st} if final else {"y": y}

    return FnCase(f"mamba_ssd_{name}", "mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined (main)",
                  f"SSM recurrence (float64): b={b} l={l} h={h} p={p} g={g} n={n} chunk={chunk} D={D} z={z} "
                  f"dt_bias={dt_bias} softplus={softplus} dt_limit={dt_limit} h0={h0} final={final} "
                  f"seq_idx={'yes' if seq_idx else None}", make, run, ref)


_seq = [0] * 40 + [1] * 50 + [2] * 38
CASES += [
    _ssd_case("basic"),
    _ssd_case("D_z", D=True, z=True),
    _ssd_case("dtbias_softplus", dt_bias=True, softplus=True),
    _ssd_case("dt_limit", dt_bias=True, softplus=True, dt_limit=(0.02, 0.3)),
    _ssd_case("h0_final", h0=True, final=True),
    _ssd_case("groups2", h=4, g=2),
    _ssd_case("len100", l=100),
    _ssd_case("seq_idx", seq_idx=_seq),
]


def _ssu_case(name, D=True, z=True, dt_bias=True, softplus=True, g=1):
    bsz, h, p, n = 4, 4, 32, 16

    def make(gen):
        d = {"state": rn(gen, bsz, h, p, n), "x": rn(gen, bsz, h, p), "dt": rn(gen, bsz, h, p) - 1.0,
             "A": -torch.rand(h, p, n, generator=gen) - 0.5, "B": rn(gen, bsz, g, n), "C": rn(gen, bsz, g, n)}
        if D:
            d["D"] = rn(gen, h, p)
        if z:
            d["z"] = rn(gen, bsz, h, p)
        if dt_bias:
            d["dtb"] = 0.5 * rn(gen, h, p)
        return d

    def run(inp):
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update

        state = inp["state"]
        y = selective_state_update(state, inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], D=inp.get("D"),
                                   z=inp.get("z"), dt_bias=inp.get("dtb"), dt_softplus=softplus)
        return {"y": y, "state": state}

    def ref(inp):
        state = inp["state"].clone()
        dt = inp["dt"] + inp["dtb"] if "dtb" in inp else inp["dt"]
        if softplus:
            dt = F.softplus(dt)
        rep = h // g
        Bh, Ch = inp["B"].repeat_interleave(rep, 1), inp["C"].repeat_interleave(rep, 1)
        dA = torch.exp(dt[..., None] * inp["A"])
        state = state * dA + dt[..., None] * inp["x"][..., None] * Bh[:, :, None, :]
        y = (state * Ch[:, :, None, :]).sum(-1)
        if "D" in inp:
            y = y + inp["x"] * inp["D"]
        if "z" in inp:
            y = y * F.silu(inp["z"])
        return {"y": y, "state": state}

    return FnCase(f"mamba_ssu_{name}", "mamba_ssm.ops.triton.selective_state_update (main)",
                  f"one recurrence step (float64): D={D} z={z} dt_bias={dt_bias} softplus={softplus} groups={g}",
                  make, run, ref)


CASES += [_ssu_case("full"), _ssu_case("plain", z=False, softplus=False),  # D=None / dt_bias=None crash: mamba #1028
          _ssu_case("groups2", g=2)]
