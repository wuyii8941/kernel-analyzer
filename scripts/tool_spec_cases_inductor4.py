"""PyTorch Inductor, fourth batch: where compilers historically go wrong.

* layout: transposed, strided, expanded inputs into reductions, norms, softmax, scans;
* large reductions (Inductor splits them into two kernels, loops long rows, uses online softmax): LM-sized
  vocabularies (32k, 128k), 2^21-element sums / variances / norms;
* NaN / inf semantics: fully masked softmax rows, NaN in max / argmax / sort / clamp, all -inf logsumexp, 0 / 0;
* integers and type promotion: overflow, negative floor division and remainder, int -> float, bool masks;
* in-place mutation through views and aliasing (class 2: reading the wrong data);
* common training code: total gradient norm + clipping over many tensors, EMA with foreach lerp.

Each case: ``make(g)`` builds base tensors; ``prep`` builds the actual arguments (views) from them on every call;
the compiled function and the float64 eager function see the same structure.  Outputs include mutated inputs.
Specification: eager PyTorch in float64 (integers unchanged), declared bound 2^-40 * max|f| on finite values;
special values (NaN / +-inf) are compared as classes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tool_spec_check import Case, f64_point_spec


class ProgCase(Case):
    implementation = "torch.compile(fn) -> Inductor Triton kernels"
    spec_bound = "float64 eager evaluation of the same program, declared bound 2^-40 * max|f| (finite values)"

    def __init__(self, name, fn, make, prep=None, mutated=(), grad_of=(), doc=""):
        self.name, self.fn, self.make, self.prep = name, fn, make, prep or (lambda **t: t)
        self.mutated, self.grad_of = list(mutated), list(grad_of)
        self.specification = f"eager PyTorch semantics (float64) of: {doc or name}" + ("; backward" if grad_of else "")

    def setup(self):
        self._compiled = torch.compile(self.fn, dynamic=False)
        self.launch(self.inputs(10_000))
        torch.cuda.synchronize()

    def inputs(self, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        inp = {k: v.cuda() for k, v in self.make(g).items()}
        if self.grad_of:
            with torch.no_grad():
                y = self._outputs(self.fn, self._args(inp, torch.float64))["out"]
            inp["_g"] = torch.randn(y.shape, generator=g).cuda()
        return inp

    def _args(self, inp, dtype, leaves=None):
        t = {}
        for k, v in inp.items():
            if k == "_g":
                continue
            v = v.clone()
            if dtype is not None and v.is_floating_point():
                v = v.to(dtype)
            if leaves is not None and k in self.grad_of:
                v.requires_grad_(True)
                leaves[k] = v
            t[k] = v
        return t

    def _outputs(self, fn, t):
        args = self.prep(**t)
        res = fn(**args)
        out = dict(res) if isinstance(res, dict) else {"out": res}
        for k in self.mutated:
            out[f"mutated_{k}"] = t[k]
        return out

    def launch(self, inp):
        if not self.grad_of:
            return self._outputs(self._compiled, self._args(inp, None))
        leaves = {}
        y = self._outputs(self._compiled, self._args(inp, None, leaves))["out"]
        y.backward(inp["_g"])
        return {f"d{k}": v.grad for k, v in leaves.items()}

    def spec(self, inp):
        if not self.grad_of:
            with torch.no_grad():
                out = self._outputs(self.fn, self._args(inp, torch.float64))
            return {k: f64_point_spec(v.detach().double().cpu().numpy()) for k, v in out.items()}
        leaves = {}
        y = self._outputs(self.fn, self._args(inp, torch.float64, leaves))["out"]
        grads = torch.autograd.grad(y, list(leaves.values()), inp["_g"].double())
        return {f"d{k}": f64_point_spec(gr.detach().cpu().numpy()) for k, gr in zip(leaves, grads)}


def rn(g, *s, scale=1.0):
    return torch.randn(*s, generator=g) * scale


CASES = []


def add(name, fn, make, prep=None, mutated=(), bwd=None, doc=""):
    CASES.append(ProgCase(f"i4_{name}", fn, make, prep, mutated, doc=doc))
    if bwd:
        CASES.append(ProgCase(f"i4_{name}_bwd", fn, make, prep, mutated, grad_of=bwd, doc=doc))


X = lambda g: {"x": rn(g, 96, 160)}  # noqa: E731

# ---- layout --------------------------------------------------------------------------------------------------
add("sum_transposed", lambda x: x.sum(1), X, prep=lambda x: {"x": x.t()}, bwd=["x"], doc="x.t().sum(1)")
add("softmax_dim0", lambda x: torch.softmax(x, 0), X, bwd=["x"], doc="softmax over dim 0")
add("layernorm_permuted", lambda x, w: F.layer_norm(x, (x.shape[-1],), w),
    lambda g: {"x": rn(g, 8, 48, 40) + 1, "w": 1 + 0.1 * rn(g, 8)}, prep=lambda x, w: {"x": x.permute(2, 1, 0), "w": w},
    bwd=["x", "w"], doc="layer_norm of a permuted input")
add("var_strided", lambda x: torch.var(x, 0), lambda g: {"x": rn(g, 200, 300) + 2},
    prep=lambda x: {"x": x[::2, ::3]}, bwd=["x"], doc="var over dim 0 of x[::2, ::3]")
add("cumsum_dim0", lambda x: torch.cumsum(x, 0), X, bwd=["x"], doc="cumsum over dim 0")
add("expanded_softmax", lambda x: torch.softmax(x * 3, -1), lambda g: {"x": rn(g, 1, 512)},
    prep=lambda x: {"x": x.expand(64, 512)}, doc="softmax of an expanded (stride 0) input")
add("logsumexp_transposed", lambda x: torch.logsumexp(x, 1), lambda g: {"x": rn(g, 300, 70, scale=5)},
    prep=lambda x: {"x": x.t()}, bwd=["x"], doc="logsumexp of x.t()")
add("mean_mid_dim", lambda x: x.mean(1), lambda g: {"x": rn(g, 16, 333, 24)}, bwd=["x"], doc="mean over the middle dim")
add("norm_cols", lambda x: torch.linalg.vector_norm(x, dim=0), X, prep=lambda x: {"x": x[:, ::2]}, bwd=["x"],
    doc="column norms of a strided view")

# ---- large reductions ----------------------------------------------------------------------------------------
add("sum_2e21", lambda x: x.sum().reshape(1), lambda g: {"x": rn(g, 1 << 21) + 0.25}, doc="sum of 2^21 elements")
add("var_mean_2e21", lambda x: torch.stack(torch.var_mean(x)), lambda g: {"x": rn(g, 1 << 21) * 3 + 7},
    doc="var_mean of 2^21 elements (correction 1)")
add("norm_2e21", lambda x: torch.linalg.vector_norm(x).reshape(1), lambda g: {"x": rn(g, 1 << 21)}, bwd=["x"],
    doc="2-norm of 2^21 elements")
add("softmax_65536", lambda x: torch.softmax(x, -1), lambda g: {"x": rn(g, 4, 65536, scale=4)}, bwd=["x"],
    doc="softmax over rows of 65536")
add("log_softmax_vocab128k", lambda x: torch.log_softmax(x, -1), lambda g: {"x": rn(g, 4, 128256, scale=4)},
    doc="log_softmax over the Llama-3 vocabulary (128256)")


def _ce_make(vocab, rows=8):
    def make(g):
        t = torch.randint(0, vocab, (rows,), generator=g)
        t[1] = -100
        return {"x": rn(g, rows, vocab, scale=4), "t": t}
    return make


add("ce_vocab128k", lambda x, t: F.cross_entropy(x, t).reshape(1), _ce_make(128256), bwd=["x"],
    doc="cross_entropy mean over the Llama-3 vocabulary, one ignored row")
add("ce_vocab32k_smooth", lambda x, t: F.cross_entropy(x, t, label_smoothing=0.1).reshape(1), _ce_make(32000, 16),
    bwd=["x"], doc="cross_entropy label_smoothing 0.1, vocab 32000")
add("logsumexp_2e20", lambda x: torch.logsumexp(x, 0).reshape(1), lambda g: {"x": rn(g, 1 << 20, scale=5)},
    doc="logsumexp of 2^20 elements")
add("cumsum_65536", lambda x: torch.cumsum(x, -1), lambda g: {"x": rn(g, 2, 65536)}, doc="cumsum over 65536")
add("argmax_2e20_ties", lambda x: torch.argmax(x).reshape(1).to(torch.float32),
    lambda g: {"x": torch.round(rn(g, 1 << 20) * 2) / 2}, doc="argmax of 2^20 elements with ties (first index)")
add("mean_rows_2e18", lambda x: x.mean(1), lambda g: {"x": rn(g, 4, 1 << 18) + 1}, doc="mean over rows of 2^18")

# ---- NaN / inf semantics -------------------------------------------------------------------------------------


def _masked(g):
    x = rn(g, 64, 128, scale=3)
    x[::7] = float("-inf")       # fully masked rows
    x[3::5, ::4] = float("-inf")  # partially masked rows
    return {"x": x}


add("softmax_masked_rows", lambda x: torch.softmax(x, -1), _masked, doc="softmax with fully / partially -inf rows")
add("log_softmax_masked", lambda x: torch.log_softmax(x, -1), _masked, doc="log_softmax with -inf entries")
add("logsumexp_all_ninf", lambda x: torch.logsumexp(x, -1), _masked, doc="logsumexp with all -inf rows")


def _with_nan(g):
    x = rn(g, 64, 128)
    x[::9, 5] = float("nan")
    return {"x": x}


add("amax_nan", lambda x: torch.amax(x, -1), _with_nan, doc="amax with NaN entries (propagates)")
add("max_dim_nan", lambda x: torch.max(x, -1).values, _with_nan, doc="max(dim) with NaN entries")
add("argmax_nan", lambda x: torch.argmax(x, -1).to(torch.float32), _with_nan, doc="argmax with NaN entries")
add("sort_nan", lambda x: torch.sort(x, -1).values, _with_nan, doc="sort with NaN (NaN last)")
add("sort_nan_idx", lambda x: torch.sort(x, dim=-1, stable=True).indices.to(torch.float32), _with_nan,
    doc="stable sort indices with NaN")
add("clamp_nan", lambda x: torch.clamp(x, -0.5, 0.5), _with_nan, doc="clamp propagates NaN")
add("maximum_nan", lambda x, y: torch.maximum(x, y), lambda g: {**_with_nan(g), "y": rn(g, 64, 128)},
    doc="maximum propagates NaN")
add("fmax_nan", lambda x, y: torch.fmax(x, y), lambda g: {**_with_nan(g), "y": rn(g, 64, 128)}, doc="fmax ignores NaN")
add("div_zero", lambda x, y: x / y, lambda g: {"x": torch.cat([rn(g, 32, 64), torch.zeros(32, 64)]),
                                                 "y": torch.cat([torch.zeros(32, 64), torch.zeros(16, 64), rn(g, 16, 64)])},
    doc="x / 0 and 0 / 0")
add("nansum", lambda x: torch.nansum(x, -1), _with_nan, doc="nansum")
add("nanmean", lambda x: torch.nanmean(x, -1), _with_nan, doc="nanmean")
add("ce_all_ignored", lambda x, t: F.cross_entropy(x, t).reshape(1),
    lambda g: {"x": rn(g, 8, 50), "t": torch.full((8,), -100)}, doc="cross_entropy mean with every target ignored (0/0)")
add("var_single", lambda x: torch.var(x, -1), lambda g: {"x": rn(g, 64, 1)}, doc="var (correction 1) of one element")
add("xlogx_zero", lambda x: x * torch.log(x), lambda g: {"x": torch.cat([torch.zeros(8, 64), torch.rand(56, 64, generator=g)])},
    doc="x log x at 0 (0 * -inf = NaN)")
add("pow_edges", lambda x: torch.pow(x, -1.5) + torch.pow(-x, 0.5),
    lambda g: {"x": torch.cat([torch.zeros(8, 64), torch.rand(56, 64, generator=g) + 0.1])}, doc="pow at 0 and negative base")

# ---- integers and promotion ----------------------------------------------------------------------------------
add("int32_sum_overflow", lambda x: x.sum(-1), lambda g: {"x": torch.randint(2**30, 2**31 - 1, (16, 64), generator=g, dtype=torch.int32)},
    doc="int32 sum (promotes to int64)")
add("int8_mul_wrap", lambda x, y: x * y, lambda g: {"x": torch.randint(-128, 127, (64, 64), generator=g, dtype=torch.int8),
                                                    "y": torch.randint(-128, 127, (64, 64), generator=g, dtype=torch.int8)},
    doc="int8 multiply wraps")
add("int_floordiv_neg", lambda x, y: torch.floor_divide(x, y), lambda g: {"x": torch.randint(-1000, 1000, (64, 64), generator=g),
                                                                         "y": torch.randint(1, 50, (64, 64), generator=g) * (torch.randint(0, 2, (64, 64), generator=g) * 2 - 1)},
    doc="integer floor_divide with negatives")
add("int_remainder_neg", lambda x, y: torch.remainder(x, y), lambda g: {"x": torch.randint(-1000, 1000, (64, 64), generator=g),
                                                                       "y": torch.randint(1, 50, (64, 64), generator=g) * (torch.randint(0, 2, (64, 64), generator=g) * 2 - 1)},
    doc="integer remainder with negatives")
add("int_fmod_neg", lambda x, y: torch.fmod(x, y), lambda g: {"x": torch.randint(-1000, 1000, (64, 64), generator=g),
                                                             "y": torch.randint(1, 50, (64, 64), generator=g) * (torch.randint(0, 2, (64, 64), generator=g) * 2 - 1)},
    doc="integer fmod with negatives")
add("int32_cumsum", lambda x: torch.cumsum(x, -1), lambda g: {"x": torch.randint(2**28, 2**30, (8, 256), generator=g, dtype=torch.int32)},
    doc="int32 cumsum (promotes to int64)")
add("bool_mask_mean", lambda x, m: (x * m).sum(-1) / m.sum(-1), lambda g: {"x": rn(g, 64, 128), "m": torch.rand(64, 128, generator=g) > 0.3},
    doc="masked mean with a bool mask")
add("uint8_int8_add", lambda x, y: x + y, lambda g: {"x": torch.randint(0, 255, (64, 64), generator=g, dtype=torch.uint8),
                                                     "y": torch.randint(-128, 127, (64, 64), generator=g, dtype=torch.int8)},
    doc="uint8 + int8 (promotes to int16)")
add("int64_to_float", lambda x: x.to(torch.float32) * 1.0, lambda g: {"x": torch.randint(2**40, 2**50, (64, 64), generator=g)},
    doc="int64 > 2^24 to float32 (rounds)")
add("rshift_neg", lambda x: x >> 3, lambda g: {"x": torch.randint(-10000, 10000, (64, 64), generator=g, dtype=torch.int32)},
    doc="arithmetic right shift of negatives")

# ---- mutation through views and aliasing (class 2) -----------------------------------------------------------


def _mut_view(x):
    x[:, :8].mul_(2.0)
    return x.sum(1)


def _mut_transpose(x):
    y = x.t()
    y.add_(torch.arange(y.shape[1], device=y.device, dtype=y.dtype))
    return x * 2


def _mut_then_read(x, y):
    x.add_(y)
    return x.sum() * y


def _index_put_acc(x, idx, v):
    x.index_put_((idx,), v, accumulate=True)
    return x.sum(0)


def _index_add(x, idx, v):
    return x.index_add(0, idx, v, alpha=0.5)


def _scatter_add(x, idx, v):
    return x.scatter_add(1, idx, v)


def _diag_inplace(x):
    x.diagonal().mul_(3.0)
    return x @ torch.ones(x.shape[1], device=x.device, dtype=x.dtype)


def _two_views(x):
    a, b = x[:, :32], x[:, 32:]
    a.add_(b)
    b.mul_(a)
    return x.mean(0)


add("mut_slice_then_reduce", _mut_view, X, mutated=["x"], doc="x[:, :8].mul_(2) then x.sum(1)")
add("mut_transpose_view", _mut_transpose, X, mutated=["x"], doc="in-place add through x.t()")
add("mut_then_read", _mut_then_read, lambda g: {"x": rn(g, 64, 64), "y": rn(g, 64, 64)}, mutated=["x"],
    doc="x.add_(y) then x.sum() * y")
add("index_put_accumulate_dups", _index_put_acc,
    lambda g: {"x": rn(g, 32, 64), "idx": torch.randint(0, 32, (200,), generator=torch.Generator().manual_seed(5)),
               "v": rn(g, 200, 64)},
    mutated=["x"], doc="index_put_ accumulate with duplicate indices (fixed index set)")
add("index_add_dups", _index_add,
    lambda g: {"x": rn(g, 32, 64), "idx": torch.randint(0, 32, (200,), generator=g), "v": rn(g, 200, 64)},
    doc="index_add alpha 0.5 with duplicates")
add("scatter_add_dups", _scatter_add,
    lambda g: {"x": rn(g, 32, 64), "idx": torch.randint(0, 64, (32, 200), generator=g), "v": rn(g, 32, 200)},
    doc="scatter_add with duplicate indices")
add("diag_inplace", _diag_inplace, lambda g: {"x": rn(g, 64, 64)}, mutated=["x"], doc="diagonal().mul_ then matvec")
add("two_views_inplace", _two_views, lambda g: {"x": rn(g, 48, 64)}, mutated=["x"],
    doc="a += b; b *= a on two halves of one tensor, then mean")

# ---- common training code ------------------------------------------------------------------------------------
_SHAPES = [(256, 128), (128,), (64, 64, 3), (1000,), (17, 33)]


def _grads(g):
    return {f"g{i}": rn(g, *s) * (10 ** (i - 2)) for i, s in enumerate(_SHAPES)}


def _clip(norm_type, max_norm=1.0):
    def fn(**t):
        grads = [t[f"g{i}"] for i in range(len(_SHAPES))]
        total = torch.nn.utils.get_total_norm(grads, norm_type, foreach=True)
        clipped = list(grads)
        coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
        out = {"total_norm": total.reshape(1)}
        for i, gr in enumerate(clipped):
            out[f"clipped{i}"] = gr * coef
        return out
    return fn


add("total_norm_2", _clip(2.0), _grads, doc="get_total_norm(norm 2, foreach) + clip coefficient over 5 tensors")
add("total_norm_inf", _clip(float("inf")), _grads, doc="get_total_norm(norm inf, foreach) + clip coefficient")
add("total_norm_1", _clip(1.0), _grads, doc="get_total_norm(norm 1, foreach) + clip coefficient")


def _ema(**t):
    emas = [t[f"e{i}"] for i in range(3)]
    params = [t[f"p{i}"] for i in range(3)]
    torch._foreach_lerp_(emas, params, 1 - 0.999)
    return {f"ema{i}": e for i, e in enumerate(emas)}


add("ema_foreach_lerp", _ema, lambda g: {**{f"e{i}": rn(g, 64, 64) for i in range(3)}, **{f"p{i}": rn(g, 64, 64) for i in range(3)}},
    doc="EMA update with _foreach_lerp_ (weight 1 - 0.999)")
