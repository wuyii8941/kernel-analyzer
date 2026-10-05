"""PyTorch OpInfo database x Inductor for tool_spec_check: systematic screening of torch.compile lowerings.

For each OpInfo (torch.testing._internal.common_methods_invocations.op_db) that supports float32 and float64 on CUDA,
up to SAMPLES_PER_OP of its sample inputs become cases.  Seed s regenerates the op's sample inputs under
torch.manual_seed(s) and takes the same sample index, so the values change while the structure (shapes, kwargs) and
the op's input domain (positive inputs for log, [-1, 1] for acos, ...) are kept; a seed whose sample changes structure
is skipped by the case (its inputs are re-drawn from the next seed).

Implementation: torch.compile(op, dynamic=False) on float32 CUDA tensors -> Inductor Triton kernels.
Specification: the same op, eager, on the same inputs cast to float64 (integer / bool tensors unchanged), declared
bound 2^-40 * max|f|.  Ops whose float64 semantics differ by design (dtype-dependent constants such as
finfo(dtype).eps defaults) show up as candidates and are triaged by hand.

    PYTHONPATH=src python scripts/tool_spec_check.py --group opinfo --case list
"""

from __future__ import annotations

import os
import re

import torch

from tool_spec_check import Case, f64_point_spec

SAMPLES_PER_OP = 3
MAX_ELEMENTS = 200_000

# random, allocation-only, or not lowered to Triton (cuBLAS / cuSOLVER / cuFFT / cuDNN): nothing to evaluate
_SKIP_PREFIX = ("as_strided", "_softmax_backward_data", "_log_softmax_backward_data", "linalg.", "fft.", "nn.functional.conv", "nn.functional.scaled_dot_product_attention", "sparse.",
                "_refs.", "_native_batch_norm_legit", "nn.functional.dropout", "nn.functional.feature_alpha_dropout",
                "nn.functional.alpha_dropout", "nn.functional.rrelu", "nn.functional.multi_head_attention_forward",
                "nn.functional.gaussian_nll_loss", "special.airy", "to_sparse", "torch._scaled_mm", "jiterator",
                "signal.windows")
_SKIP_NAMES = {"bernoulli", "multinomial", "normal", "uniform", "exponential", "cauchy", "geometric", "log_normal",
               "poisson", "randint", "randint_like", "rand_like", "randn", "randn_like", "rand", "empty", "empty_like",
               "new_empty", "new_empty_strided", "empty_strided", "empty_permuted", "nonzero", "nonzero_static",
               "unique", "unique_consecutive", "masked_select", "item", "tolist", "matmul", "mm", "bmm", "addmm",
               "addbmm", "baddbmm", "addmv", "mv", "addr", "dot", "vdot", "inner", "outer", "ger", "tensordot",
               "einsum", "chain_matmul", "cholesky", "cholesky_inverse", "cholesky_solve", "lu", "lu_solve",
               "lu_unpack", "triangular_solve", "geqrf", "ormqr", "pinverse", "svd", "svd_lowrank", "pca_lowrank",
               "qr", "symeig", "det", "logdet", "slogdet", "inverse", "matrix_exp", "nn.functional.linear",
               "nn.functional.bilinear", "cdist", "pdist", "stft", "istft", "nn.functional.ctc_loss",
               "nn.functional.embedding_bag", "searchsorted", "histc", "histogram", "histogramdd", "bincount",
               "argwhere", "combinations", "cov", "corrcoef", "sort", "argsort", "msort", "topk", "kthvalue",
               "median", "nanmedian", "mode", "nn.functional.unfold", "nn.functional.fold", "repeat_interleave",
               "nn.functional.interpolate", "special.zeta", "nn.functional.grid_sample",
               "nn.functional.max_unpool1d", "nn.functional.max_unpool2d", "nn.functional.max_unpool3d"}


_RANDOM = {"normal", "nn.functional.alpha_dropout", "nn.functional.dropout", "nn.functional.dropout2d",
           "nn.functional.dropout3d", "nn.functional.feature_alpha_dropout", "new_empty_strided", "svd_lowrank",
           "linalg.eig", "histogram", "histogramdd", "einsum", "addmv", "nn.functional.bilinear", "corrcoef",
           "nn.functional.conv2d", "nn.functional.conv_transpose1d", "nn.functional.conv_transpose2d",
           "nn.functional.conv_transpose3d", "linalg.householder_product", "nn.functional.fractional_max_pool2d",
           "nn.functional.fractional_max_pool3d"}


def _floating_inputs(sample):
    return [t for t in _tensors(sample) if t.is_floating_point()]


def _tensors(sample):
    out = []

    def walk(x):
        if torch.is_tensor(x):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            for y in x.values():
                walk(y)

    walk(sample.input)
    walk(sample.args)
    walk(sample.kwargs)
    return out


def _structure(sample):
    def sig(x):
        if torch.is_tensor(x):
            return ("T", tuple(x.shape), str(x.dtype), tuple(x.stride()))
        if isinstance(x, (list, tuple)):
            return (type(x).__name__, tuple(sig(y) for y in x))
        if isinstance(x, dict):
            return ("dict", tuple((k, sig(v)) for k, v in sorted(x.items())))
        if isinstance(x, float):
            return ("f",)  # values of python scalars may be drawn from the seed too
        return ("v", repr(x))

    return (sig(sample.input), sig(sample.args), sig(sample.kwargs))


def _options(args):
    """repr of the non-tensor structure of the positional arguments (tensors as their dtype only)."""
    def r(x):
        if torch.is_tensor(x):
            return f"T{x.dim()}"
        if isinstance(x, (list, tuple)):
            return "(" + ",".join(r(y) for y in x) + ")"
        if isinstance(x, float):
            return "f"
        return repr(x)
    return r(args)


def _replace(x, it):
    if torch.is_tensor(x):
        return next(it)
    if isinstance(x, list):
        return [_replace(y, it) for y in x]
    if isinstance(x, tuple):
        return tuple(_replace(y, it) for y in x)
    if isinstance(x, dict):
        return {k: _replace(v, it) for k, v in x.items()}
    return x


def _outputs(y):
    flat = []

    def walk(x):
        if torch.is_tensor(x):
            flat.append(x)
        elif isinstance(x, (list, tuple)):
            for z in x:
                walk(z)

    walk(y)
    return flat


class OpInfoCase(Case):
    implementation = "torch.compile(op, dynamic=False) on float32 CUDA tensors -> Inductor Triton kernels"
    spec_bound = "float64 eager evaluation of the same op on the same inputs, declared bound 2^-40 * max|f|"

    def __init__(self, name, op, index, structure, doc):
        self.name, self.op, self.index, self.structure = name, op, index, structure
        self.specification = f"eager torch semantics of {op.name} (variant '{op.variant_test_name}') in float64: {doc}"

    def _sample(self, seed):
        for s in range(seed, seed + 50):
            torch.manual_seed(s)
            samples = list(self.op.sample_inputs("cuda", torch.float32, requires_grad=False))
            if self.index < len(samples) and _structure(samples[self.index]) == self.structure:
                return samples[self.index]
        raise RuntimeError("no sample with the case structure")

    def _call(self, sample, fn, dtype=None):
        tens = _tensors(sample)
        if dtype is not None:
            tens = [t.to(dtype) if t.is_floating_point() else t for t in tens]
        else:
            tens = [t.clone() for t in tens]
        it = iter(tens)
        inp = _replace(sample.input, it)
        args = _replace(sample.args, it)
        kwargs = _replace(sample.kwargs, it)
        return fn(inp, *args, **kwargs)

    def setup(self):
        op = self.op.op

        def f(inp, *args, **kwargs):
            return op(inp, *args, **kwargs)

        # OPINFO_DYNAMIC=1: symbolic sizes (0/1 specialized), the path automatic-dynamic recompilation takes
        self._compiled = torch.compile(f, dynamic=os.environ.get("OPINFO_DYNAMIC") == "1")
        self.launch(self.inputs(10_000))
        torch.cuda.synchronize()

    def inputs(self, seed):
        sample = self._sample(seed)
        # the tensors are listed too, so the provenance check (torch_intermediates) recognises them as case inputs
        return {"sample": sample, "tensors": _tensors(sample)}

    def launch(self, inp):
        y = self._call(inp["sample"], self._compiled)
        return {f"out{i}": t for i, t in enumerate(_outputs(y))}

    def spec(self, inp):
        with torch.no_grad():
            y = self._call(inp["sample"], self.op.op, torch.float64)
        return {f"out{i}": f64_point_spec(t.detach().cpu().double().numpy() if t.dtype != torch.bool
                                          else t.detach().cpu().numpy().astype("float64"))
                for i, t in enumerate(_outputs(y))}


class OpInfoBwdCase(OpInfoCase):
    """Backward of the same op: gradients of sum_i <g_i, y_i> w.r.t. every floating input (g drawn per seed)."""

    def __init__(self, name, op, index, structure, doc):
        super().__init__(name, op, index, structure, doc)
        self.specification += " ; backward (float64 eager autograd)"

    def _sample(self, seed):
        # requires_grad=True samples: the op's generator marks exactly the differentiable tensors (not, e.g., the
        # class weights of nll_loss); only those become leaves
        for s in range(seed, seed + 50):
            torch.manual_seed(s)
            samples = list(self.op.sample_inputs("cuda", torch.float32, requires_grad=True))
            if self.index < len(samples) and _structure(samples[self.index]) == self.structure:
                return samples[self.index]
        # some generators build a different sample list with requires_grad=True (masked ops): fall back to the
        # requires_grad=False sample with every floating tensor differentiable
        sample = super()._sample(seed)
        for t in _tensors(sample):
            if t.is_floating_point():
                t.requires_grad_(True)
        return sample

    def inputs(self, seed):
        sample = self._sample(seed)
        with torch.no_grad():
            y = self._call(sample, self.op.op, torch.float64)
        g = torch.Generator(device="cpu").manual_seed(seed)
        # float32 upstream gradients (exact in float64): the compiled side uses them as they are, so they are
        # recognised as case inputs by the provenance check
        gs = [torch.randn(t.shape, generator=g, dtype=torch.float32).cuda() for t in _outputs(y)
              if t.is_floating_point()]
        return {"sample": sample, "tensors": _tensors(sample), "g": gs}

    def _grads(self, sample, fn, gs, dtype):
        tens = _tensors(sample)
        leaves = []
        conv = []
        for t in tens:
            if t.is_floating_point():
                differentiable = t.requires_grad
                t = (t.to(dtype) if dtype is not None else t.clone()).detach()
                if differentiable:
                    t.requires_grad_(True)
                    leaves.append(t)
            conv.append(t)
        it = iter(conv)
        y = fn(_replace(sample.input, it), *_replace(sample.args, it), **_replace(sample.kwargs, it))
        outs = [t for t in _outputs(y) if t.is_floating_point() and t.requires_grad]
        if not outs:
            raise RuntimeError("no differentiable output")
        gr = torch.autograd.grad(outs, leaves, [g if g.dtype == o.dtype else g.to(o.dtype) for g, o in zip(gs, outs)],
                                 allow_unused=True)
        return {f"d{i}": x for i, x in enumerate(gr) if x is not None}

    def launch(self, inp):
        return self._grads(inp["sample"], self._compiled, inp["g"], None)

    def spec(self, inp):
        return {k: f64_point_spec(v.detach().cpu().numpy())
                for k, v in self._grads(inp["sample"], self.op.op, inp["g"], torch.float64).items()}


def _build():
    import os

    from torch.testing._internal.common_methods_invocations import op_db

    # OPINFO_CASES (comma-separated case names): build only those ops, so a one-case process does not sample all
    wanted = [w for w in os.environ.get("OPINFO_CASES", "").split(",") if w]
    # OPINFO_SELECT=v2: samples grouped by all non-tensor options (positional too), up to 8 per op
    select_v2 = os.environ.get("OPINFO_SELECT") in ("v2", "onesample")
    # OPINFO_SELECT=onesample: only the ops that PyTorch's Inductor OpInfo test runs on one sample per op on CUDA
    # (inductor_one_sample["cuda"] in test/inductor/test_torchinductor_opinfo.py, main 2026-10-05), up to 12 samples
    one_sample = None
    if os.environ.get("OPINFO_SELECT") == "onesample":
        from pathlib import Path
        one_sample = set((Path(__file__).resolve().parent / "data/inductor_one_sample_cuda.txt").read_text().split())
    cases = []
    for op in op_db:
        full = op.name + ("." + op.variant_test_name if op.variant_test_name else "")
        if one_sample is not None:
            if full not in one_sample or op.name in _RANDOM:
                continue
        elif op.name in _SKIP_NAMES or op.name.startswith(_SKIP_PREFIX):
            continue
        base = re.sub(r"[^A-Za-z0-9]+", "_", op.name + ("_" + op.variant_test_name if op.variant_test_name else ""))
        if wanted and not any(re.fullmatch(rf"oib?_{re.escape(base)}_\d+", w) for w in wanted):
            continue
        try:
            if torch.float32 not in op.supported_dtypes("cuda") or torch.float64 not in op.supported_dtypes("cuda"):
                continue
            torch.manual_seed(0)
            samples = list(op.sample_inputs("cuda", torch.float32, requires_grad=False))
        except Exception:  # noqa: BLE001
            continue
        # group by (arity, kwargs); per group the largest sample up to MAX_ELEMENTS; groups with kwargs first
        groups = {}
        for i, smp in enumerate(samples):
            if not _floating_inputs(smp):
                continue
            n = sum(t.numel() for t in _tensors(smp))
            if n > MAX_ELEMENTS:
                continue
            kw = repr(sorted(smp.kwargs.items())) if isinstance(smp.kwargs, dict) else ""
            if select_v2:  # options passed positionally (pooling flags, dims, ...) distinguish samples too
                kw = kw + "|" + _options(smp.args)
            key = (len(_tensors(smp)), kw)
            if key not in groups or n > groups[key][0]:
                groups[key] = (n, i, smp)
        order = sorted(groups.items(), key=lambda kv: (kv[0][1] in ("", "[]", "[]|()"), -kv[1][0]))
        picked = []
        limit = 12 if one_sample is not None else (8 if select_v2 else SAMPLES_PER_OP)
        for _, (n, i, smp) in order[:limit]:
            picked.append((i, _structure(smp), f"sample {i}: {smp.summary() if hasattr(smp, 'summary') else ''}"[:300]))
        for i, st, doc in picked:
            cases.append(OpInfoCase(f"oi_{base}_{i}", op, i, st, doc))
            if op.supports_autograd:
                cases.append(OpInfoBwdCase(f"oib_{base}_{i}", op, i, st, doc))
    return cases


CASES = _build()
