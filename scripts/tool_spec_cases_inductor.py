"""PyTorch Inductor (torch.compile) Triton kernels for tool_spec_check: compiled function against eager semantics.

Each case is a small PyTorch function on float32 CUDA inputs.  The implementation is ``torch.compile(fn)`` (Inductor
lowers it to Triton kernels, which the recorder captures); the specification f is the same function run eagerly in
float64 on the same inputs (PyTorch's documented semantics are dtype-generic), taken with a declared bound
2^-40 * max|f|.  Backward cases compile forward + backward (AOT autograd) and compare the input gradients with
float64 eager autograd for the same upstream gradient.  Environment: ka_main (torch 2.10.0, triton 3.6.0).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tool_spec_check import Case, f64_point_spec


class OpCase(Case):
    implementation = "torch.compile(fn) -> Inductor Triton kernels"
    spec_bound = "float64 eager evaluation of the same function, declared bound 2^-40 * max|f|"

    def __init__(self, name, fn, make, backward=False, grad_of=None, doc=""):
        self.name, self.fn, self.make, self.backward = name, fn, make, backward
        self.grad_of = grad_of or []
        self.specification = f"eager PyTorch semantics of: {doc or name} (float64)" + (" ; backward" if backward else "")

    def setup(self):
        self._compiled = torch.compile(self.fn, dynamic=False)
        self.launch(self.inputs(10_000))
        torch.cuda.synchronize()

    def inputs(self, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        inp = self.make(g)
        out = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}
        if self.backward:
            with torch.no_grad():
                y = self.fn(**{k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v)
                               for k, v in out.items()})
            out["_g"] = torch.randn(y.shape, generator=g).cuda()
        return out

    def _args(self, inp, dtype=None, leaves=None):
        args = {}
        for k, v in inp.items():
            if k == "_g":
                continue
            if torch.is_tensor(v) and v.is_floating_point():
                v = v.to(dtype) if dtype is not None else v.clone()
                if leaves is not None and k in self.grad_of:
                    v.requires_grad_(True)
                    leaves[k] = v
            args[k] = v
        return args

    def launch(self, inp):
        if not self.backward:
            return {"out": self._compiled(**self._args(inp))}
        leaves = {}
        y = self._compiled(**self._args(inp, leaves=leaves))
        y.backward(inp["_g"])
        return {f"d{k}": v.grad for k, v in leaves.items()}

    def spec(self, inp):
        if not self.backward:
            with torch.no_grad():
                y = self.fn(**self._args(inp, dtype=torch.float64))
            return {"out": f64_point_spec(y.cpu().numpy())}
        leaves = {}
        y = self.fn(**self._args(inp, dtype=torch.float64, leaves=leaves))
        grads = torch.autograd.grad(y, list(leaves.values()), inp["_g"].double())
        return {f"d{k}": f64_point_spec(gr.detach().cpu().numpy()) for k, gr in zip(leaves, grads)}


def rn(g, *shape, scale=1.0):
    return torch.randn(*shape, generator=g) * scale


def _case(name, fn, make, doc="", bwd=None):
    out = [OpCase(f"ind_{name}", fn, make, doc=doc)]
    if bwd:
        out.append(OpCase(f"ind_{name}_bwd", fn, make, backward=True, grad_of=bwd, doc=doc))
    return out


R, D = 64, 384
CASES = []
# normalization
CASES += _case("rms_norm", lambda x, w: F.rms_norm(x, (D,), w, 1e-6),
               lambda g: {"x": rn(g, R, D), "w": 1 + 0.1 * rn(g, D)}, "F.rms_norm eps 1e-6", bwd=["x", "w"])
CASES += _case("layer_norm", lambda x, w, b: F.layer_norm(x, (D,), w, b, 1e-5),
               lambda g: {"x": rn(g, R, D, scale=3) + 1, "w": 1 + 0.1 * rn(g, D), "b": 0.1 * rn(g, D)},
               "F.layer_norm", bwd=["x", "w", "b"])
CASES += _case("group_norm", lambda x, w, b: F.group_norm(x, 8, w, b, 1e-5),
               lambda g: {"x": rn(g, 4, 32, 96) + 0.5, "w": 1 + 0.1 * rn(g, 32), "b": 0.1 * rn(g, 32)},
               "F.group_norm 8 groups", bwd=["x", "w", "b"])
CASES += _case("normalize", lambda x: F.normalize(x, p=2.0, dim=-1, eps=1e-3),
               lambda g: {"x": torch.cat([rn(g, R // 2, D), 1e-5 * rn(g, R // 2, D)])},
               "F.normalize eps 1e-3 (half the rows below eps)", bwd=["x"])
CASES += _case("cosine_similarity", lambda x, y: F.cosine_similarity(x, y, dim=-1, eps=1e-4),
               lambda g: {"x": torch.cat([rn(g, R // 2, D), 1e-4 * rn(g, R // 2, D)]), "y": rn(g, R, D)},
               "F.cosine_similarity eps 1e-4 (small rows)", bwd=["x", "y"])
CASES += _case("var_unbiased", lambda x: torch.var(x, dim=-1, correction=1), lambda g: {"x": rn(g, R, D) + 3},
               "torch.var correction=1", bwd=["x"])
# softmax family
CASES += _case("softmax", lambda x: torch.softmax(x, -1), lambda g: {"x": rn(g, R, D, scale=4)}, "softmax", bwd=["x"])
CASES += _case("log_softmax", lambda x: torch.log_softmax(x, -1), lambda g: {"x": rn(g, R, D, scale=4)},
               "log_softmax", bwd=["x"])
CASES += _case("logsumexp", lambda x: torch.logsumexp(x, -1), lambda g: {"x": rn(g, R, D, scale=4)}, "logsumexp",
               bwd=["x"])
CASES += _case("logcumsumexp", lambda x: torch.logcumsumexp(x, -1), lambda g: {"x": rn(g, R, 256, scale=2)},
               "logcumsumexp", bwd=["x"])
CASES += _case("cumsum", lambda x: torch.cumsum(x, -1), lambda g: {"x": rn(g, R, 256)}, "cumsum", bwd=["x"])
# activations
CASES += _case("gelu_erf", lambda x: F.gelu(x), lambda g: {"x": rn(g, R, D, scale=3)}, "gelu exact", bwd=["x"])
CASES += _case("gelu_tanh", lambda x: F.gelu(x, approximate="tanh"), lambda g: {"x": rn(g, R, D, scale=3)},
               "gelu tanh", bwd=["x"])
CASES += _case("silu", lambda x: F.silu(x), lambda g: {"x": rn(g, R, D, scale=3)}, "silu", bwd=["x"])
CASES += _case("mish", lambda x: F.mish(x), lambda g: {"x": rn(g, R, D, scale=3)}, "mish", bwd=["x"])
CASES += _case("softplus", lambda x: F.softplus(x, beta=2.0, threshold=5.0), lambda g: {"x": rn(g, R, D, scale=4)},
               "softplus beta 2 threshold 5", bwd=["x"])
CASES += _case("elu", lambda x: F.elu(x, alpha=0.7), lambda g: {"x": rn(g, R, D, scale=3)}, "elu 0.7", bwd=["x"])
CASES += _case("selu", lambda x: F.selu(x), lambda g: {"x": rn(g, R, D, scale=3)}, "selu", bwd=["x"])
CASES += _case("logsigmoid", lambda x: F.logsigmoid(x), lambda g: {"x": rn(g, R, D, scale=6)}, "logsigmoid",
               bwd=["x"])
CASES += _case("hardswish", lambda x: F.hardswish(x), lambda g: {"x": rn(g, R, D, scale=4)}, "hardswish", bwd=["x"])
CASES += _case("swiglu", lambda a, b: F.silu(a) * b, lambda g: {"a": rn(g, R, D, scale=3), "b": rn(g, R, D)},
               "silu(a) * b", bwd=["a", "b"])
# losses
V = 1000


def _ce_make(g):
    t = torch.randint(0, V, (R,), generator=g)
    t[::5] = -100
    return {"x": rn(g, R, V, scale=3), "t": t, "w": torch.rand(V, generator=g) + 0.5}


CASES += _case("ce_none", lambda x, t, w: F.cross_entropy(x, t, reduction="none"), _ce_make,
               "cross_entropy reduction none, ignore_index -100", bwd=["x"])
CASES += _case("ce_mean_smooth", lambda x, t, w: F.cross_entropy(x, t, label_smoothing=0.1).reshape(1), _ce_make,
               "cross_entropy mean, label_smoothing 0.1", bwd=["x"])
CASES += _case("ce_weighted_smooth", lambda x, t, w: F.cross_entropy(x, t, weight=w, label_smoothing=0.1).reshape(1),
               _ce_make, "cross_entropy mean, class weights, label_smoothing 0.1", bwd=["x"])
CASES += _case("nll_weighted", lambda x, t, w: F.nll_loss(torch.log_softmax(x, -1), t, weight=w).reshape(1),
               _ce_make, "nll_loss(log_softmax) mean with class weights", bwd=["x"])
CASES += _case("kl_div_logtarget", lambda x, y: F.kl_div(torch.log_softmax(x, -1), torch.log_softmax(y, -1),
                                                         reduction="batchmean", log_target=True).reshape(1),
               lambda g: {"x": rn(g, R, 256, scale=2), "y": rn(g, R, 256, scale=2)}, "kl_div batchmean log_target",
               bwd=["x"])
CASES += _case("bce_logits_posweight", lambda x, y, pw: F.binary_cross_entropy_with_logits(x, y, pos_weight=pw,
                                                                                           reduction="none"),
               lambda g: {"x": rn(g, R, 64, scale=4), "y": torch.rand(R, 64, generator=g), "pw": torch.rand(64, generator=g) * 3},
               "binary_cross_entropy_with_logits pos_weight", bwd=["x"])
CASES += _case("smooth_l1", lambda x, y: F.smooth_l1_loss(x, y, beta=0.5, reduction="none"),
               lambda g: {"x": rn(g, R, D), "y": rn(g, R, D)}, "smooth_l1 beta 0.5", bwd=["x"])
CASES += _case("huber", lambda x, y: F.huber_loss(x, y, delta=0.7, reduction="none"),
               lambda g: {"x": rn(g, R, D), "y": rn(g, R, D)}, "huber delta 0.7", bwd=["x"])
# pooling / resampling (edge semantics: ceil_mode, padding, align_corners)
CASES += _case("avg_pool2d_ceil_pad", lambda x: F.avg_pool2d(x, 3, stride=2, padding=1, ceil_mode=True,
                                                             count_include_pad=True),
               lambda g: {"x": rn(g, 2, 8, 17, 18)}, "avg_pool2d k3 s2 p1 ceil_mode count_include_pad", bwd=["x"])
CASES += _case("avg_pool2d_ceil_nopad", lambda x: F.avg_pool2d(x, 3, stride=2, padding=1, ceil_mode=True,
                                                               count_include_pad=False),
               lambda g: {"x": rn(g, 2, 8, 17, 18)}, "avg_pool2d ceil_mode, count_include_pad False", bwd=["x"])
CASES += _case("avg_pool2d_divisor", lambda x: F.avg_pool2d(x, 3, stride=2, padding=1, ceil_mode=True,
                                                            divisor_override=5),
               lambda g: {"x": rn(g, 2, 8, 17, 18)}, "avg_pool2d divisor_override 5", bwd=["x"])
CASES += _case("max_pool2d_ceil_dil", lambda x: F.max_pool2d(x, 3, stride=2, padding=1, dilation=2, ceil_mode=True),
               lambda g: {"x": rn(g, 2, 8, 17, 18)}, "max_pool2d dilation 2 ceil_mode", bwd=["x"])
CASES += _case("avg_pool1d_ceil", lambda x: F.avg_pool1d(x, 4, stride=3, padding=2, ceil_mode=True),
               lambda g: {"x": rn(g, 4, 8, 37)}, "avg_pool1d k4 s3 p2 ceil_mode", bwd=["x"])
CASES += _case("adaptive_avg_pool2d", lambda x: F.adaptive_avg_pool2d(x, (5, 7)),
               lambda g: {"x": rn(g, 2, 8, 17, 18)}, "adaptive_avg_pool2d (5, 7)", bwd=["x"])
CASES += _case("interp_bilinear", lambda x: F.interpolate(x, size=(23, 29), mode="bilinear", align_corners=False),
               lambda g: {"x": rn(g, 2, 4, 17, 18)}, "bilinear align_corners False", bwd=["x"])
CASES += _case("interp_bilinear_ac", lambda x: F.interpolate(x, size=(23, 29), mode="bilinear", align_corners=True),
               lambda g: {"x": rn(g, 2, 4, 17, 18)}, "bilinear align_corners True", bwd=["x"])
CASES += _case("interp_bicubic", lambda x: F.interpolate(x, size=(23, 29), mode="bicubic", align_corners=False),
               lambda g: {"x": rn(g, 2, 4, 17, 18)}, "bicubic align_corners False", bwd=["x"])
CASES += _case("interp_nearest_exact", lambda x: F.interpolate(x, scale_factor=1.7, mode="nearest-exact"),
               lambda g: {"x": rn(g, 2, 4, 17, 18)}, "nearest-exact scale 1.7", bwd=["x"])
CASES += _case("interp_area", lambda x: F.interpolate(x, size=(7, 11), mode="area"),
               lambda g: {"x": rn(g, 2, 4, 17, 18)}, "area (7, 11)", bwd=["x"])
CASES += _case("grid_sample", lambda x, grid: F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                                                            align_corners=False),
               lambda g: {"x": rn(g, 2, 4, 17, 18), "grid": torch.rand(2, 9, 11, 2, generator=g) * 2.4 - 1.2},
               "grid_sample bilinear border", bwd=["x", "grid"])
CASES += _case("pad_reflect", lambda x: F.pad(x, (3, 2, 1, 4), mode="reflect"), lambda g: {"x": rn(g, 2, 4, 9, 10)},
               "pad reflect", bwd=["x"])
CASES += _case("pad_circular", lambda x: F.pad(x, (3, 2, 1, 4), mode="circular"), lambda g: {"x": rn(g, 2, 4, 9, 10)},
               "pad circular", bwd=["x"])
