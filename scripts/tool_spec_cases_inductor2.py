"""PyTorch Inductor, second batch: operators with edge semantics, and compiled optimizer steps.

Same contract as tool_spec_cases_inductor (torch.compile(fn) against the eager function in float64, declared bound
2^-40 * max|f|), plus:

* batch_norm in training mode: the output and the in-place running statistics (running_var uses the unbiased
  variance, n / (n - 1));
* amax / max backward with deliberate ties (amax splits the gradient evenly over ties; max(dim) sends it to the
  returned index);
* scatter_reduce with "mean" / "amax", include_self True / False;
* optimizer steps: torch.compile(opt.step) for one step from a populated state, the parameter and the state
  tensors measured (in place), against the eager optimizer (foreach=False) in float64 from the same state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tool_spec_check import Case, f64_point_spec
from tool_spec_cases_inductor import OpCase, _case, rn

R, D = 64, 256
CASES = []

# ---- losses with edge terms ------------------------------------------------------------------------------------


def _pois_make(g):
    t = torch.poisson(torch.rand(R, D, generator=g) * 4, generator=g)  # 0, 1 and > 1 targets
    return {"x": rn(g, R, D), "t": t}


CASES += _case("poisson_nll_full", lambda x, t: F.poisson_nll_loss(x, t, log_input=True, full=True, reduction="none"),
               _pois_make, "poisson_nll_loss log_input full (Stirling term for target > 1)", bwd=["x"])
CASES += _case("poisson_nll_rate", lambda x, t: F.poisson_nll_loss(x.abs() + 0.05, t, log_input=False, eps=1e-3,
                                                                  reduction="none"),
               _pois_make, "poisson_nll_loss log_input False eps 1e-3", bwd=["x"])
CASES += _case("gaussian_nll_full", lambda x, t, var: F.gaussian_nll_loss(x, t, var, full=True, eps=1e-3,
                                                                          reduction="none"),
               lambda g: {"x": rn(g, R, D), "t": rn(g, R, D),
                          "var": torch.cat([torch.rand(R // 2, D, generator=g) * 1e-3, torch.rand(R // 2, D, generator=g) + 0.1])},
               "gaussian_nll_loss full, eps 1e-3 (half the variances below eps)", bwd=["x", "var"])
CASES += _case("bce_probs_edges", lambda p, y: F.binary_cross_entropy(p, y, reduction="none"),
               lambda g: {"p": torch.cat([torch.rand(R - 4, D, generator=g), torch.zeros(2, D), torch.ones(2, D)]),
                          "y": torch.rand(R, D, generator=g)},
               "binary_cross_entropy with p in {0, 1} rows (log clamped at -100)", bwd=None)
CASES += _case("multi_margin_p2", lambda x, t, w: F.multi_margin_loss(x, t, p=2, margin=0.7, weight=w, reduction="none"),
               lambda g: {"x": rn(g, R, 40), "t": torch.randint(0, 40, (R,), generator=g), "w": torch.rand(40, generator=g) + 0.5},
               "multi_margin_loss p 2 margin 0.7 class weights", bwd=["x"])
CASES += _case("multilabel_soft_margin", lambda x, y, w: F.multilabel_soft_margin_loss(x, y, weight=w, reduction="none"),
               lambda g: {"x": rn(g, R, 40, scale=3), "y": (torch.rand(R, 40, generator=g) > 0.5).float(),
                          "w": torch.rand(40, generator=g) + 0.5}, "multilabel_soft_margin_loss with weights", bwd=["x"])
CASES += _case("triplet_swap", lambda a, p, n: F.triplet_margin_loss(a, p, n, margin=0.5, p=2, eps=1e-6, swap=True,
                                                                     reduction="none"),
               lambda g: {"a": rn(g, R, 32), "p": rn(g, R, 32), "n": rn(g, R, 32)}, "triplet_margin_loss swap",
               bwd=["a", "p", "n"])
CASES += _case("cosine_embedding", lambda x1, x2, y: F.cosine_embedding_loss(x1, x2, y, margin=0.2, reduction="none"),
               lambda g: {"x1": rn(g, R, 32), "x2": rn(g, R, 32), "y": torch.randint(0, 2, (R,), generator=g) * 2 - 1},
               "cosine_embedding_loss margin 0.2", bwd=["x1", "x2"])
CASES += _case("margin_ranking", lambda a, b, y: F.margin_ranking_loss(a, b, y, margin=0.3, reduction="none"),
               lambda g: {"a": rn(g, R * D), "b": rn(g, R * D), "y": torch.randint(0, 2, (R * D,), generator=g) * 2.0 - 1},
               "margin_ranking_loss margin 0.3", bwd=["a", "b"])
CASES += _case("hinge_embedding", lambda x, y: F.hinge_embedding_loss(x, y, margin=0.8, reduction="none"),
               lambda g: {"x": rn(g, R, D).abs(), "y": torch.randint(0, 2, (R, D), generator=g) * 2.0 - 1},
               "hinge_embedding_loss margin 0.8", bwd=["x"])
CASES += _case("soft_margin", lambda x, y: F.soft_margin_loss(x, y, reduction="none"),
               lambda g: {"x": rn(g, R, D, scale=3), "y": torch.randint(0, 2, (R, D), generator=g) * 2.0 - 1},
               "soft_margin_loss", bwd=["x"])

# ---- activations with parameters / edges -----------------------------------------------------------------------
CASES += _case("glu", lambda x: F.glu(x, dim=-1), lambda g: {"x": rn(g, R, D, scale=3)}, "glu", bwd=["x"])
CASES += _case("celu", lambda x: F.celu(x, alpha=0.5), lambda g: {"x": rn(g, R, D, scale=3)}, "celu 0.5", bwd=["x"])
CASES += _case("softsign", lambda x: F.softsign(x), lambda g: {"x": rn(g, R, D, scale=3)}, "softsign", bwd=["x"])
CASES += _case("prelu", lambda x, w: F.prelu(x, w), lambda g: {"x": rn(g, 8, 16, 64), "w": torch.rand(16, generator=g)},
               "prelu per channel", bwd=["x", "w"])
CASES += _case("hardtanh", lambda x: F.hardtanh(x, -0.7, 1.3), lambda g: {"x": rn(g, R, D, scale=2)},
               "hardtanh(-0.7, 1.3)", bwd=["x"])
CASES += _case("threshold", lambda x: F.threshold(x, 0.3, -2.0), lambda g: {"x": rn(g, R, D)},
               "threshold 0.3 value -2", bwd=["x"])
CASES += _case("softshrink", lambda x: F.softshrink(x, 0.4), lambda g: {"x": rn(g, R, D)}, "softshrink 0.4", bwd=["x"])
CASES += _case("tanhshrink", lambda x: F.tanhshrink(x), lambda g: {"x": rn(g, R, D, scale=2)}, "tanhshrink", bwd=["x"])
CASES += _case("logit_eps", lambda x: torch.logit(x, eps=0.05), lambda g: {"x": torch.rand(R, D, generator=g)},
               "logit eps 0.05 (input clamped to [eps, 1 - eps])", bwd=["x"])
CASES += _case("local_response_norm", lambda x: F.local_response_norm(x, 5, alpha=1e-3, beta=0.75, k=2.0),
               lambda g: {"x": rn(g, 4, 16, 9, 9, scale=3)}, "local_response_norm size 5", bwd=["x"])
CASES += _case("instance_norm", lambda x, w, b: F.instance_norm(x, weight=w, bias=b, eps=1e-5),
               lambda g: {"x": rn(g, 4, 16, 50) + 1, "w": 1 + 0.1 * rn(g, 16), "b": 0.1 * rn(g, 16)},
               "instance_norm", bwd=["x", "w", "b"])

# ---- reductions with ties / norms ------------------------------------------------------------------------------


def _ties(g):
    return {"x": torch.round(rn(g, R, D) * 2) / 2}  # many exact ties


CASES += _case("amax_ties", lambda x: torch.amax(x, dim=-1), _ties, "amax with ties (gradient split evenly)", bwd=["x"])
CASES += _case("max_dim_ties", lambda x: torch.max(x, dim=-1).values, _ties, "max(dim).values with ties", bwd=["x"])
CASES += _case("vector_norm_p3", lambda x: torch.linalg.vector_norm(x, ord=3, dim=-1), lambda g: {"x": rn(g, R, D)},
               "vector_norm ord 3", bwd=["x"])
CASES += _case("vector_norm_half", lambda x: torch.linalg.vector_norm(x, ord=0.5, dim=-1), lambda g: {"x": rn(g, R, D)},
               "vector_norm ord 0.5", bwd=["x"])
CASES += _case("std_mean_c0", lambda x: torch.std_mean(x, dim=-1, correction=0)[0], lambda g: {"x": rn(g, R, D) + 3},
               "std_mean correction 0", bwd=["x"])
CASES += _case("cdist_p1", lambda a, b: torch.cdist(a, b, p=1.0), lambda g: {"a": rn(g, 32, 16), "b": rn(g, 40, 16)},
               "cdist p 1", bwd=["a", "b"])


def _scatter_make(g):
    return {"src": rn(g, R, D), "base": rn(g, 16, D), "idx": torch.randint(0, 16, (R, 1), generator=g).expand(R, D).contiguous()}


for _red in ("mean", "amax", "sum"):
    for _inc in (True, False):
        CASES += _case(f"scatter_reduce_{_red}_{'self' if _inc else 'noself'}",
                       (lambda red, inc: lambda src, base, idx: base.scatter_reduce(0, idx, src, red, include_self=inc))(_red, _inc),
                       _scatter_make, f"scatter_reduce {_red} include_self={_inc}", bwd=None)

# ---- batch_norm training: output and running statistics --------------------------------------------------------


class BatchNormTrain(Case):
    implementation = "torch.compile(F.batch_norm, training=True) -> Inductor Triton kernels"
    spec_bound = "float64 eager evaluation, declared bound 2^-40 * max|f|"

    def __init__(self, name, shape, momentum):
        self.name, self.shape, self.momentum = name, shape, momentum
        self.specification = (f"F.batch_norm training, momentum {momentum}: output and in-place running_mean / "
                              "running_var (unbiased variance)")

    def setup(self):
        def fn(x, rm, rv, w, b):
            return F.batch_norm(x, rm, rv, w, b, training=True, momentum=self.momentum, eps=1e-5)
        self._fn = fn
        self._compiled = torch.compile(fn, dynamic=False)
        self.launch(self.inputs(10_000))

    def inputs(self, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        c = self.shape[1]
        return {"x": (rn(g, *self.shape) * 2 + 1).cuda(), "rm": rn(g, c).cuda(), "rv": (torch.rand(c, generator=g) + 0.5).cuda(),
                "w": (1 + 0.1 * rn(g, c)).cuda(), "b": (0.1 * rn(g, c)).cuda()}

    def launch(self, inp):
        a = {k: v.clone() for k, v in inp.items()}
        out = self._compiled(**a)
        return {"out": out, "running_mean": a["rm"], "running_var": a["rv"]}

    def spec(self, inp):
        a = {k: v.double().clone() for k, v in inp.items()}
        out = self._fn(**a)
        return {"out": f64_point_spec(out.cpu().numpy()), "running_mean": f64_point_spec(a["rm"].cpu().numpy()),
                "running_var": f64_point_spec(a["rv"].cpu().numpy())}


CASES += [BatchNormTrain("ind_batch_norm_train_2d", (16, 32, 7, 9), 0.1),
          BatchNormTrain("ind_batch_norm_train_1d", (8, 32, 5), 0.3)]

# ---- compiled optimizer steps ----------------------------------------------------------------------------------


class OptimStep(Case):
    implementation = "torch.compile(optimizer.step) -> Inductor Triton kernels (one step from a populated state)"
    spec_bound = "float64 eager optimizer (foreach=False) from the same state, declared bound 2^-40 * max|f|"

    def __init__(self, name, cls, kw, warm=3):
        self.name, self.cls, self.kw, self.warm = name, cls, kw, warm
        self.specification = f"torch.optim.{cls} {kw}: parameter and state after one step, eager float64"

    def _opt(self, p, dtype_kw=None):
        return getattr(torch.optim, self.cls)([p], **self.kw, **(dtype_kw or {}))

    def setup(self):
        self._p = torch.nn.Parameter(torch.zeros(48, 64, device="cuda"))
        self._o = self._opt(self._p, {"foreach": False})
        self._p.grad = torch.ones_like(self._p)
        self._o.step()  # create the state
        self._step = torch.compile(self._o.step)
        self.launch(self.inputs(10_000))
        torch.cuda.synchronize()

    def inputs(self, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        p0 = rn(g, 48, 64).cuda()
        grads = [rn(g, 48, 64).cuda() * (0.5 + i) for i in range(self.warm + 1)]
        # populate a state with eager fp32 steps (the state values are then inputs of both sides)
        p = torch.nn.Parameter(p0.clone())
        o = self._opt(p, {"foreach": False})
        for gr in grads[:-1]:
            p.grad = gr.clone()
            o.step()
        st = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in o.state[p].items()}
        return {"p": p.detach().clone(), "state": st, "g": grads[-1]}

    def _load(self, opt, p, inp, dtype):
        with torch.no_grad():
            p.copy_(inp["p"].to(dtype))
        p.grad = inp["g"].to(dtype).clone()
        st = opt.state[p]
        for k, v in inp["state"].items():
            if torch.is_tensor(v) and k in st and torch.is_tensor(st[k]):
                st[k].copy_(v.to(st[k].dtype))
            else:
                st[k] = v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() and k != "step" else v

    def launch(self, inp):
        self._load(self._o, self._p, inp, torch.float32)
        self._step()
        out = {"param": self._p.data}
        for k, v in self._o.state[self._p].items():
            if torch.is_tensor(v) and v.is_floating_point() and v.dim() > 0 and k != "step":
                out[k] = v
        return out

    def spec(self, inp):
        p = torch.nn.Parameter(inp["p"].double().clone())
        o = self._opt(p, {"foreach": False})
        p.grad = inp["g"].double().clone()
        st = o.state[p]
        for k, v in inp["state"].items():
            st[k] = v.double().clone() if torch.is_tensor(v) and v.is_floating_point() and k != "step" else (
                v.clone() if torch.is_tensor(v) else v)
        o.step()
        res = {"param": f64_point_spec(p.detach().cpu().numpy())}
        for k, v in o.state[p].items():
            if torch.is_tensor(v) and v.is_floating_point() and v.dim() > 0 and k != "step":
                res[k] = f64_point_spec(v.cpu().numpy())
        return res


CASES += [
    OptimStep("opt_adamw", "AdamW", dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)),
    OptimStep("opt_adam_amsgrad", "Adam", dict(lr=1e-2, amsgrad=True, weight_decay=0.05)),
    OptimStep("opt_adam_maximize", "Adam", dict(lr=1e-2, maximize=True)),
    OptimStep("opt_nadam", "NAdam", dict(lr=1e-2, weight_decay=0.05, momentum_decay=4e-3)),
    OptimStep("opt_nadam_decoupled", "NAdam", dict(lr=1e-2, weight_decay=0.05, decoupled_weight_decay=True)),
    OptimStep("opt_radam", "RAdam", dict(lr=1e-2, weight_decay=0.05), warm=6),
    OptimStep("opt_radam_early", "RAdam", dict(lr=1e-2), warm=1),
    OptimStep("opt_adamax", "Adamax", dict(lr=1e-2, weight_decay=0.05)),
    OptimStep("opt_adagrad", "Adagrad", dict(lr=1e-2, lr_decay=1e-2, weight_decay=0.05, initial_accumulator_value=0.1)),
    OptimStep("opt_rmsprop_centered", "RMSprop", dict(lr=1e-2, alpha=0.9, momentum=0.9, centered=True, weight_decay=0.05)),
    OptimStep("opt_sgd_nesterov", "SGD", dict(lr=1e-2, momentum=0.9, dampening=0.0, nesterov=True, weight_decay=0.05)),
    OptimStep("opt_sgd_dampening", "SGD", dict(lr=1e-2, momentum=0.9, dampening=0.3, weight_decay=0.05)),
    OptimStep("opt_adadelta", "Adadelta", dict(lr=1.0, rho=0.9, weight_decay=0.05)),
    OptimStep("opt_asgd", "ASGD", dict(lr=1e-2, t0=2.0, weight_decay=0.05), warm=4),
]
