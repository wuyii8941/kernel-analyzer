#!/usr/bin/env python3
"""Probe: does torch.compile reuse a cached graph for a different callable / configuration value?  For pairs (a, b)
that must give different results, compile once, call with a, then with b (argument and closure forms), and compare
with eager.  B017 / pytorch#197811 are the known instances; this looks for others.

    python scripts/probes/probe_dynamo_guard_pairs.py
"""
import enum
import functools
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple

x = torch.rand(4, 6) + 0.5


class Mode(enum.Enum):
    ADD = 1
    MUL = 2


def by_mode(t, m):
    return t + 1 if m is Mode.ADD else t * 3


class Holder:
    def __init__(self, k):
        self.k = k

    def apply(self, t):
        return t * self.k


lin1, lin2 = nn.Linear(6, 6), nn.Linear(6, 6)
PAIRS = {
    "functools.partial(torch.add, alpha=2 / 3)": (functools.partial(torch.add, alpha=2), functools.partial(torch.add, alpha=3), lambda f, t: f(t, t)),
    "functools.partial(F.gelu, approximate=none / tanh)": (functools.partial(F.gelu, approximate="none"), functools.partial(F.gelu, approximate="tanh"), lambda f, t: f(t)),
    "operator.methodcaller('add', 1) / ('mul', 2)": (operator.methodcaller("add", 1), operator.methodcaller("mul", 2), lambda f, t: f(t)),
    "operator.attrgetter('mT') / ('T')": (operator.attrgetter("mT"), operator.attrgetter("H"), lambda f, t: f(t.reshape(2, 2, 6)).sum(-1)),
    "operator.itemgetter(0) / (1)": (operator.itemgetter(0), operator.itemgetter(1), lambda f, t: f(t)),
    "Enum member Mode.ADD / Mode.MUL": (Mode.ADD, Mode.MUL, lambda m, t: by_mode(t, m)),
    "class nn.ReLU / nn.Tanh": (nn.ReLU, nn.Tanh, lambda c, t: c()(t)),
    "bound method of two objects (k=2 / k=5)": (Holder(2).apply, Holder(5).apply, lambda f, t: f(t)),
    "bound forward of two nn.Linear": (lin1.forward, lin2.forward, lambda f, t: f(t)),
    "torch.nn.modules.utils._pair / _triple": (_pair, _triple, lambda f, t: t * len(f(3))),
    "numpy ufunc np.add / np.multiply (on tensors)": (np.add, np.multiply, lambda f, t: torch.as_tensor(f(t.numpy(), t.numpy()))),
    "torch.dtype float32 / float64 (to)": (torch.float32, torch.float64, lambda d, t: t.to(d) * 1.1),
    "staticmethod of two classes": (type("A", (), {"f": staticmethod(lambda t: t + 1)}).f, type("B", (), {"f": staticmethod(lambda t: t - 1)}).f, lambda f, t: f(t)),
    "torch.Tensor.add_ / mul_ (known #197811)": (torch.Tensor.add_, torch.Tensor.mul_, lambda f, t: f(t.clone(), 2.0)),
}


def check(label, a, b, use):
    rows = []
    for form in ("argument", "closure"):
        torch._dynamo.reset()
        if form == "argument":
            cf = torch.compile(lambda v, t: use(v, t), backend="eager")
            call = lambda v: cf(v, x)  # noqa: E731
        else:
            def make(v):
                return torch.compile(lambda t: use(v, t), backend="eager")
            fa, fb = make(a), make(b)
            # both closures come from the same code object, as factories in user code do
            call = lambda v: (fa if v is a else fb)(x)  # noqa: E731
        try:
            ya, yb = call(a), call(b)
            ea, eb = use(a, x), use(b, x)
            ok = torch.allclose(ya.double(), ea.double()) and yb.shape == eb.shape and torch.allclose(yb.double(), eb.double())
            rows.append(f"{form}: {'ok' if ok else 'WRONG'}")
        except Exception as e:  # noqa: BLE001
            rows.append(f"{form}: {type(e).__name__}: {str(e).splitlines()[0][:50]}")
    print(f"{label:52s} {rows}", flush=True)


for label, (a, b, use) in PAIRS.items():
    check(label, a, b, use)
