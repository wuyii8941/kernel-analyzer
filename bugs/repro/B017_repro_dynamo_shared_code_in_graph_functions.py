"""B017: torch.compile reuses a graph across distinct in-graph torch functions that share one code object.

Dynamo classifies torch.nn.functional.max_pool*/adaptive_max_pool*/fractional_max_pool* (and torch.unique,
unique_consecutive, lu) as in-graph functions (TorchInGraphFunctionVariable). When such a function arrives through an
argument, a closure cell, a default or a module attribute, create_with_source installs CLOSURE_MATCH, which for a
Python function only checks __code__. All of these functions are made by torch._jit_internal.boolean_dispatch and share
the code of its inner `fn`, so the guard accepts any of them and the cached graph of the first one is reused.

    python bugs/repro/B017_repro_dynamo_shared_code_in_graph_functions.py            # torch as installed
    python ... --pr197845   # also guard method/wrapper descriptors (the fix proposed for pytorch#197811)
    python ... --fixed      # in-graph Python functions guarded by identity instead of __code__
"""
import os
import sys

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

if "--fixed" in sys.argv or "--pr197845" in sys.argv:
    import inspect

    from torch._dynamo.guards import GuardBuilder, install_guard
    from torch._dynamo.variables.torch import BaseTorchVariable
    from torch._dynamo.utils import is_wrapper_or_member_descriptor
    import types

    orig = BaseTorchVariable.create_with_source.__func__

    def create_with_source(cls, value, source):
        if "--fixed" in sys.argv and inspect.isfunction(value):
            install_guard(source.make_guard(GuardBuilder.ID_MATCH))  # identity: each in-graph function is one object
            return cls(value, source=source)
        if isinstance(value, (types.MethodDescriptorType, types.WrapperDescriptorType)):
            install_guard(source.make_guard(GuardBuilder.ID_MATCH))  # what PR #197845 adds
            return cls(value, source=source)
        return orig(cls, value, source)

    BaseTorchVariable.create_with_source = classmethod(create_with_source)

dev = "cuda" if torch.cuda.is_available() else "cpu"
# B017_BACKEND=eager: the bug is in Dynamo's guards, so it shows without any code generation
BACKEND = os.environ.get("B017_BACKEND", "inductor")
print(torch.__version__, dev, BACKEND, [a for a in sys.argv[1:]])


def show(name, eager, compiled):
    same = eager.shape == compiled.shape and torch.allclose(eager, compiled)
    print(f"  {name:34s} eager {tuple(eager.shape)!s:14s} compiled {tuple(compiled.shape)!s:14s} "
          f"{'ok' if same else 'WRONG'}" + ("" if same or eager.shape != compiled.shape else
                                             f" (max diff {(eager - compiled).abs().max().item():.3g})"))


print("1. argument of a compiled function (same output shape, different values)")
torch._dynamo.reset()


@torch.compile(backend=BACKEND)
def apply(pool, x, k):
    return pool(x, k)


x = torch.randn(2, 3, 10, device=dev)
for pool in (F.max_pool1d, F.adaptive_max_pool1d):
    show(pool.__name__ + "(x, 3)", pool(x, 3), apply(pool, x, 3))

print("2. module attribute: the pooling function is a constructor argument")
torch._dynamo.reset()


class Head(nn.Module):
    def __init__(self, pool):
        super().__init__()
        self.conv = nn.Conv1d(3, 8, 3, padding=1)
        self.pool = pool

    def forward(self, x):
        return self.pool(self.conv(x), 3).flatten(1)


for pool in (F.adaptive_max_pool1d, F.max_pool1d):
    torch.manual_seed(0)
    m = Head(pool).to(dev)
    show(f"Head({pool.__name__})", m(x), torch.compile(m, backend=BACKEND)(x))

print("3. closure (factory of per-config functions)")
torch._dynamo.reset()


def make(pool):
    def f(t):
        return pool(t, 3)
    return torch.compile(f, backend=BACKEND)


for pool in (F.max_pool1d, F.adaptive_max_pool1d):
    show(pool.__name__, pool(x, 3), make(pool)(x))

print("4. control: method descriptors (pytorch#197811, PR #197845)")
torch._dynamo.reset()
a, b = torch.rand(5, device=dev) + 1, torch.rand(5, device=dev) + 1
for op in (torch.Tensor.add, torch.Tensor.mul):
    show("Tensor." + op.__name__, op(a, b), torch.compile(lambda o, u, v: o(u, v), backend=BACKEND)(op, a, b))

print("5. regional compilation (one compile per submodule) of a two-branch model")


class Branch(nn.Module):
    def __init__(self, c, pool):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 3, padding=1)
        self.pool = pool

    def forward(self, t):
        return self.pool(torch.relu(self.conv(t)), 3).flatten(1)


class TwoBranch(nn.Module):  # max- and adaptive-max-pooled features of the same input
    def __init__(self, c):
        super().__init__()
        self.a, self.b = Branch(c, F.max_pool1d), Branch(c, F.adaptive_max_pool1d)

    def forward(self, t):
        return torch.cat([self.a(t), self.b(t)], 1)


torch.manual_seed(0)
m = TwoBranch(3).to(dev)
eager = m(x)
torch._dynamo.reset()
show("torch.compile(model)", eager, torch.compile(m, backend=BACKEND)(x))
torch._dynamo.reset()
m.a.compile(backend=BACKEND)
m.b.compile(backend=BACKEND)
show("m.a.compile(); m.b.compile()", eager, m(x))
