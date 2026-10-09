# [dynamo] Compiled code silently reuses the graph of F.max_pool1d for F.adaptive_max_pool1d (in-graph torch functions guarded only by `__code__`)

### 🐛 Describe the bug

`F.max_pool{1,2,3}d`, `F.adaptive_max_pool{1,2,3}d`, `F.fractional_max_pool{2,3}d`, `torch.unique`,
`torch.unique_consecutive` and `torch.lu` are all built by `torch._jit_internal.boolean_dispatch`, so they are distinct
closures of one inner function `fn` and share its `__code__`. When one of them reaches a compiled region as a value
(argument, closure cell, default, module attribute), the next one is accepted by the guards and the first one's graph
runs:

```python
import torch
import torch.nn.functional as F

@torch.compile          # same with backend="eager"
def apply(pool, x, k):
    return pool(x, k)

x = torch.randn(2, 3, 10)
print(torch.allclose(apply(F.max_pool1d, x, 3), F.max_pool1d(x, 3)))                    # True
print(torch.allclose(apply(F.adaptive_max_pool1d, x, 3), F.adaptive_max_pool1d(x, 3)))  # False: same shape, max_pool1d values
```

A more realistic form is regional compilation of a model whose branches are configured with different pooling
functions:

```python
import torch.nn as nn

class Branch(nn.Module):
    def __init__(self, c, pool):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 3, padding=1)
        self.pool = pool
    def forward(self, t):
        return self.pool(torch.relu(self.conv(t)), 3).flatten(1)

class TwoBranch(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a, self.b = Branch(c, F.max_pool1d), Branch(c, F.adaptive_max_pool1d)
    def forward(self, t):
        return torch.cat([self.a(t), self.b(t)], 1)

m = TwoBranch(3)
ref = m(x)
m.a.compile(); m.b.compile()
print(torch.allclose(m(x), ref))   # False (torch.compile(m) as a whole is correct)
```

**Cause.** These functions are in `torch_non_c_binding_in_graph_functions` (`TorchInGraphFunctionVariable`). For a
Python function, `BaseTorchVariable.create_with_source` installs `CLOSURE_MATCH`, which for `types.FunctionType` checks
only `__code__`. That is right for user functions, which are inlined and whose closure cells get their own guards, but
an in-graph function is not inlined: the code guard is the only guard, and it cannot tell the boolean_dispatch
functions apart. `TORCH_LOGS=guards` shows just `___check_obj_id(L['pool'].__code__, ...)` pointing at
`_jit_internal.py:609`.

This is related to #197811 (method descriptors get no guard at all), but #197845 does not fix it: with that change
applied, the `Tensor.add`/`Tensor.mul` case is fixed and the pooling cases above still return wrong values.

### Suggested fix

In `BaseTorchVariable.create_with_source`, guard in-graph Python functions by identity (`ID_MATCH` /
`FUNCTION_MATCH`) instead of `CLOSURE_MATCH`. In-graph torch functions are module-level objects with stable identity.
Patching this at runtime makes all the cases above (argument, closure, default argument, module attribute, regional
compilation) correct.

### Versions

torch 2.10.0+cu128 (CUDA and CPU) and nightly 2.15.0.dev20261005+cpu (all five forms above still wrong with
`backend="eager"`). `create_with_source`, `CLOSURE_MATCH` and the trace_rules entries are unchanged on main (2026-10-06).
