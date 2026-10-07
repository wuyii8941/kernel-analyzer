"""2b G8 probe: Inductor CPU (C++ backend) fails to compile index_add / index_reduce / scatter into a single-element target
(2.10: AssertionError in the C++ code generator; nightly: CppCompileError).  Eager and the CUDA (Triton) backend compile."""
import torch

print(torch.__version__)
s = torch.tensor([2.0])
idx = torch.zeros(3, dtype=torch.long)
src = torch.tensor([3.0, -1.5, 0.5])
ops = {
    "index_add": lambda a, i, b: a.index_add(0, i, b),
    "index_reduce_prod": lambda a, i, b: a.index_reduce(0, i, b, "prod"),
    "index_reduce_mean": lambda a, i, b: a.index_reduce(0, i, b, "mean"),
    "index_reduce_amax": lambda a, i, b: a.index_reduce(0, i, b, "amax"),
    "scatter_add": lambda a, i, b: a.scatter_add(0, i, b),
    "scatter_reduce_sum": lambda a, i, b: a.scatter_reduce(0, i, b, "sum"),
    "scatter_reduce_mean": lambda a, i, b: a.scatter_reduce(0, i, b, "mean"),
    "scatter_reduce_amax": lambda a, i, b: a.scatter_reduce(0, i, b, "amax"),
}
for name, fn in ops.items():
    row = []
    for grad in (False, True):
        torch._dynamo.reset()
        a, b = s.clone().requires_grad_(grad), src.clone().requires_grad_(grad)
        try:
            y = torch.compile(fn)(a, idx, b)
            if grad:
                y.sum().backward()
            ref = fn(s.clone(), idx, src.clone())
            row.append(f"{'fwd+bwd' if grad else 'fwd'}: ok ({'equal' if torch.allclose(y.detach(), ref) else 'DIFFERENT'})")
        except Exception as e:  # noqa: BLE001
            row.append(f"{'fwd+bwd' if grad else 'fwd'}: {type(e).__name__}: {str(e).splitlines()[0][:60] if str(e) else ''}")
    print(f"{name:22s}", " | ".join(row))
