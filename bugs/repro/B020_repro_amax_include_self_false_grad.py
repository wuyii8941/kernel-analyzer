"""B020: scatter_reduce / index_reduce with reduce='amax'/'amin' and include_self=False - the backward counts the
EXCLUDED self value in the tie split whenever it happens to equal the result, so the source gradients are too small.

Independent check (no tool code): the forward is correct, and out[0] = max over the contributions only; its derivative
w.r.t. the unique maximiser is 1.  The central difference below is exact for this piecewise-linear function (h far
below the gap 3 - 1 between the largest and second-largest contribution), so it needs no tolerance argument.

    python bugs/repro/B020_repro_amax_include_self_false_grad.py
"""
import torch

print("torch", torch.__version__)
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def case(dev, op, red):
    big, small = (3.0, 1.0) if red == "amax" else (1.0, 3.0)
    self_ = torch.tensor([big, 5.0], device=dev, dtype=torch.float64)      # self[0] is EXCLUDED and equals the result
    src = torch.tensor([big, small], device=dev, dtype=torch.float64)      # position 0 receives big (unique) and small
    idx = torch.tensor([0, 0], device=dev)

    def f(s, x):
        if op == "scatter_reduce":
            return s.scatter_reduce(0, idx, x, red, include_self=False)
        return s.index_reduce(0, idx, x, red, include_self=False)

    s, x = self_.clone().requires_grad_(True), src.clone().requires_grad_(True)
    f(s, x).backward(torch.ones(2, device=dev, dtype=torch.float64))
    h = 2.0 ** -20                                                          # exact central difference (piecewise linear)
    e = torch.zeros(2, device=dev, dtype=torch.float64)
    e[0] = h
    fd = (f(self_, src + e)[0] - f(self_, src - e)[0]) / (2 * h)
    e_self = torch.zeros(2, device=dev, dtype=torch.float64)
    e_self[0] = h
    fd_self = (f(self_ + e_self, src)[0] - f(self_ - e_self, src)[0]) / (2 * h)
    print(f"{dev:4s} {op:14s} {red}: autograd d out[0]/d src[0] = {x.grad[0].item():.4f}, "
          f"exact difference = {fd.item():.4f};  d out[0]/d self[0]: autograd {s.grad[0].item():.4f}, "
          f"difference {fd_self.item():.4f}")
    return x.grad[0].item(), fd.item()


bad = 0
for dev in devices:
    for op in ("scatter_reduce", "index_reduce"):
        for red in ("amax", "amin"):
            g, fd = case(dev, op, red)
            bad += g != fd
print("MISMATCHES:", bad)
