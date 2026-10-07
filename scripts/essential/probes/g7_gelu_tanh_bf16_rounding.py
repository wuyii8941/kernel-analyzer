import torch, math, torch.nn.functional as F
torch.manual_seed(0)
x = torch.randn(8192, 257, dtype=torch.float64).float().to(torch.bfloat16).cuda()
xd = x.double()
exact = 0.5 * xd * (1 + torch.tanh(math.sqrt(2 / math.pi) * (xd + 0.044715 * xd ** 3)))
rn = exact.to(torch.bfloat16).double()
f = torch.compile(lambda t: F.gelu(t, approximate="tanh"))
for name, k in (("inductor", f(x)), ("eager", F.gelu(x, approximate="tanh"))):
    k = k.double()
    m = exact.abs() > 1e-6
    rel = ((k - exact) * exact.sign() / exact.abs())[m]
    print(f"{name:9s} K == RN(exact): {(k == rn).float().mean().item():.4f} | mean signed rel {rel.mean().item():.3e} +- {(rel.std()/rel.numel()**0.5).item():.1e}"
          f" | RN(exact) mean signed rel {(((rn - exact) * exact.sign() / exact.abs())[m]).mean().item():.3e}")
    d = (k != rn)
    if d.any():
        idx = d.nonzero()[:3]
        for i in idx: print("    x", x[tuple(i)].item(), "K", k[tuple(i)].item(), "RN(exact)", rn[tuple(i)].item(), "exact", exact[tuple(i)].item())
