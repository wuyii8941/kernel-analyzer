import torch
torch.manual_seed(0)
x = torch.randn(4096, 4099, dtype=torch.float64).float().to(torch.bfloat16).cuda()
exact = x.double().sum(-1)
f = torch.compile(lambda t: t.sum(-1))
for name, k in (("inductor", f(x)), ("eager", x.sum(-1))):
    k = k.double()
    rel = (k - exact) * exact.sign() / exact.abs().clamp_min(1e-30)
    # bf16 rounding of the exact sum (round to nearest even) for comparison
    rn = exact.to(torch.bfloat16).double()
    relrn = (rn - exact) * exact.sign() / exact.abs().clamp_min(1e-30)
    print(name, "mean signed rel (K-exact)*sign:", f"{rel.mean().item():.3e}", "+-", f"{(rel.std()/rel.numel()**0.5).item():.1e}",
          "| same for RN(exact):", f"{relrn.mean().item():.3e}", "| K == RN(exact):", f"{(k == rn).float().mean().item():.3f}")
