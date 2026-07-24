# Sum Legal-Reassociation Confirmation Manifest v0.1 — 2026-07-16

> Frozen before executing the confirmation input below. The discovery inputs used `2^24` with unit residuals; those observations are excluded from this confirmation row.

## Case N4C1

```text
framework: PyTorch 2.13.0.dev20260609+cu126
hardware: Tesla T4 CUDA
subject: torch.sum, float32, contiguous length-3 tensor, default dtype
candidate: torch.compile(..., backend="inductor", fullgraph=True)
input: [33554432.0, 2.0, -33554432.0] = [2^25, 2, -2^25]
mathematical truth: 2.0, evaluated exactly in float64/integer arithmetic
```

## Contract

PyTorch does not require bitwise equality between mathematically equivalent floating computations and permits implementation-dependent evaluation order. For a finite length-3 float32 sum under the standard round-to-nearest model and no overflow/underflow, use the independently specified enclosure

```text
abs(z - sum_R(x)) <= gamma_2 * sum(abs(x))
u = 2^-24
gamma_2 = 2u / (1 - 2u)
```

For the frozen input, the bound is approximately `8.0000007`. This is computed from dtype and operands, not from candidate output.

## Verdict rule

```text
INVALID  if compiled-path identity or matched operands fail
REJECT   for either output outside its analytical enclosure
ACCEPT   for each covered output inside the enclosure
```

Eager/compiled equality is not required. Raw max absolute delta and default `torch.allclose` are descriptive baselines only.

## Expected control role

The semantic label is conforming for any valid floating result inside the frozen enclosure. The confirmation is useful as a real-candidate large-delta control only if eager and compiled differ enough to fail default allclose; otherwise it remains a small-delta conforming control and is reported without replacement or retuning.

This one transformed point confirms mechanics within the discovered reduction family; it does not establish prevalence or cross-family generality.
