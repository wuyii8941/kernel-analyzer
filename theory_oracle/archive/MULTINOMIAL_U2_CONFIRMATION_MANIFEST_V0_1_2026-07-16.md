# Multinomial U2 Confirmation Manifest v0.1 — 2026-07-16

> Frozen before executing the candidate draws. This replaces the uninstantiated U2 row of the earlier validation schema for a new, separately versioned mechanics confirmation; it is not retroactively counted as original v0.1 preregistration.

## Subject and target law

```text
framework: PyTorch 2.13.0.dev20260609+cu126
hardware: Tesla T4 CUDA
subject: torch.multinomial([1.0, 1.0], num_samples=100, replacement=True)
candidate: torch.compile(..., backend="inductor", fullgraph=True)
generator protocol: torch.cuda.manual_seed_all(271828) immediately before candidate call
target law: categorical p* = (0.5, 0.5)
```

The target follows directly from normalized nonnegative weights. Same-seed token identity is not part of the contract.

## Estimand and acceptable set

For binary output, let `p_C(0)` be the candidate probability of token 0. The categorical total-variation distance from the target equals `abs(p_C(0)-0.5)`.

```text
acceptable law set: abs(p_C(0) - 0.5) <= 0.01
sample estimate: p_hat = count(token 0) / 100
```

The `0.01` boundary is frozen before draws and is a mechanics-control margin, not a claim about an application requirement.

## Uncertainty rule

Use a two-sided 95% Hoeffding confidence interval:

```text
radius h = sqrt(log(2/0.05) / (2*100))
C_hat = [max(0,p_hat-h), min(1,p_hat+h)]
A = [0.49, 0.51]
```

Verdict:

```text
INVALID         if candidate graph identity, weights, draw count or support checks fail
ACCEPT          if C_hat is wholly inside A
REJECT          if C_hat and A are disjoint
INDETERMINATE   otherwise
```

Because the sample budget is deliberately too small for a plus/minus `0.01` equivalence claim, a central observed count should be indeterminate. An extreme draw may legitimately reject; outputs will not be replaced or the margin retuned.

## Reported descriptive fields

Report counts, support validity, `p_hat`, Hoeffding radius/interval, verdict and graph-capture counters. Do not compare a single sampled token to eager.
