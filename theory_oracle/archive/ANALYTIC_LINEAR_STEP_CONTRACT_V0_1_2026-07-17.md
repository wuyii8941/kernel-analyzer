# Analytic Linear Training-Step Contract v0.1 — 2026-07-17

## Status and role

Frozen before execution. This subject exists to instantiate the numerical
correctness ledger of the Training-Step Oracle with an independent mathematical
reference. It is not offered as a representative modern network workload.

Qwen/BERT supply architecture and impact realism; this subject supplies a complete,
auditable floating transition relation.

## State and transition

For dimension `n=257` and state index `s`, all weights, features, target and learning
rate are binary rational values exactly representable in float32. They are generated
by the integer formulas embedded in the executor; no RNG library defines the bank.

The mathematical transition is

```text
p       = sum_j w_j x_j
r       = p - y
loss    = 0.5 r^2
g_j     = r x_j
w'_j    = w_j - lr g_j
```

The executed program uses a float32 CUDA parameter, float32 tensors, autograd and
one `torch.optim.SGD` step with:

```text
lr = 2^-5
momentum = 0
dampening = 0
weight_decay = 0
nesterov = false
maximize = false
foreach = false
fused = false
differentiable = false
```

The candidate compiles the forward/loss core with Inductor, `fullgraph=True` and
`dynamic=False`. Backward and the common optimizer are included in the observed
transition even though the optimizer implementation is not the treatment.

## Independent truth

Each float32 input is converted through its exact integer ratio. Python `Fraction`
arithmetic computes exact real `p`, `r`, `loss`, `g` and `w'`. Eager is not used as
truth.

## Numerical envelope

Let `u=2^-24` and `gamma(k)=ku/(1-ku)`. Preconditions are finite normal-range
intermediates, round-to-nearest IEEE-style float32 primitive arithmetic, no TF32
matmul (the program uses elementwise multiplication plus reduction), and no
fast-math approximation outside ordinary rounding.

For any reduction association, the dot-product bound uses

```text
e_p = gamma(2n) * sum_j |w_j x_j|
```

which conservatively charges one rounding for each product and addition. The
residual, loss, gradient and update bounds propagate `e_p` and add conservative
`gamma` terms for their primitive operation counts. The exact formulas are fixed in
the executor before any candidate result is read.

This envelope is deliberately conservative. It is nevertheless independent of the
observed eager/compiled discrepancy. A value outside the envelope is a numerical
contract `REJECT`; a value inside is `ACCEPT` for this subject and these
preconditions.

If a precondition fails, the numerical verdict is `INAPPLICABLE` rather than
silently widening the bound.

## Modes and controls

- `correct`: compiled program evaluates the declared reduction;
- `reverse`: compiled program reverses term order before the reduction; this is a
  permitted reassociation and must remain within the same envelope;
- `drop_last`: negative control omits the final term only in the candidate. It must
  be rejected by the numerical transition ledger while candidate execution identity
  remains valid.

The negative control is an injected wrong program, not evidence of a natural
Inductor bug.

## Repeats and bank

- 32 deterministic states;
- at least two exact-state repeats per state;
- CUDA deterministic-algorithm mode;
- same graph identity and one compiled invocation per scored candidate arm.

Any repeated signature change is reported as runtime variability and invalidates a
claim of deterministic repeatability. The finite bank is the target of the first
claim; no broader state-distribution rate is inferred.

## Required result

For each arm and state report:

- candidate identity and exact structure/control verdict;
- absolute error, analytic bound and error/bound ratio for prediction, loss,
  complete gradient and complete next parameter vector;
- numerical `ACCEPT/REJECT/INAPPLICABLE`;
- repeated-signature stability.

The positive modes pass only if every covered field in every state lies within its
predeclared bound. The negative control passes validation only if it produces at
least one candidate numerical `REJECT` without invalidating candidate identity.

