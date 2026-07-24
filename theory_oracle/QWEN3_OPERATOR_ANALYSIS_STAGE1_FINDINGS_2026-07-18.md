# Qwen3 operator analysis stage-1 findings

## What is established

At one frozen Qwen3-0.6B GRPO step-29 matched state, two endpoint-preserving
operator interventions are valid:

| Invocation | Endpoint preservation | Repair | Injection | Supported conclusion |
|---|---:|---:|---:|---|
| final `lm_head` Linear | exact | 0 | 0 | no implementation-specific effect at this state |
| final `model.norm` Qwen3RMSNorm | exact | 0 | 0 | no implementation-specific effect at this state |

The original selected-token log-probability discrepancy has L2
`0.09163622558116913`. It is already present before final RMSNorm.

These null contrasts concern eager-versus-compiled implementation choice at the
target invocation. They do not test how the operator's mathematical map
propagates an upstream discrepancy.

## What failed

Splitting the model around decoder layer 27 did not reproduce the original
compiled endpoint. The resulting nonzero layer repair/injection contrasts are
invalid for attribution. This shows that the statistical Oracle can measure an
effect correctly while the causal treatment is still ill-defined.

## Revised operator-analysis rule

An operator result needs two independent ledgers:

1. **Oracle ledger**: the B/H/N/U or selected-state impact contrast;
2. **treatment-integrity ledger**: whether the intervention preserved the
   original graph/kernel/layout context apart from the declared target.

If treatment integrity fails, the result is not operator causality. At most it is
an intervention-dependent attribution for a newly partitioned realization.

For fused operators, the appropriate controlled design is:

- `C0`: original whole-compiled candidate;
- `Cb`: candidate with an opaque boundary but candidate-equivalent target
  semantics;
- `Cr`: the same boundary and topology as `Cb`, with only the target semantics
  repaired.

`Cr-Cb` estimates a barrier-conditioned target intervention. `Cb-C0` separately
measures the boundary/fusion disturbance. Only if `Cb=C0` can the target contrast
be transported directly to the original candidate.

## Original-candidate inventory result

The unchanged whole-compiled realization was subsequently inventoried with exact
graph and scorer gates. Its dynamic graph contains a cross-layer fused kernel
family invoked 27 times. That family combines residual adds with the next layer's
RMSNorm `pow/mean/rsqrt/mul/cast` chain. This explains why a decoder-layer split
changes the candidate realization and confirms that constituent operators are
not independently materialized treatments in the original candidate.

## Next causal experiment

The next experiment should target the repeated cross-layer fusion with three
compiled arms: original candidate `C0`, matched barrier control `Cb`, and repaired
barrier arm `Cr`. It must report `Cb-C0` separately from `Cr-Cb`. Reduction or
RMSNorm is only a candidate treatment; the inventory does not establish it as a
root cause.
