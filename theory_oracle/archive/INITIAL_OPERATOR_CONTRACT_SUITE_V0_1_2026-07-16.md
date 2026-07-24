# Initial Operator Contract Suite v0.1 — 2026-07-16

> Purpose: show that Operator Oracle v0.1 can produce genuine contracts across semantic classes. These are archetypal contracts, not a claim that a small list covers all DL operators.

## 1. Contract C1 — specified metadata and discrete behavior

### Domain

Any operator field for which the applicable semantics specify output structure, shape, dtype, integer/boolean/index result, mutation, aliasing, exception or RNG-state behavior.

### Semantic envelope

`S_C1(x)` is the finite relation given by the governing specification. Unspecified behavior is removed from the claim rather than copied from eager.

### Verdict

```text
candidate field not in S_C1(x) -> REJECT correctness for this witness
all fields in S_C1(x)          -> ACCEPT for this covered input
missing specification          -> UNINSTANTIATED for that field
failed candidate identity      -> INVALID
```

### Why this is a real Oracle

There is no learned threshold and no statistical averaging. It directly checks observable program semantics. A universal family acceptance still requires exhaustive/formal coverage; sampled cases remain scoped evidence.

## 2. Contract C2 — exact ranking/selection on a matched operand

### Domain

Argmax, top-k, threshold or routing selection when both implementations receive the identical input tensor bits and the tie/NaN/order policy is specified. Cases outside the specified policy are excluded or handled by a separate relation.

### Semantic envelope

Compute the specified index/set/assignment relation directly from the matched operand:

```text
S_C2(x) = specified valid index/set/assignment results
```

It may contain multiple outputs if ties are explicitly allowed to resolve nondeterministically.

### Interpretation

- candidate output outside `S_C2(x)` is a selection-operator correctness violation;
- eager and compiled selecting differently because they received different upstream logits is not a C2 operator violation—it is downstream semantic impact of upstream discrepancy;
- this separation prevents assigning an upstream numerical error to argmax/top-k itself.

## 3. Contract C3 — floating reduction under a declared arithmetic model

### Restricted domain

Inputs are finite floating values; accumulation precision and rounding mode are declared; overflow/underflow and special-value cases are either excluded with explicit guards or covered by separate rules. Candidate execution must use an arithmetic strategy admitted by the contract.

### Independent reference

Compute the exact real sum of the input floating values or a certified high-precision enclosure `s*`.

### Conservative accuracy envelope

Under the standard round-to-nearest model and `n-1` additions in precision with unit roundoff `u`, a conservative absolute forward-error envelope is:

```text
gamma_k = k*u / (1-k*u),  where k=n-1 and k*u < 1
B(x)    = gamma_k * sum_i |x_i|
S_C3(x) = [s* - B(x), s* + B(x)]
```

Additional output casts or permitted lower-precision products/accumulation require their own certified terms. If the actual kernel violates the assumed arithmetic model, the evidence is `INVALID` for C3 or `REJECT` if that model is itself the normative requirement.

This interval is an output-accuracy contract, not necessarily the exact set of values reachable by every allowed reduction tree. Acceptance therefore means satisfaction of this conservative bound. If a governing specification requires membership in a narrower set of permitted evaluations, that narrower relation must replace the interval.

### Verdict and limitations

- candidate outside the certified envelope gives a correctness rejection under the declared model;
- candidate inside gives conformance for the covered input/model;
- this envelope is intentionally conservative and may miss meaningful accuracy regressions;
- a separate truth-relative non-degradation or application contract may be stricter, but it must state its own policy source.

This contract demonstrates that at least some floating operators can have non-arbitrary acceptable sets derived from numerical analysis rather than observed eager/compiled deltas.

## 4. Contract C4 — stochastic law and RNG semantics

### Two distinct promises

1. **Algorithm/replay promise:** if the specification fixes RNG algorithm, state consumption and mapping, `S_C4a(x, rng)` is an exact output/next-RNG-state relation.
2. **Law promise:** if only a conditional distribution is specified, `S_C4b(x)` is the target law or a specification-permitted set of laws.

### Verdict

- C4a can produce exact per-input conformance verdicts;
- C4b compares the candidate law with the target law under the declared metric;
- when the allowed law distance is exactly zero, finite sampling can provide evidence of difference but ordinarily cannot prove exact equality; acceptance requires exhaustive/formal analysis or a nonzero independently justified equivalence margin;
- same-seed token disagreement is not a law violation unless replay identity is part of the promise.

This contract prevents algorithmic RNG, implementation law shift and finite-sample uncertainty from being called one variance.

## 5. Contract C5 — stateful update/transition

### Domain

An operator with declared state `(parameters, optimizer/control state, mutable buffers, RNG state)` and declared inputs.

### Semantic envelope

Construct `S_C5(s,x)` as a product of:

- exact relations for discrete control/mutation/RNG obligations;
- certified numerical enclosures for floating state fields;
- a transition-law relation when the update is stochastic.

### Verdict

A violation of any mandatory conjunct rejects the covered transition contract. Passing only the parameter norm while optimizer state, skip flag or RNG state is wrong is not acceptance. If the contract omits state that can affect future observable behavior, its claim must be narrowed or marked uninstantiated.

### Relation to long-term training

C5 judges one specified transition. It does not claim that a passing transition guarantees identical long-run training or that a failing transition causes final accuracy loss. Those require separate stability/dynamics assumptions.

## 6. Contract C6 — baseline-relative floating compatibility

### When it is needed

Some operator APIs do not specify a sufficiently sharp floating semantic envelope. A release policy may still require that compilation does not materially degrade behavior relative to eager.

### Contract

If high-precision truth exists, compare truth-relative errors:

```text
Delta_e(x) = G(Z_C(x), Z*(x)) - G(Z_R(x), Z*(x))
```

Otherwise use a predeclared semantic/application loss between candidate and baseline, not raw tensor mean. The acceptable population set constrains average excess loss, tail exceedance and required strata separately.

### Claim label

C6 produces `BEHAVIORAL_EQUIVALENCE` or `NON_DEGRADATION`, never correctness solely because eager is the baseline. If the compatibility margin has no independent source, C6 is `UNINSTANTIATED`.

## 7. Coverage and non-coverage

| Semantic structure | Initial contract |
|---|---|
| metadata/discrete/control | C1 |
| ranking/routing on identical operand | C2 |
| arithmetic reduction | C3 |
| stochastic output/RNG transition | C4 |
| optimizer/AMP/mutable transition | C5 |
| underspecified floating API compatibility | C6 |

Elementwise transcendentals, contraction/matmul, normalization, backward and mixed-precision kernels are not “covered” merely by analogy to C3. They require their own arithmetic envelopes or compatibility contracts. The suite is deliberately explicit about this external-validity boundary.

## 8. Bias/variance output for every contract

After each primary conformance verdict, report the relative discrepancy decomposition only as explanation:

```text
average relative shift
input/signature-conditioned heterogeneity
exact-input runtime variability
finite-sample uncertainty
```

For C3/C5 with truth, also decompose truth-relative error. A contract violation remains a violation even if global relative bias is zero; a nonzero bias inside the semantic envelope remains conforming.

## 9. Status

C1 and C2 are immediately determinate wherever the governing specification is explicit. C3 is determinate under its restricted arithmetic assumptions. C4 and C5 are determinate for specified exact subrelations and otherwise require law/numerical envelopes. C6 is an operational policy and cannot be completed by theory alone without an external compatibility margin.
