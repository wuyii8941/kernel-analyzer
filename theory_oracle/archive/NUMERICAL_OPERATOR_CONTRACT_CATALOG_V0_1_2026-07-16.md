# Numerical Operator Contract Catalog v0.1 — 2026-07-16

> Purpose: prevent one reduction-style error metric or one global `rtol/atol` from being applied to every floating operator. Each family below defines the semantic object, preferred geometry, admissible envelope source and failure boundary.

## 1. Common numerical contract

For an input case `x`, every numerical contract must state:

```text
f_o(x)        mathematical/specification object
Z*(x)         exact or certified high-precision reference/enclosure
G_o           semantically meaningful error geometry
S_o(x)        accepted output-accuracy envelope
Cfg_o         dtype, accumulation, rounding, precision and backend policy
Q_o           nominal/stress input population and conditioning strata
```

The candidate conformance loss is distance to `S_o(x)`. Relative eager/compiled bias is reported separately.

The standard floating model and bounds below require their usual assumptions: declared round-to-nearest arithmetic, bounded unit roundoff, no unhandled overflow/underflow, and known operation/precision structure. If these assumptions do not match the candidate realization, the contract is invalid or must use a broader model.

Primary numerical-analysis grounding and limitations are summarized in [ORACLE_PRELIMINARY_SURVEY.md](ORACLE_PRELIMINARY_SURVEY.md), especially its reference to Higham's *Accuracy and Stability of Numerical Algorithms*.

## 2. Cast and representation conversion

### Semantic object

Conversion of an exactly represented source value to a declared destination representation, including the applicable rounding, overflow, subnormal, saturation and special-value policy.

### Preferred envelope

- singleton correctly rounded destination value when conversion semantics are fixed;
- finite set when multiple rounding modes/flush policies are explicitly allowed;
- exact structural checks for dtype/shape and special-value class where specified.

### Failure boundary

Wrong destination value outside the permitted conversion relation is a correctness violation. Cast placement in a larger fused expression is not “cast runtime variance”; it changes the deterministic expression/configuration and must be judged at the corresponding expression/operator contract.

### Uninstantiated case

If the governing framework permits backend-dependent conversion behavior without a quantitative relation, only documented exact fields and an external compatibility policy are available.

## 3. Elementary arithmetic and elementwise algebra

### Semantic object

One specified floating operation or a specified elementwise expression.

### Preferred envelope

- for a single correctly rounded operation, the permitted rounded result relation;
- for a multi-operation expression with allowed reassociation/FMA, the union or certified enclosure of permitted evaluation forms;
- componentwise special-value and exception rules.

### Failure boundary

An output outside the certified expression envelope rejects numerical conformance. A large absolute discrepancy near a large-scale result may be conforming; a one-bit error in an exact predicate or special-value class may not be.

### Key risk

Treating an entire fused expression as the same semantic operator as one source-level operation can silently change the contract. Operator identity and permitted transformations must be explicit.

## 4. Transcendental and approximate elementwise functions

### Semantic object

The mathematical function over its declared domain plus special-value/domain behavior.

### Preferred geometry

- ULP or componentwise error only when a library/framework guarantee supports it;
- otherwise high-precision forward error with an externally justified accuracy envelope;
- exact checks for domain, sign, monotonicity or special values only where those properties are normative.

### Uninstantiated boundary

A high-precision reference alone tells us the candidate error but not how much error the API permits. Without a documented bound or external application/compatibility policy, the Oracle may report truth-relative error and reject exact semantic violations, but floating acceptance remains `UNINSTANTIATED`.

## 5. Reductions

### Semantic object

Exact real reduction of the input floating values, followed by the declared output representation, or another explicitly specified reduction semantics.

### Preferred geometry

- absolute forward error scaled by `sum |x_i|`, avoiding division by a near-zero exact sum;
- condition number/cancellation stratum;
- special-value/overflow behavior;
- exact-input runtime distribution when the reduction order may vary.

### Envelope

Use an operation-count/precision bound such as the conservative `gamma_{n-1} sum|x_i|` contract from C3, or a sharper tree-specific certified bound when execution identity proves the tree/model. Do not use the tighter bound while merely assuming a balanced tree.

### Bias/variance interpretation

A fixed tree is a deterministic mapping and contributes to conditional shift/heterogeneity. A varying atomic/order schedule contributes runtime variability. Either may violate the same numerical envelope.

## 6. Dot products, matmul and convolution

### Semantic object

Each output coordinate is a dot product over declared input values, with layout/batching not changing the underlying mathematical object unless the API says otherwise.

### Preferred geometry

For coordinate `ij`, use exact/high-precision dot product `c*_{ij}` and a conditioning scale:

```text
s_ij = sum_l |a_il * b_lj|
```

Under a known homogeneous accumulation model, derive a coordinate bound from multiplication/addition/FMA counts and unit roundoff. Aggregate with maximum/tail and matrix norm only after coordinate bounds are defined.

### Precision envelope

- full-precision product and accumulation: corresponding certified dot-product bound;
- FMA: an FMA-specific bound;
- TF32 or mixed precision: explicitly quantize inputs/products according to the declared mode and bound accumulation separately;
- a configuration that permits multiple modes: union/enclosure of those modes, or separate strata if the union is too uninformative.

### Failure boundary

Raw `max_abs` without conditioning can overflag large-scale coordinates and miss catastrophic relative error under cancellation. Batched-versus-sliced differences are not automatically violations because PyTorch documents they may differ; correctness requires an independent envelope, and compatibility requires a declared policy.

## 7. Normalization and softmax-like operators

### Semantic object

The complete normalized vector/tensor function, including axes, epsilon, dtype/accumulation and special-value behavior.

### Preferred geometry

- high-precision vector result under the mathematical definition;
- invariant residuals such as normalization constraints as necessary but not sufficient checks;
- probability geometry such as total variation for softmax when downstream semantics are categorical;
- boundary/ranking impact as a separate impact contract.

### Failure boundary

Passing `sum(probabilities)≈1` does not prove softmax correctness: many wrong distributions satisfy the invariant. Conversely, a common logit shift may be irrelevant to softmax but is not irrelevant to contracts that expose logits themselves. The geometry follows the operator's declared output semantics.

### Envelope source

Use certified interval/error propagation, an operator-specific documented bound, or an external compatibility/application margin. A generic tensor norm alone is insufficient.

## 8. Linear algebra solvers and decompositions

### Semantic object

The mathematical problem/relation, which may admit non-unique representations such as eigenvector signs, bases in repeated eigenspaces or factor permutations.

### Preferred geometry

- backward error or equation residual;
- orthogonality/factorization residual;
- subspace distance rather than coordinate equality for non-unique bases;
- condition number and problem-domain validity;
- forward error only when conditioning makes it meaningful.

### Failure boundary

Large coordinate difference between two valid decompositions need not be a violation. Small residual with an ill-conditioned problem may coexist with large forward difference. A raw eager/compiled tensor comparison is therefore not a valid universal Oracle for this family.

## 9. Backward and gradient operators

### Semantic object

The derivative, subgradient convention or vector-Jacobian product specified for the forward operator, including saved state and nondifferentiable-point policy.

### Preferred geometry

- high-precision/analytical derivative where available;
- adjoint/directional-derivative identity under carefully chosen perturbation scale;
- vector-Jacobian product error in task-relevant and adversarial directions;
- exact checks for declared zero/undefined/subgradient cases.

### Failure boundary

Finite differences alone are not a truth Oracle without truncation/rounding error control. A correct but different subgradient at an unspecified nondifferentiable point must not be rejected. Gradient norm alone can miss directionally severe errors.

## 10. Mixed-precision composite and fused operators

### Semantic object

The composite computation plus an explicit precision schedule: input quantization, intermediate dtype, accumulator precision, reassociation/FMA, special-value and output-cast behavior.

### Contract rule

Different precision schedules are distinct configurations. If several are normatively allowed, either:

1. form a justified union/envelope and accept its lower discriminating power; or
2. stratify them and issue configuration-specific verdicts.

Do not pool them into runtime variance. Fusion is not itself an error source category; it changes the realized expression/precision/schedule, whose semantic envelope must be analyzed.

## 11. Family verdict fields

Every numerical family result contains:

```text
truth/specification source strength
domain and excluded exceptional cases
candidate realization and precision schedule
semantic envelope or UNINSTANTIATED reason
per-input conformance distance/violation
population tail and stratum verdicts
relative bias/input heterogeneity/runtime variability
impact status
coverage and claim quantifier
```

## 12. Kill criteria for a numerical contract

Reject or narrow the contract if:

- the reference error is not certified below the contract resolution;
- the geometry changes the verdict under an irrelevant representation choice;
- the arithmetic assumptions do not cover the realized kernel;
- excluded overflow/underflow/NaN cases silently enter the data;
- a residual/invariant admits obvious wrong results that the contract calls correct;
- an accuracy margin is selected from candidate observations;
- family aggregation hides a rejected signature or conditioning stratum;
- the contract cannot outperform a raw delta ranking on seeded semantic violations.

## 13. Current status

This catalog defines defensible contract forms for the main numerical structures. It does not claim that PyTorch supplies quantitative promises for all of them. Where a governing bound is absent, the result remains a truth-relative measurement or `S4` compatibility contract, not correctness acceptance.
