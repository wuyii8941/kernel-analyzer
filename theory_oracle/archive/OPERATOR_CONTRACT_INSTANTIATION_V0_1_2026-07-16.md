# Operator Contract Instantiation v0.1 — 2026-07-16

> Purpose: turn the parameterized decision relation in `OPERATOR_ORACLE_V0_1_DEFINITION_2026-07-16.md` into genuine operator contracts. This is a theoretical contract catalog, not an implementation plan.

## 1. The non-negotiable distinction

An implementation discrepancy decomposition is not automatically the quantity that should be accepted or rejected.

```text
D(x,r) = Z_C(x,r) - Z_R(x,r)
```

supports explanations such as average relative shift, input heterogeneity and runtime variability. A correctness contract instead requires an error relative to independent semantics:

```text
E_C(x,r) = Z_C(x,r) - Z*(x)
```

or another specification-derived loss. `D` and its bias can be useful predictors and diagnostics, but they cannot detect a common eager/compiled error and can be distorted by cancellation or irrelevant output directions.

Therefore v0.1 uses this priority:

1. judge specification-backed semantic obligations directly;
2. judge truth-relative numerical error when an independent reference/bound exists;
3. otherwise judge a clearly labeled baseline-relative compatibility policy;
4. if no acceptable policy can be justified, return `UNINSTANTIATED` and retain only the discrepancy profile.

## 2. Source-strength ladder

Every contract records one of these source levels:

| Level | Contract source | Permitted claim |
|---|---|---|
| `S1` | normative language/operator/API semantics or formal specification | correctness/conformance within stated scope |
| `S2` | exact/high-precision mathematical reference plus independently justified error bound | numerical correctness within the bound |
| `S3` | application/deployment loss and independently fixed tolerance | semantic-impact acceptability |
| `S4` | predeclared compatibility policy relative to eager/reference | behavioral equivalence only |
| `S0` | no independent relation or tolerance | measurement only; `UNINSTANTIATED` |

The level cannot be upgraded by sample size. A million eager/compiled pairs do not turn `S4` into correctness evidence.

## 3. Core conformance contract

The strongest common core does not depend on a universal floating-point tolerance. For every operator instance, include every observable semantic obligation that the governing specification actually fixes:

```text
output arity and structure
shape and declared dtype/device semantics
integer/boolean/index/control outputs
mutation and aliasing behavior when program-observable
exception/error behavior when specified
RNG-state consumption/advancement when specified
declared special-value semantics
```

Each obligation is an exact or finite-relation contract. A valid mismatching witness rejects that obligation. If the API leaves an item unspecified, it must be marked `UNSPECIFIED`; eager behavior cannot silently promote it into normative truth.

This core is a genuine, determinate Oracle for specified discrete/structural semantics. It deliberately does not claim that arbitrary floating outputs must be bitwise identical.

### 3.1 Semantic envelope

For every admitted input, construct the set of permitted behaviors:

```text
S_o(x) = {z : the governing semantics permit output/transition z for x}
```

This is a singleton for exact semantics. It may be a numerical enclosure for explicitly permitted floating evaluation choices, a set of probability laws for a stochastic operator, or a state relation for a mutable operator. The per-input conformance question is whether candidate behavior belongs to `S_o(x)`.

This envelope is preferable to a generic tolerance because it derives allowance from semantics. If the framework does not specify permitted precision/reassociation/approximation choices strongly enough to construct it, correctness remains partial; a project-defined tolerance creates an `S3` or `S4` policy, not an `S1` correctness fact.

## 4. Contract A — exact/discrete operators

### Scope

Operators or output fields whose semantics are exact: integer/boolean predicates, indices, permutations, shapes, declared value-preserving movement, and exact control/state decisions.

### Contract

For each admitted input `x`:

```text
L_exact(x) = 1[Z_C(x) violates the specified relation Z*(x)]
A_exact = {L_exact(x)=0 for every covered x}
```

### Verdict

- one valid witness gives `REJECT` for the covered contract;
- all exhaustively covered inputs satisfying the relation give `ACCEPT` for that finite domain;
- sampled success gives `ACCEPT` only for observed cases and `INDETERMINATE` for an unproven universal domain;
- undefined specification behavior gives `UNINSTANTIATED`, not eager-relative correctness.

### Important edge cases

NaN payload, signed zero, alias identity, ordering of equal keys and exception timing are exact obligations only if the applicable specification fixes them. “It differed bitwise” is not sufficient by itself.

## 5. Contract B — deterministic numerical operators

### Scope

Elementwise mathematical functions, reductions, normalization, contractions, floating transformations, backward operators and other deterministic floating computations.

### Required reference

Use an independently defined `Z*(x)` appropriate to the mathematical problem: exact arithmetic, sufficiently high precision with certified rounding/error, an analytical result, or a formal/normative bound. Eager output alone is not `Z*`.

### Required geometry

Select geometry according to operator semantics, in this order:

1. a specification-mandated componentwise/ULP/error rule;
2. problem-conditioned forward or backward error;
3. invariant/residual error that characterizes the mathematical task;
4. an application projection/loss;
5. only as a weaker fallback, a scale-normalized norm with independently justified scale.

Raw tensor mean and an arbitrary `rtol/atol` are not valid defaults.

### Primary loss

Let:

```text
e_C(x,r) = G_o(Z_C(x,r), Z*(x); x)
```

where `G_o >= 0` is the declared numerical error geometry. The correctness contract may constrain:

```text
P_X(e_C(X) > epsilon_o(X)) <= p_o
quantile_q(e_C(X))         <= kappa_o
runtime variability of e_C <= nu_o
specified invariants and exceptional cases hold
```

`epsilon_o`, `p_o`, `kappa_o` and `nu_o` must come from `S1`--`S3`. If they do not exist, these fields cannot be inferred from candidate measurements.

When the numerical specification permits a set of evaluation strategies, define `S_o(x)` as a certified enclosure of every permitted result and use distance to that enclosure. For example, a floating reduction contract can derive a conservative envelope from the input floats, declared accumulation precision, rounding model, allowed operation count/order, and exceptional-value rules. This is a correctness envelope; a narrower “match eager” band is only compatibility.

### Optional non-degradation contract

When both candidate and eager can be compared with truth, define excess error:

```text
Delta_e(x,r) = e_C(x,r) - e_R(x,r)
```

A predeclared constraint on `Delta_e` tests numerical non-degradation. This is stronger than raw candidate-reference discrepancy because it can reveal which implementation is closer to truth, but it remains a compatibility policy unless the specification requires non-degradation.

### Diagnostic decomposition

After the primary loss is fixed, retain:

```text
relative discrepancy D
relative average bias B
input-conditioned heterogeneity H
exact-input runtime variability N
```

These explain how and where the error contract is approached or violated. They do not replace `e_C`.

## 6. Contract C — distributional/stochastic operators

### Scope

Sampling, stochastic rounding, dropout-like operations, randomized algorithms, or execution whose normative object is a conditional probability law.

### Contract

For input `x`, let `P*(.|x)` be the specified target law and `P_C(.|x)` the candidate law. Select a distribution metric from the event geometry:

```text
categorical/no outcome geometry: total variation
ordered/geometric outcomes: Wasserstein or specified transport cost
moments only: only if the specification constrains only those moments
```

The primary loss is:

```text
e_C(x) = Dist(P_C(.|x), P*(.|x))
```

with a specification/application-derived acceptable set. If the intended law is exactly computable and finite, compare against it directly; otherwise quantify estimation uncertainty and tail resolution.

### Coupling rule

A shared-RNG or same-seed disagreement rate is secondary and coupling-dependent. It may measure reproducibility under that coupling, but it is not the distributional correctness distance unless the contract explicitly defines that coupling as semantic.

### Verdict limitations

Failure to reject equality is not acceptance. If the evidence cannot place the distribution distance wholly inside or outside the acceptable set, return `INDETERMINATE`.

## 7. Contract D — state-transition operators

### Scope

Optimizer steps, gradient transformations, loss-scale/overflow controllers, RNG-state transitions, mutable buffers and other operators whose semantic output is a next state.

### Contract

Declare the complete observable transition state before comparison:

```text
T*(s,x,xi) -> s'
T_C(s,x,xi) -> s'_C
```

The primary loss is a specified transition relation or geometry:

```text
e_C = G_transition(s'_C, s'; s,x)
```

The contract must separately cover, where applicable:

- exact discrete decisions such as skip/update/overflow flags;
- parameter and optimizer-state numerical error;
- mutation/alias effects;
- RNG-state advancement;
- stochastic next-state law.

Comparing only parameter norm or loss is insufficient when omitted state can affect the next step.

## 8. Boundary/event impact contract

Decision boundaries do not define numerical correctness, but they can define application impact.

Let `phi` map the declared numerical or transition object to an event `E`. The impact contract compares:

```text
binary event:        marginal rate difference + paired disagreement if relevant
categorical event:   outcome-law distance + declared outcome cost
top-k/set event:     set loss/assignment cost + marginal inclusion effects
routing event:       assignment/load/capacity consequences
```

The impact acceptable set comes from `S3` or a normative event relation. Boundary distance may stratify or explain impact but cannot supply the tolerance by observing how close current cases happen to be.

## 9. Bias and variance constraints after instantiation

The user-visible decomposition is retained, but each term has a precise role:

| Object | What varies | Oracle role |
|---|---|---|
| truth-relative error | candidate versus independent semantics | primary correctness quantity when available |
| relative average bias | mean candidate-baseline discrepancy over `Q_o` | compatibility constraint or explanatory diagnostic |
| input heterogeneity | conditional mean effect across real inputs/signatures | stratification/tail/generalization constraint |
| runtime variability | exact-input repeat result under declared runtime randomness | repeatability/distribution constraint |
| sampling uncertainty | finite observed inputs/checkpoints/repeats | width of confidence set; never behavior itself |

No general rule says bias is harmful and variance is harmless. A zero-mean stochastic perturbation can violate a tail/event contract; a deterministic shift can remain safely inside a numerical bound.

## 10. Operator-family coverage matrix

The contract program covers semantic structures, not only operators where forks have already appeared.

| Operator structure | Primary contract | Mandatory secondary questions |
|---|---|---|
| structural/index/boolean/control | exact/discrete | unspecified edge semantics, alias/mutation |
| elementwise/cast/transcendental | numerical | special values, monotonicity/invariants if specified |
| reductions | numerical | conditioning, order policy, overflow, runtime nondeterminism |
| matmul/conv/contraction | numerical | precision mode, conditioning, layout/signature strata |
| normalization/softmax | numerical + invariants | probability/event impact where applicable |
| ranking/top-k/routing | exact relation over declared order policy + impact | ties, set/assignment geometry |
| stochastic sampling/dropout | distributional | RNG transition and coupling-dependent reproducibility |
| backward/gradient transform | numerical or transition | downstream update geometry |
| optimizer/AMP/stateful control | transition | exact decisions plus full mutable state |

This matrix is a completeness checklist. It does not assert in advance which family has bias or variance.

## 11. Input-population contract

`Q_o` is part of the meaning of the verdict, not merely a source of test cases.

| Population | Supported interpretation | Main limitation |
|---|---|---|
| reference-trajectory inputs | compatibility on states/operands reached by the baseline | may miss candidate-only operational states |
| candidate-trajectory inputs replayed to both sides | behavior on states/operands actually reached by the candidate | distribution depends on the candidate and may change across versions |
| predeclared mixture/union | symmetric workload claim with explicit weights | weights encode a policy and must not be tuned post hoc |
| external workload corpus | deployment/workload generalization | corpus representativeness must be justified |
| stress/adversarial inputs | counterexample discovery and robustness | cannot estimate natural prevalence without transport assumptions |

The recommended default is to issue separate verdicts for a nominal workload population and a stress population. Never mix them and report one prevalence. A correctness witness may reject a universal specification claim regardless of natural frequency; a workload-weighted compatibility claim must preserve its declared weights.

If training phase/checkpoint changes the operand law, it is a sampling stratum or part of the domain. Calling all such cases one IID input sample would understate uncertainty and hide phase-specific effects.

## 12. Randomness and configuration contract

Every potential source is assigned exactly one role:

| Source | Contract treatment |
|---|---|
| input/checkpoint selection | part of `Q_o`, not runtime variance |
| batch/token selection | part of the input sampling unit unless the algorithm specifies it as transition randomness |
| algorithmic RNG | semantic input or conditional-law randomness; coupling declared separately |
| GPU scheduling/atomic order | exact-input runtime randomness if allowed to vary across repeats |
| compiler autotuning choice | fixed configuration or explicitly sampled configuration distribution; never silently both |
| stochastic rounding | algorithmic/runtime randomness according to the specified semantics |
| fixed reduction tree/reassociation/cast placement | deterministic candidate mapping, contributing to conditional mean discrepancy |

Changing the randomness or autotuning policy creates a new contract version. A deterministic run can legitimately have zero runtime variability and nonzero input-dependent discrepancy.

## 13. Candidate-realization gate

An operator verdict requires evidence that the declared candidate realization executed. This is difficult when compilation fuses, decomposes or removes semantic operators.

- If semantic input/output correspondence can be observed without changing the candidate realization, the operator behavior contract is identifiable.
- If observation or replacement recompiles the neighborhood into a different computation, the evidence belongs to that new realization.
- If only a fused region can be identified, issue a region contract; do not relabel it as a constituent-operator verdict.
- Kernel identity is implementation provenance, not automatically semantic operator identity.

Failure to establish realization identity produces `INVALID` for behavior comparison or `NOT_IDENTIFIABLE` for causal impact.

## 14. Concrete contract record

A contract is ready for confirmation only when this record has no unresolved required field:

```text
ContractRecord = {
  operator_instance_and_signature,
  domain_and_input_population,
  semantic_obligations,
  source_level: S1 | S2 | S3 | S4,
  reference_construction_and_error,
  geometry_and_loss,
  acceptable_set_and_justification,
  randomness_and_coupling,
  sampling_unit_and_coverage,
  confidence_and_rare-event_resolution,
  validity_and_missing-data_rules,
  instance_or_family_quantifier,
  correctness_or_compatibility_claim
}
```

Any missing required field causes `UNINSTANTIATED`. This fail-closed rule is what makes the Oracle reliable rather than merely convenient.

## 15. Four mandatory sanity witnesses per contract

Before a contract sees confirmation data, construct:

1. **clear accept:** a conforming output safely inside the acceptable set;
2. **clear reject:** a semantic or numerical violation that must be caught;
3. **indeterminate:** evidence whose uncertainty overlaps the boundary;
4. **invalid:** wrong input pairing, fallback, missing/nonfinite mishandling or another failed validity gate.

Additionally test mean cancellation, rare tail failure, deterministic nonzero discrepancy, stochastic mean-zero failure and eager-wrong/compiled-right cases. A contract that gives the common-sense wrong answer is revised before real evaluation.

## 16. What is now genuinely available

The core conformance contract is directly instantiable whenever an external operator/API semantic obligation exists. Exact and discrete obligations can already produce determinate evidence-level verdicts without inventing a floating tolerance.

Numerical, distributional and transition contracts are also fully specified as decision forms, but a particular operator cannot receive `ACCEPT` or `REJECT` until its independent truth/policy source and acceptable set are recorded. This remaining work is semantic specification work, not additional variance decomposition.
