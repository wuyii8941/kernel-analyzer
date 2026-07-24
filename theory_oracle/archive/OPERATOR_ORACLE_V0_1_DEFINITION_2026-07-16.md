# Operator Oracle v0.1 Definition — 2026-07-16

> Status: candidate normative definition. This document defines an Oracle decision relation, not merely a discrepancy profile. Numerical tolerances must still be instantiated per operator contract; until then the result is `UNINSTANTIATED`, not pass or fail.

## 1. Plain-language definition

An optimized operator implementation is acceptable only when, on a declared population of real inputs, it satisfies a declared semantic or numerical contract with enough valid evidence.

The Oracle must answer five questions before it can judge anything:

1. **What is being judged?** One identified operator instance/signature and candidate implementation.
2. **What was promised?** Exact correctness, bounded numerical error, distributional equivalence or bounded next-state change.
3. **What counts as too different?** A tolerance justified independently of the observations being judged.
4. **What evidence is admissible?** Matched inputs, execution identity, repeats, uncertainty and coverage rules.
5. **What is the verdict?** `ACCEPT`, `REJECT`, `INDETERMINATE`, `INVALID` or `UNINSTANTIATED` by an explicit rule.

This is the missing zero-to-one step. Bias and variance help evaluate a contract; they do not define the contract.

## 2. The Oracle as a contract checker

For an operator instance `o`, define:

```text
Oracle(o, C_o, E_o) -> V_o
```

where `C_o` is a predeclared contract, `E_o` is collected evidence, and `V_o` is a structured verdict.

An operator instance is identified by at least:

```text
semantic operator + call-site role + phase
+ attributes + shape/dtype/layout/device signature
+ candidate/compiler configuration
```

The contract `C_o` is incomplete unless it contains all of the following:

| Contract field | Required meaning |
|---|---|
| `domain` | Admitted operator identity/signatures and exclusions |
| `Q_o` | Population or workload distribution of real operator inputs |
| `relation` | Exact, numerical, distributional or state-transition promise |
| `reference_status` | Specification, high-precision truth, baseline only, or another status |
| `semantic_envelope S_o(x)` | Input-conditioned set/relation/law permitted by the governing semantics |
| `geometry` | Semantically meaningful output comparison or loss |
| `estimands` | Predeclared population quantities to estimate |
| `acceptable_set A_o` | Allowed values of all required estimands |
| `evidence_protocol` | Sampling unit, repeats, coupling, confidence and power conventions |
| `validity_gates` | Input matching, execution identity, fallback, missing and nonfinite rules |
| `aggregation` | Instance-to-family claim rule, if a family claim is requested |
| `claim_label` | Correctness, behavioral equivalence, impact, or a combination |

If `relation`, semantic envelope/geometry or population acceptable set is absent, the relevant contract is `UNINSTANTIATED`. The system may still report measurements, but it must not issue an acceptance verdict.

## 3. Four relation kinds

There is no defensible universal scalar metric for all operators. v0.1 uses one common verdict mechanism with four contract types.

### 3.1 Exact relation

Use when the operator specification requires exact equality: for example, an exact shape/index/control result or a relation with an unambiguous exact representation.

```text
candidate output = specified output for every admitted input
```

A valid counterexample is enough to reject a universal exact claim. Passing sampled cases is not a proof for unseen inputs unless exhaustive testing or formal verification supplies that coverage.

### 3.2 Numerical relation

Use for deterministic floating-point numerical operators. The contract specifies a meaningful error geometry and budget, such as a scaled norm, componentwise bound, invariant residual, ULP bound where appropriate, or a problem-conditioned forward/backward error.

With independent truth `Z*`, candidate correctness error is:

```text
E_C(x,r) = Z_C(x,r) - Z*(x)
```

Without independent truth, eager/reference is only a baseline and the admissible object is:

```text
D(x,r) = Z_C(x,r) - Z_R(x,r)
```

The latter supports a behavioral-equivalence claim, not mathematical correctness.

### 3.3 Distributional relation

Use when the operator is intrinsically stochastic or its specification is a probability law. The contract compares conditional output laws using a metric selected for the semantics, such as total variation for categorical choices or Wasserstein distance when output geometry matters.

The primary object is the conditional distribution, not whether one coupled execution returned the same token. Coupled single-run disagreement may be a diagnostic but is not invariant to the coupling protocol.

### 3.4 State-transition relation

Use when the semantic object is an update or control transition: optimizer steps, AMP state changes, gradient transformations, RNG-state advancement, or another declared next state.

The contract compares the complete declared transition or a predeclared projection that is sufficient for the claimed semantics. A small forward-output discrepancy does not automatically imply an acceptable transition.

## 4. Where bias and variance enter

For a deterministic/numerical relative comparison, let `X ~ Q_o` be a real operator input case and `R` denote only the randomness retained by the execution protocol. Define the conditional mean discrepancy:

```text
m_o(x) = E_R[D_o(x,R) | X=x]
```

Then keep three different objects separate:

```text
B_o = E_X[m_o(X)]
H_o = distribution/covariance/tails of m_o(X) across X
N_o = E_X[Var_R(D_o(X,R) | X)]
```

- `B_o` is implementation-relative average shift unless independent truth exists.
- `H_o` is input-conditioned heterogeneity. It is not runtime randomness.
- `N_o` is exact-input runtime variability under the declared protocol.
- uncertainty from observing finitely many inputs/checkpoints is sampling uncertainty and belongs to confidence sets, not to `N_o`.

In deterministic execution, `N_o` can be zero while `B_o` and `H_o` are nonzero. A fixed reduction tree, fixed reassociation or fixed cast placement normally creates a deterministic mapping and therefore belongs to `m_o(x)`, hence to average shift and/or input heterogeneity. Floating-point computation is not inherently variance: it becomes runtime variance only when a declared random mechanism changes the result across exact repeats.

## 5. The acceptable set

The Oracle distinguishes two acceptable sets.

First, an input-conditioned **semantic envelope** `S_o(x)` states which outputs or transition laws the governing semantics permit for input `x`:

```text
exact semantics:         S_o(x) is a singleton value/relation
floating semantics:      S_o(x) is the set/enclosure allowed by declared precision,
                         rounding, reassociation and exceptional-value rules
stochastic semantics:    S_o(x) is a target law or permitted set of laws
transition semantics:    S_o(x) is the permitted next-state relation/law
```

Define a semantic violation or distance from this envelope:

```text
v_o(x,r) = 1[Z_C(x,r) not in S_o(x)]
e_o(x,r) = Dist(Z_C(x,r), S_o(x))
```

When external semantics fully determine `S_o`, this is the primary correctness object. It avoids inventing a generic `rtol/atol`: exact semantics give a singleton, while explicitly permitted floating choices give a non-singleton numerical envelope.

Second, the population-level acceptable set `A_o` states which estimand values over `Q_o` are allowed. A universal correctness claim normally requires zero valid violations; a workload/risk contract may instead contain independently justified conjunctive constraints such as:

```text
geometry(B_o)                         <= beta_o
P_X(loss(m_o(X)) > tau_o)             <= p_o
quantile_q(loss(m_o(X)))              <= kappa_o
runtime_noise_functional(N_o)         <= nu_o
distribution_or_transition_distance   <= delta_o
```

These are templates, not universal mandatory metrics. The contract selects only semantically justified components. No weighted sum is used by default because a weighted score could let a severe tail failure cancel against a harmless average. The input-level envelope and population policy must not be conflated: a rare specification violation can refute a universal claim even if its workload-weighted rate is small.

Acceptable bounds must come, in descending evidential strength, from:

1. a normative operator or application specification;
2. a rigorous numerical-error or conditioning argument;
3. high-precision/reference calibration plus an externally stated application tolerance;
4. a predeclared reference-relative compatibility requirement.

Reference/reference self-pair noise may establish a measurement floor, but it does not establish what is semantically acceptable. The candidate data being judged may not be used post hoc to choose a threshold that makes itself pass.

### 5.1 Contract precedence

Composite operator/program contracts are conjunctive and checked in this order:

```text
1. exact structural/option/index/state obligations
2. domain, special-value and exception obligations
3. numerical/distributional accuracy envelope
4. workload-level risk/compatibility constraints
5. downstream impact contract
```

An ignored option, wrong stride/index or missing state update is not forgiven because the final floating value happens to lie inside a loose numerical bound. Conversely, after exact obligations pass, ordinary nonzero floating discrepancy is judged by its numerical envelope rather than exact equality.

### 5.2 Three non-interchangeable output objects

Every instantiated result separates:

```text
theta_conformance = population functionals of v_o and e_o
theta_discrepancy = relative bias B_o, input heterogeneity H_o,
                    runtime variability N_o and their tails
theta_impact      = optional event/transition/application effect
```

`theta_conformance` is primary for a specification-backed correctness contract. `theta_discrepancy` explains direction, conditional structure and repeatability; it is primary only when an explicit baseline-compatibility contract says so. `theta_impact` answers a separate application-risk question. None may silently substitute for another.

This gives “error, bias and variance” precise roles:

- truth/specification-relative error measures distance from `S_o(x)`;
- implementation-relative bias summarizes the direction of candidate-baseline discrepancy over `Q_o`;
- input and runtime variation describe different axes of that discrepancy or truth-relative error;
- sampling uncertainty controls how confidently any population constraint can be judged.

## 6. Evidence and validity

The Oracle first determines whether evidence is admissible.

Required validity gates are:

- the same declared operator input and relevant control/RNG context were compared;
- the intended candidate implementation actually executed;
- output correspondence and comparison geometry are defined;
- fallback, missing outputs and nonfinite values follow predeclared rules;
- sampling did not select cases because a discrepancy or fork was already known;
- the statistical unit is the input/checkpoint cluster, not tensor coordinates treated as independent replicates;
- repeated executions vary only the randomness assigned to the repeat axis;
- all required contract endpoints and multiplicity rules were fixed before confirmation.

A failed gate produces `INVALID`, not a favorable or unfavorable behavior verdict.

For compiled/fused programs, validity additionally consumes the correspondence levels defined in [OPERATOR_REALIZATION_IDENTITY_CONTRACT_V0_1_2026-07-16.md](OPERATOR_REALIZATION_IDENTITY_CONTRACT_V0_1_2026-07-16.md). An isolated lowering, fused region and physical kernel are not silently promoted to an in-program semantic-operator result.

Evidence must quantify sampling uncertainty. Difference testing alone cannot establish acceptance: “not statistically significant” is not evidence of equivalence. Acceptance requires an equivalence/non-inferiority style confidence statement with enough precision relative to `A_o`. Tail and rare-event contracts additionally require a sample-size or coverage statement capable of detecting the declared rate.

## 7. Explicit verdict function

Let

```text
theta_o = T(Law(observations | X~Q_o, declared randomness protocol))
```

be the complete required estimand object, `A_o` its acceptable set, and `C_hat_o` a simultaneous confidence set with the contract's declared coverage.

The behavioral verdict is:

```text
UNINSTANTIATED  if relation, semantic envelope/geometry or A_o is missing
INVALID         if any required validity gate fails
ACCEPT          if C_hat_o is wholly contained in A_o
REJECT          if C_hat_o and A_o are disjoint
INDETERMINATE   otherwise
```

For an exact universal relation, a valid exact counterexample can produce `REJECT` directly. `ACCEPT` still means only the declared coverage level unless exhaustive or formal evidence is available.

This asymmetric rule is deliberate:

- a tiny but precisely estimated permissible difference can be accepted;
- a large but imprecisely estimated difference remains indeterminate rather than being waved through;
- lack of statistical significance never becomes automatic acceptance;
- a missing tolerance never becomes an implicit zero-tolerance or anything-goes rule.

## 8. Structured result, not one scalar

The emitted result is:

```text
OperatorOracleResult = {
  contract_id_and_version,
  operator_identity_and_signature,
  realization_correspondence_and_claim_scope,
  validity: VALID | INVALID,
  behavioral_verdict: ACCEPT | REJECT | INDETERMINATE | UNINSTANTIATED,
  claim_level: CORRECTNESS | BEHAVIORAL_EQUIVALENCE,
  conformance_estimands_and_confidence_set,
  discrepancy_decomposition: {relative_bias, input_heterogeneity, runtime_variability},
  violated_or_unresolved_constraints,
  impact_verdict: ACCEPT | REJECT | INDETERMINATE | NOT_TESTED | NOT_IDENTIFIABLE,
  coverage_scope,
  evidence_provenance
}
```

The `behavioral_verdict` is primary. `impact_verdict` is a separate, optional contract:

- a specification violation may be rejected even if no current workload impact is observed;
- a legal implementation difference may still create application risk;
- without an integrity-preserving operator intervention, impact is `NOT_IDENTIFIABLE`, not zero.

The overall human-facing conclusion is therefore a conjunction of an explicit claim label and verdict, for example:

```text
REJECT — behavioral-equivalence contract violated; correctness not established
ACCEPT — specification-backed numerical contract satisfied on declared workload scope
INDETERMINATE — evidence is too imprecise for the declared tail bound
INVALID — candidate execution identity was not established
```

## 9. Operator impact and causal attribution

Behavior comparison and in-program contribution answer different questions.

For a downstream endpoint/loss `Y`, define only under a valid intervention:

```text
repair effect    = effect of replacing candidate behavior of o with reference behavior
injection effect = effect of inserting candidate behavior/discrepancy of o into reference context
```

These are intervention-dependent causal effects. Repair can test whether the candidate behavior of `o` is necessary in that execution context; injection can test whether it is sufficient in the reference context. Neither proves unique root cause when there are substitute causes or interactions.

They may be asymmetric because later computation is nonlinear, reference and candidate contexts differ, or several discrepancies interact. If changing `o` also changes fusion, layout, scheduling or neighboring implementations, the treatment is no longer “operator `o` only.” The result must be labeled region/configuration sensitivity, and operator impact becomes `NOT_IDENTIFIABLE`.

Attribution should distinguish:

- a discrepancy-generating operator;
- an operator that propagates/amplifies/suppresses an existing discrepancy;
- an operator or control point that converts a continuous discrepancy into a semantic event.

These roles can coincide, but the Oracle does not assume that they do.

## 10. Instance and family claims

The primary verdict applies to an operator instance/signature. Family aggregation must state its quantifier.

### Universal family claim

“All admitted signatures satisfy the contract.” One valid rejected member refutes this claim. Sampled acceptance cannot prove the universal claim unless coverage is exhaustive or formally generalized.

### Workload-weighted family claim

“The family satisfies the contract under declared workload distribution `Q_family`.” The aggregation uses predeclared input/signature weights and tail constraints. Opposite shifts may cancel in an average, so family acceptance also requires every non-averaging safety/tail constraint to pass.

### Stratified family report

When signatures differ materially, report per-stratum verdicts. If coverage or weights are missing, the family result is `INDETERMINATE` rather than an average of convenient cases.

## 11. Correctness boundary

The Oracle may issue a correctness claim only when the expected relation is anchored by independent evidence:

- normative semantics;
- a high-precision or exact mathematical reference appropriate to the problem;
- a formal proof/certificate;
- or a previously established wrong-code relation.

Eager/compiled disagreement by itself supports only relative equivalence or impact claims. A persistent deterministic difference may be a compatibility risk or reproducibility issue without being a bug. Conversely, absence of observed downstream impact does not repair a proven specification violation.

## 12. Recommended default policy

Until operator-specific contracts are available, use these defaults:

1. use specification-backed correctness when independent truth exists;
2. otherwise use behavioral equivalence and explicitly name eager/reference as baseline only;
3. report downstream impact separately rather than requiring it to reject a correctness violation;
4. use conjunctive bias, tail and runtime-noise constraints instead of a weighted omnibus score;
5. use instance/signature verdicts as primary; make family claims only with a declared quantifier and coverage;
6. emit `UNINSTANTIATED` whenever no independent tolerance source exists.

## 13. What v0.1 has and has not defined

v0.1 defines:

- the judgment subject;
- the allowed relation types;
- the role of bias, heterogeneity, runtime variability and sampling uncertainty;
- the acceptable-set requirement;
- evidence validity;
- the verdict map;
- correctness/impact boundaries;
- instance-to-family claim semantics.

v0.1 does not pretend to provide one universal numerical threshold for every operator. Contract instantiation rules are defined in [OPERATOR_CONTRACT_INSTANTIATION_V0_1_2026-07-16.md](OPERATOR_CONTRACT_INSTANTIATION_V0_1_2026-07-16.md): specified structural/discrete semantics form the directly usable core, while floating, distributional and transition contracts require independently justified geometry and acceptable sets.

## 14. Theoretical provenance

This definition is a composition of established ideas, not a claim that every ingredient is new:

- software-testing Oracle and differential-testing work motivates the distinction between disagreement and correctness;
- numerical analysis motivates semantic error geometry, conditioning and independent high-precision references;
- metrology motivates keeping truth-relative bias separate from implementation-relative paired difference;
- equivalence/non-inferiority testing motivates confidence-set containment rather than treating nonsignificance as acceptance;
- repeated-measure and variance-component analysis motivates separating input heterogeneity, repeat variability and sampling uncertainty;
- distribution testing motivates comparing stochastic laws rather than isolated coupled outcomes;
- causal debugging motivates repair/injection while requiring treatment integrity.

The project-specific question is whether these pieces form a useful operator contract for compiled DL programs and whether the resulting verdict adds stable information beyond raw discrepancy. Sources, limitations and the novelty audit are recorded in [ORACLE_PRELIMINARY_SURVEY.md](ORACLE_PRELIMINARY_SURVEY.md). Gauge R&R is used there only as an experimental-design analogy; it does not supply the expected relation, acceptable set or correctness semantics of this Oracle.
