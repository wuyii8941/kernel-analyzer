# Operator Oracle v0.1 Sanity Audit — 2026-07-16

> Target: [OPERATOR_ORACLE_V0_1_DEFINITION_2026-07-16.md](OPERATOR_ORACLE_V0_1_DEFINITION_2026-07-16.md)

## 1. Common-sense tests

| Case | Required v0.1 result | Why |
|---|---|---|
| Candidate differs by a tiny, precisely bounded amount inside an independently justified tolerance | `ACCEPT` | Statistical detectability is not the same as practical violation |
| No statistically significant difference, but the confidence interval extends far beyond tolerance | `INDETERMINATE` | Failure to reject equality is not evidence of equivalence |
| A deterministic fixed reduction choice gives the same nonzero discrepancy on every exact repeat | nonzero shift, zero runtime variability | Floating-point discrepancy is not automatically variance |
| Mean discrepancy is zero because two input strata have opposite shifts, but one stratum violates a tail/safety constraint | `REJECT` or per-stratum rejection | Global cancellation must not hide a relevant failure |
| Mean discrepancy is large but always lies in a semantically irrelevant invariant direction | determined by declared geometry; possibly `ACCEPT` | Raw magnitude is not the semantic contract |
| Mean discrepancy is tiny but consistently aligned with a decision boundary and violates a transition/event tolerance | behavior may pass while impact rejects | Behavior and application impact are distinct axes |
| A single token differs under two independent sampling RNG draws while categorical laws are equal | distributional `ACCEPT` if evidence is sufficient | Single coupled outcomes do not define distributional correctness |
| A categorical distribution changes beyond tolerance although a finite sample happens to contain no token disagreement | `REJECT` or `INDETERMINATE`, never automatic accept | The law, not a lucky sample, is the object |
| Eager and compiled disagree, but neither is independently specified as correct | behavioral-equivalence result only | Disagreement does not identify which implementation is wrong |
| High-precision truth shows eager is worse and compiled is inside the specification | compiled may `ACCEPT` correctness | Eager is not privileged as truth |
| Region repair removes downstream drift but also changes fusion/layout | operator impact `NOT_IDENTIFIABLE` | The intervention did not isolate one operator |
| One admitted signature has a reproducible exact wrong-code witness while many common signatures pass | universal family claim `REJECT` | Population averaging cannot erase a universal counterexample |

All cases are handled by the explicit contract fields and verdict rule. None can be resolved by bias/variance decomposition alone.

## 2. Conceptual leakage audit

### Measurement versus judgment

**Pass:** `B_o`, `H_o`, `N_o`, tails and confidence intervals are measurements. They become judgment inputs only after `A_o` is declared.

### State/input heterogeneity versus runtime variance

**Pass:** exact-input repeat randomness is the only source of `N_o`; variation across real input cases is `H_o`; finite-data imprecision is carried by the confidence set.

### Reference equivalence versus correctness

**Pass:** the result carries a `claim_level`. Baseline discrepancy cannot silently become mathematical error.

### Numerical behavior versus downstream impact

**Pass:** separate verdict fields prevent “different therefore harmful” and “not currently harmful therefore correct.”

### Operator versus region/kernel

**Pass:** operator identity is semantic. Failed single-operator intervention integrity produces `NOT_IDENTIFIABLE`; it does not transfer a region effect to an operator.

### Instance versus family

**Pass with an important limitation:** family verdicts require an explicit universal or workload-weighted quantifier. Statistical generalization to unseen signatures remains a separate modeling problem.

## 3. Statistical reliability audit

| Risk | v0.1 defense | Remaining requirement when instantiated |
|---|---|---|
| Tensor coordinates treated as independent samples | input/checkpoint cluster is the statistical unit | choose cluster-aware uncertainty method |
| Paired eager/compiled runtime noise is correlated | estimate the paired discrepancy directly | preserve or declare RNG/execution coupling |
| “No significance” declared equivalent | confidence-set containment in acceptable set | choose equivalence margin and adequate power |
| Many endpoints searched until one fails | endpoints and simultaneous coverage predeclared | multiplicity method or joint confidence set |
| Rare errors hidden by means | tail/exceedance constraints can be mandatory | sample size capable of resolving target rate |
| Threshold fitted to observed candidate errors | independent tolerance-source hierarchy | versioned contract frozen before confirmation |
| Dataset shift changes the meaning of bias | `Q_o` is part of contract identity | new distribution requires a new/updated verdict |
| Fallback makes candidate secretly execute reference | execution identity is a validity gate | fail-closed implementation evidence |
| Missing/nonfinite cases dropped | predeclared handling required | count them as violations or invalidate as specified |

The definition deliberately does not mandate a t-test, pooled variance formula or Gaussian random-effects model. Those are optional estimators whose assumptions must match the instantiated protocol. The estimand and verdict remain valid even when a different estimator is required.

## 4. Counterexamples that kill simpler Oracles

### Raw numerical delta Oracle

Fails when a large discrepancy lies in an invariant direction or a small discrepancy aligns with a sensitive boundary. v0.1 requires declared geometry and allows a separate impact contract.

### Global mean-shift Oracle

Fails under opposing strata, rare catastrophic tails, and mean-zero runtime instability. v0.1 permits conjunctive average, tail, heterogeneity and runtime constraints.

### Single-fork Oracle

Fails because one coupled observation mixes boundary proximity, implementation effect and algorithmic randomness. v0.1 judges a declared deterministic relation or conditional event/distribution law.

### Statistical-difference Oracle

Fails because any tiny difference becomes significant with enough samples, while low power makes a large difference look harmless. v0.1 tests membership in an independently meaningful acceptable set.

### Repair-only root-cause Oracle

Fails under substitute causes, interaction, altered fusion and context asymmetry. v0.1 reports intervention-dependent impact and withholds operator causality when integrity fails.

## 5. What remains unresolved

The definition is internally complete as a parameterized Oracle family, but a concrete operator verdict is impossible until these choices are made:

1. operator-specific semantic geometry;
2. independently justified tolerance/acceptable set;
3. target real-input population and family quantifier;
4. required confidence/power and tail resolution;
5. whether an impact contract is required for the intended release policy.

This is not a defect to hide. It is exactly where scientific and application assumptions enter. If these choices cannot be justified, the honest output remains a discrepancy measurement profile plus `UNINSTANTIATED`.

## 6. v0.1 decision

The Oracle now has a real decision form and survives the principal conceptual counterexamples. It is not yet a universal ready-to-run Oracle because acceptable numerical contracts are operator- and use-case-dependent. The next step is not to collect more undirected data; it is to instantiate and challenge the contract for a small coverage-balanced set of operator semantics, then test whether those choices generalize.
