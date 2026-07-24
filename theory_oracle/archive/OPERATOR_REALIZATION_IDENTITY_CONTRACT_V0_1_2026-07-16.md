# Operator Realization Identity Contract v0.1 — 2026-07-16

> Purpose: determine when evidence actually supports an operator-level Oracle verdict after compilation, fusion, decomposition, elimination or layout transformation. This is a construct-validity contract, not an implementation-instrumentation design.

## 1. The identity problem

A semantic source operator and a physical compiled artifact are not generally one-to-one:

```text
one source operator -> several lowered operations/kernels
several source operators -> one fused kernel/region
source operator -> optimized away or algebraically absorbed
one kernel -> partial work for several semantic operators
```

Therefore the statement “kernel K differed” does not identify semantic operator `o`, and the statement “the graph contained o” does not prove a distinct candidate realization of `o` executed.

## 2. Four different subjects

| Subject | Definition | Permitted claim |
|---|---|---|
| semantic operator instance | API/program computation with declared input-output relation | operator conformance only if realization correspondence is identified |
| isolated lowered operator | candidate generated/evaluated outside original fused context | isolated-implementation behavior, not automatically in-program behavior |
| compiled region | composite candidate realization with a semantic boundary relation | region conformance/impact |
| physical kernel | executable implementation artifact | execution provenance and localization evidence, not semantic identity by itself |

The Oracle result must name exactly one subject. Results cannot be promoted upward by terminology.

## 3. Realization evidence levels

### `R0 — label only`

Evidence consists of a configuration string, requested compile mode or assumed kernel name.

```text
operator verdict: INVALID
```

### `R1 — source/graph presence`

The semantic operator is observed before lowering, but no evidence establishes how it maps to executed candidate work.

```text
source identity: established
candidate operator realization: NOT_IDENTIFIABLE
```

### `R2 — isolated candidate realization`

An isolated lowering/callable receives the declared matched operands and its execution identity is established. Its relation to the original in-program fused realization is not proven.

```text
isolated operator behavior verdict: valid
in-program operator verdict: not inferred
```

### `R3 — region realization`

An executed fused/decomposed region has a stable semantic input-output boundary corresponding to a composite specification, but constituent contributions are not separately identified.

```text
region conformance verdict: valid
constituent operator verdict: NOT_IDENTIFIABLE unless additional evidence exists
```

### `R4 — operator realization correspondence`

Evidence establishes that the declared candidate computation corresponding to operator `o` received the matched semantic operands and produced an observable result/relation, while observation did not change the realization relevant to the claim. Correspondence may be established by a standalone one-to-one implementation, translation-validation/certificate evidence, or another semantics-preserving provenance argument.

```text
operator behavior verdict: valid within the certified correspondence scope
```

`R4` does not automatically identify causal contribution to a downstream endpoint; that needs intervention integrity.

## 4. Observation invariance

Instrumentation can alter graph breaks, fusion, layout, scheduling, precision or kernel selection. Therefore every observation method must answer:

```text
Did observing the boundary preserve the candidate realization named in the contract?
```

- If yes with evidence, the observation is admissible.
- If observation creates a different compiled program, it defines a new contract/realization.
- If preservation cannot be determined, the original operator result is `INVALID` or `NOT_IDENTIFIABLE`.

Numerical equality of final outputs between instrumented and original programs is necessary but not always sufficient to prove internal realization identity; different implementations can agree on observed samples.

## 5. Behavior versus causal contribution

### Operator behavior question

Does a candidate realization satisfy `S_o(x)` on matched operator operands?

This can be answered at `R2` for an explicitly isolated-implementation claim or at `R4` for an in-program operator realization claim.

### In-program contribution question

Did candidate behavior of `o` cause a declared downstream conformance/impact difference in the original program?

This requires a treatment that changes only the semantic behavior assigned to `o` while holding the relevant non-target realization fixed.

Behavioral nonconformance does not by itself establish downstream contribution. Downstream repair does not by itself establish unique behavioral root cause.

## 6. Repair/injection integrity levels

For original candidate program `P_C`, a repair intends `do(o := reference behavior)` and an injection intends `do(o := candidate behavior)` in a reference context.

### `I0 — configuration replacement`

The intervention changes compiler flags, graph partitioning or broad implementation choices.

```text
claim: configuration sensitivity only
```

### `I1 — region replacement`

The intervention has a stable region boundary but changes several semantic computations.

```text
claim: region intervention effect
```

### `I2 — operator-boundary value intervention`

The intervention replaces/injects the value at an identified semantic operator boundary while downstream computation is demonstrably the declared fixed context.

```text
claim: boundary-value causal effect of operator output in that context
```

This can establish context-specific necessity/sufficiency of the value discrepancy, not necessarily which internal instruction generated it.

### `I3 — operator-realization intervention`

The semantic implementation of `o` alone changes while non-target code/precision/layout/fusion and relevant inputs are held fixed or their invariance is certified.

```text
claim: operator-realization causal effect in the declared context
```

This is the strongest operator attribution. It may be impossible for some fused programs.

## 7. Required invariants for operator causal language

Before saying “operator causal effect,” establish:

1. treatment identity: exactly what semantic mapping changed;
2. matched treatment input at the operator boundary;
3. non-target realization invariance or a valid mediation formulation;
4. stable downstream endpoint definition;
5. no intervention-induced fallback/recompile confound;
6. explicit context: compiled repair and reference injection are different interventions;
7. interaction policy for multiple operators and substitute causes.

If any required invariant fails, downgrade to boundary, region or configuration attribution.

## 8. Repair and injection interpretation

Under valid integrity:

- repair effect estimates removal of candidate operator behavior in the candidate context;
- injection effect estimates addition of candidate behavior in the reference context;
- repair supports context-specific necessity only relative to other behavior held fixed;
- injection supports context-specific sufficiency only in its receiving context;
- asymmetry is expected under nonlinear propagation, context differences and interactions;
- neither establishes unique root cause when alternative sufficient causes exist.

For several interacting operators, report joint/interventional effects or interaction terms. Do not force the total effect into additive per-operator shares without an additional attribution convention.

## 9. Three operator roles

Attribution reports distinguish:

1. **discrepancy generator** — first identified realization whose output violates or differs under its contract;
2. **discrepancy propagator/amplifier** — transforms an upstream discrepancy without being its source;
3. **boundary converter** — turns continuous discrepancy into a discrete event/control change.

An operator may occupy multiple roles. A boundary converter can be perfectly correct on its received input while exposing an upstream error.

## 10. Realization-aware Oracle result

```text
RealizationEvidence = {
  claimed_subject: semantic_operator | isolated_operator | region | kernel,
  source_identity,
  candidate_artifact_identity,
  correspondence_level: R0 | R1 | R2 | R3 | R4,
  observation_invariance,
  behavior_claim_scope,
  intervention_level: NONE | I0 | I1 | I2 | I3,
  causal_claim_scope,
  downgrade_reason
}
```

The main Oracle consumes this record as a validity gate. A numerical result cannot override insufficient subject identity.

## 11. Common-sense validation cases

| Scenario | Correct output |
|---|---|
| isolated eager/compiled op differs, original program fuses it | valid isolated behavior result; no in-program operator claim |
| fused region violates composite semantic envelope | region rejection; constituents unresolved |
| observing an intermediate disables fusion | result belongs to instrumented realization |
| replacing one output value removes downstream fork with downstream code fixed | boundary-value effect at I2 |
| changing compiler flag removes fork | configuration sensitivity at I0 |
| single-operator replacement also changes neighboring kernels | not I3; operator effect not identifiable |
| argmax correctly exposes upstream changed logits | boundary converter, not discrepancy generator |
| two operators can each independently cause the endpoint | repair/injection effects are context dependent; no unique cause |

## 12. Kill criteria

Do not issue an operator-level verdict if:

- semantic operands cannot be matched;
- candidate execution identity is only assumed;
- observation changes the realization without redefining the subject;
- only a fused region boundary is identifiable;
- a physical kernel is equated with an operator without correspondence evidence;
- causal intervention changes non-target compilation choices;
- final-output parity is the only evidence of internal treatment invariance.

The honest result is often `NOT_IDENTIFIABLE`. That is preferable to a precise-looking attribution to the wrong subject.
