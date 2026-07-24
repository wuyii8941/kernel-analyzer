# Training-Step Oracle v0.1 Definition — 2026-07-16

> Status: normative extension of Operator Oracle v0.1. It defines how operator contracts, a complete one-step state transition and downstream training impact coexist. It does not claim that every DL operator already has a quantitative contract.

## 1. Plain-language object

The Training-Step Oracle asks:

> Starting from one completely specified training state, does the candidate implementation produce a next-step behavior permitted by the training-program contract?

It does **not** ask only whether eager and compiled parameters are close. It also does not declare the whole step wrong merely because one internal intermediate differs.

The result has three separate ledgers:

```text
operator conformance   which local semantic contracts are satisfied or violated
step transition        whether the complete observable next state is permitted
training impact        whether decisions, updates or application behavior change
```

These ledgers answer different questions and must not be collapsed into one scalar.

## 2. Complete matched state

For one training step, declare

```text
S = (
  model parameters,
  model mutable buffers,
  optimizer parameters and state,
  batch/tokens/targets/masks,
  algorithmic RNG state,
  AMP/autocast/loss-scaling state,
  step counters and schedules,
  train/eval/dropout mode,
  framework/compiler/hardware/precision configuration
)
```

Anything capable of changing the next state belongs either in `S` or in the declared runtime randomness `r`. Missing a relevant field makes the comparison `INVALID`; it is not harmless measurement noise.

The reference and candidate must begin from bitwise-identical exact state fields. When an exact match is impossible, the initial-state relation and its permitted set must be declared explicitly before execution.

## 3. One-step observable

Let implementation `i` induce

```text
Z_i(S,r) = (
  exposed forward/loss values,
  gradients and gradient metadata,
  discrete decisions,
  update vector,
  next parameters,
  next optimizer state,
  next AMP/loss-scale state,
  next RNG state,
  next mutable buffers and counters,
  exception/skip/overflow status
)
```

The claimed observable state is frozen per subject. Omitting optimizer moments, RNG advancement or a skip flag while claiming “the next state matches” is invalid.

## 4. Binary core

Let `T(S)` be the independently justified set, relation or law of permitted one-step observables for state `S`.

Only after state identity, candidate execution identity and `T(S)` are established is the step bit defined:

```text
fail_step(S,r) = 1   when Z_C(S,r) is not in T(S)
fail_step(S,r) = 0   when Z_C(S,r) is in T(S)
```

Otherwise the bit is undefined and the result is one of:

```text
UNINSTANTIATED   the transition relation/envelope is missing
INVALID          state or candidate execution evidence is invalid
INAPPLICABLE     the promised configuration/path is unsupported
INDETERMINATE    valid finite evidence crosses the acceptance boundary
```

For a covered exact relation, one valid witness can produce `REJECT`. Finite all-zero witness bits do not prove a universal training-program claim.

## 5. Transition-contract fields

### 5.1 Exact fields

Check exact or finite relations before numerical tolerance:

- tensor/state structure, shape, dtype and alias/mutation obligations;
- optimizer option branches such as learning rate, `alpha`, weight decay, momentum and Nesterov;
- step counters, RNG advancement and schedules;
- overflow, clipping, skip-update and loss-scale decisions;
- required gradient/autograd metadata;
- exception and unsupported-domain behavior.

A wrong exact branch cannot be forgiven by a small final parameter delta.

### 5.2 Numerical fields

Parameters, gradients, optimizer moments and floating loss/state fields need an input-conditioned geometry and an independent envelope. Possible sources are:

- a documented update equation plus certified floating-error propagation;
- a high-precision transition reference;
- an independently justified application compatibility margin;
- a confirmed wrong-code relation for a covered case.

Without one of these, report truth/relative measurements but return `UNINSTANTIATED` for numerical acceptance. A global `rtol/atol` learned from the candidate results is not a contract.

### 5.3 Stochastic fields

If dropout, sampling, stochastic rounding or nondeterministic kernels are semantic, `T(S)` is a law or permitted family of laws. Compare the declared law; do not treat one same-seed outcome as the correctness object unless identical RNG coupling is itself specified.

Finite samples produce `ACCEPT`, `REJECT` or `INDETERMINATE` by a predeclared confidence-set rule.

## 6. How operator contracts compose—and do not compose

For each covered source operator instance `o_j` encountered at state `S`, Operator Oracle produces a local record:

```text
L_j = (subject, operands, semantic envelope, identity level, bit/verdict, coverage)
```

The training-step result includes the collection `L(S)={L_j}`. However:

1. local envelopes must not be multiplied into a Cartesian product and called a step envelope; dependencies and cancellation make that unsound or uselessly broad;
2. a local internal discrepancy does not automatically reject the observable program transition—compilers may legally transform or eliminate intermediates while preserving program semantics;
3. a direct API/operator contract violation remains a valid local rejection, but lifting it to a whole-step wrong-code claim requires either an observable transition violation or a semantics-preservation argument at the claimed realization level;
4. all covered local operators accepting does not prove the whole step accepts: missing operators, interactions, control flow and state mutation remain;
5. fused-region evidence cannot be relabeled as a unique source-operator result.

Therefore `L(S)` is a conformance/diagnostic ledger; `T(S)` is the primary whole-step correctness relation.

## 7. Impact is a third axis

Let `phi_k(Z)` define a declared application event or cost, for example:

- clipping/overflow/update-skip decision;
- selected token, top-k set or MoE routing assignment;
- update direction/norm or optimizer-state change;
- task loss, prediction or reward endpoint.

Impact compares `phi_k(Z_C)` with a reference law or application contract. It does not decide numerical correctness.

The required interpretation matrix is:

| Conformance | Impact | Meaning |
|---|---|---|
| accept | no material impact | admitted implementation difference |
| accept | material impact | legal numerical choice with reproducibility/risk consequence |
| reject | no observed impact | contract violation not propagated on covered states |
| reject | material impact | violation reaches declared training behavior |
| unresolved | any | compatibility/impact may be reported; correctness withheld |

## 8. Population over training states

One state produces a scoped witness. Population claims require a declared state distribution. Keep at least these strata separate:

```text
Q_R     states anchored on reference/eager trajectories
Q_C     states anchored on candidate/compiled trajectories
Q_X     external checkpoints, tasks or independently sampled states
Q_stress deliberately enriched numerical/boundary stress states
```

Do not pool them without operationally justified weights. Stress violation frequency is not deployment prevalence.

For each stratum report:

- valid exact/numerical violation rate and tail severity;
- covered acceptance, uninstantiated, invalid and indeterminate rates;
- implementation-relative average shift where orientation is meaningful;
- state/checkpoint/signature-conditioned heterogeneity;
- exact-state runtime variability;
- sampling uncertainty for every population claim;
- semantic/update impact rates and costs.

Bias and variance describe the distribution of evidence. They do not define `T(S)`.

## 9. Verdict aggregation

Every step result remains structured:

```text
validity/applicability
operator ledger and operator coverage
step exact-core verdict
step numerical/stochastic verdict
impact verdicts
compatibility discrepancy profile
state-population stratum and coverage
```

Fail-closed aggregation rules:

1. a valid exact step witness rejects the covered universal obligation;
2. a missing numerical envelope does not erase an exact rejection and does not become numerical acceptance;
3. `ACCEPT` applies only to the declared state/signature/configuration scope;
4. incomplete operator coverage is reported explicitly, not averaged away;
5. a global “training pass” is prohibited while any required contract component is uninstantiated, invalid or statistically indeterminate.

## 10. Long-run role

Matched-state one-step comparison identifies a local transition relation without feedback divergence. It cannot by itself identify convergence, time-to-quality or final accuracy.

Free-running training changes both the state distribution and future inputs after the first discrepancy. Long-run outcomes therefore validate operational consequence; they are not direct estimators of the local compiler effect.

Deriving long-run distributional conclusions from local transition differences requires additional stability, contraction/mixing, coupling and state-coverage assumptions. Without them, long-run training is a separate research problem.

## 11. Initial framework instantiation

The first scoped instantiation is a deterministic, controlled SGD step because its full state and documented update map are tractable:

```text
subject: one BERT supervised training step
state: model/buffers + frozen example/label + RNG/mode + complete no-momentum SGD config
exact core: state structure, gradients present, option branches, counters, mutation fields
numerical core: documented SGD map; quantitative gradient/update acceptance remains
                UNINSTANTIATED until an independent envelope is supplied
impact: prediction/loss/update differences, reported separately
```

Existing BERT/Qwen transition data establish implementation-relative gradient and controlled-update discrepancies with zero observed same-state runtime variability. They do **not** establish transition correctness because their numerical acceptable set was not independently instantiated, and the Qwen study did not replay historical Adam/GRPO state.

Adam/AdamW, AMP/loss scaling, gradient clipping, stochastic layers and production GRPO are later contract strata, not silently included in this first claim.

## 12. Validation controls

The step Oracle must pass controls covering:

- tiny output delta with an exact option/state violation;
- large but permitted local numerical difference with a conforming next state;
- zero eager/candidate discrepancy with shared wrong transition;
- correct forward values with wrong gradients or optimizer metadata;
- legal local operator differences that change an impact event;
- invalid fallback or incomplete matched state;
- finite stochastic evidence that must return `INDETERMINATE`;
- local operator violation whose effect is dead/cancelled, to prevent automatic whole-step overclaim;
- interacting local discrepancies for which no unique operator cause is identifiable.

## 13. Success and kill criteria

The scoped Training-Step Oracle succeeds only if it:

1. reconstructs complete declared state and candidate identity;
2. catches exact transition violations without averaging them away;
3. accepts independently certified legal transition differences;
4. refuses missing numerical/stochastic contracts;
5. adds information beyond loss/parameter raw delta and default allclose;
6. preserves operator/region/step and correctness/impact claim levels;
7. reproduces on a held-out state stratum within the claimed scope.

Narrow or reject the claim if its verdict is equivalent to a raw endpoint threshold, most required fields remain uninstantiated, state reconstruction cannot be verified, local bits are naively OR'ed into program correctness, or long-run divergence is used as proof of a local compiler violation.

## 14. Minimal result card

```text
Subject and complete state schema
State population/stratum
Reference/candidate and realization identity
Transition relation T(S) and authority
Operator conformance ledger + coverage
Exact transition verdict
Numerical/stochastic transition verdict
Impact verdicts
Bias/heterogeneity/runtime/sampling-uncertainty profile
Claim scope and explicit non-claims
```
