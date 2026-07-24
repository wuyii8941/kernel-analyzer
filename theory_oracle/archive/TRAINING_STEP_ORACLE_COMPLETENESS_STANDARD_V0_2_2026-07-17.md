# Training-Step Oracle Completeness Standard v0.2 — 2026-07-17

## 1. Why a second standard is needed

“The Oracle is complete” has two different meanings that must not be merged.

1. **Decision-procedure completeness**: for every declared input, the procedure
   returns a justified verdict or a justified refusal. Missing truth, invalid
   execution and inadequate evidence are explicit outcomes rather than silent
   passes.
2. **Subject-instantiation completeness**: for one concrete training program,
   every endpoint that the claim requires has an independent contract and enough
   valid evidence to apply it.

The current framework is close to the first meaning. No current real-model subject
meets the second meaning for universal numerical correctness, stochastic training,
training impact, operator causality and population generalization simultaneously.
This is a scope statement, not a reason to invent a tolerance.

## 2. Required input contract

An Oracle query is the tuple

```text
O = (S, R, C, Z, Q, P_r, T, Phi, A)
```

where:

- `S`: complete matched-state schema;
- `R`, `C`: reference and candidate implementations, including realization
  identity;
- `Z`: complete one-step observable;
- `Q`: named state population or finite-bank stratum;
- `P_r`: runtime/algorithmic randomness and coupling protocol;
- `T`: independently justified permitted transition relation or law;
- `Phi`: declared semantic/application endpoints and acceptance rules;
- `A`: optional intervention contract for attribution.

A field may be declared absent only when its ledger is outside the claim. An
undeclared but outcome-relevant field makes the query `INVALID`.

## 3. Required output record

The output is structured; a single global bit is prohibited.

| Ledger | Required output |
|---|---|
| validity | `VALID`, `INVALID` or `INAPPLICABLE`, with failed gates |
| candidate identity | graph/path/realization evidence and coverage |
| exact transition | `ACCEPT`, `REJECT` or refusal |
| numerical transition | `ACCEPT`, `REJECT`, `INDETERMINATE` or `UNINSTANTIATED` |
| stochastic transition | law-level verdict or `NOT_IN_SCOPE/UNINSTANTIATED` |
| operator conformance | semantic-operator instance verdicts, R4 identity and covered/uncovered accounting |
| semantic/training impact | one verdict per predeclared endpoint |
| discrepancy profile | average relative shift, state heterogeneity and within-state runtime variability |
| variability sources | explicit classification of state/batch sampling, algorithmic RNG, execution nondeterminism, autotuning, stochastic rounding and fixed implementation effects |
| sampling uncertainty | uncertainty tied to the actual state-sampling unit |
| attribution | intervention-specific effects plus integrity and interaction status |
| correctness claim | authority, covered scope and explicit non-claims |

`UNINSTANTIATED`, `INVALID`, `INAPPLICABLE` and `INDETERMINATE` are distinct.
None may be coerced to `ACCEPT` or zero.

A correctness authority is itself auditable data. It must name its kind, source,
covered scope, independence from candidate measurements and whether the acceptance
rule was frozen. A citation string or a tolerance fitted to observed candidate
outputs does not instantiate `T(S)`. These fields make circularity visible; they do
not replace external review of whether the authority is scientifically sound.

## 4. Three non-substitutable questions

### 4.1 Correctness

Does the candidate observable belong to an independently permitted set `T(S)`?
Eager/candidate disagreement alone does not answer this.

### 4.2 Compatibility or impact

Does the candidate change a declared application or training event relative to a
baseline, or violate an application acceptance rule? This can be answered without
knowing which implementation is mathematically correct.

### 4.3 Discrepancy structure

How does candidate-minus-reference behavior vary over states and repeated
executions? Average shift, state heterogeneity and runtime variability describe the
evidence; they do not manufacture the correctness relation.

All three may be reported together, but no one is a proxy for the other two.

An impact claim also requires execution-context transport to be validated.  A
semantic event observed by a `no_grad`, inference, logging or auxiliary probe is
not automatically the event realized by the grad-enabled transition.  Autograd
specialization, AMP/Accelerate output conversion, training/evaluation mode and
compiler specialization are outcome-relevant parts of the implementation and
observable contracts.  If the selected event does not reproduce in the context
that computes the downstream endpoint, the impact experiment is invalid rather
than a zero-effect result.

Compiler state is not exhausted by configuration flags.  Shape-specialization
history, graph/cache state and autotuning selections are part of the matched
execution state when they change the realized candidate.  A query must either
condition on and reconstruct them, deliberately randomize/sample them, or declare
them uncontrolled.  Calling a cold compile and a history-warmed compile the same
fixed implementation without testing transport is an incomplete state contract.

Operator conformance is an additional local ledger, not a fourth proxy. A terminal
operator verdict requires a semantic operator with R4 correspondence to the
realization, an independent semantic contract and operand/result evidence. Region
and kernel observations remain their own units. Local operator bits are never
OR'ed into a whole-step correctness verdict; incomplete coverage is explicit.

## 5. Bias and variance terminology gate

The unqualified terms `compiler bias`, `floating-point variance` and `total error`
are forbidden in result cards.

Use:

- **average implementation-relative shift** for the mean conditional discrepancy
  under the named `Q` and `P_r`;
- **state-conditioned heterogeneity** for stable variation across states;
- **within-state runtime variability** for repeated-execution variation at exactly
  matched state and protocol;
- **sampling uncertainty** for uncertainty from observing finitely many state
  sampling units.

For each discrepancy estimand, the query must additionally predeclare:

- the observable being compared;
- the oriented comparison or nonnegative relation between candidate and reference;
- the aggregation/conditioning unit and weighting implied by it;
- the scalar, vector, set or distribution geometry.

Evidence must report an estimate together with its finite-bank scope or uncertainty.
The names above do not by themselves define an estimand: for example, token-weighted
and state-weighted means are different average shifts, and an L2 norm has no signed
direction. An undeclared post-hoc discrepancy component is invalid rather than an
optional extra result.

Every executable query must also name its population kind, selection design,
aggregation rule and explicit strata. Each state stratum records provenance, an
inclusion rule and a reference-trajectory, candidate-trajectory, externally frozen
or synthetic anchor. A separate enrichment field records whether selection is
natural/unconditioned, event-conditioned, boundary-stress or synthetic-control.
These are orthogonal: a reference-anchored event witness is not a natural
reference-state prevalence sample. An operator-coverage population is labelled
separately. Missing provenance, enrichment or aggregation makes the query invalid.
Changing `Q` may change the average shift; it does not establish that a universal
compiler bias appeared or disappeared.

Fixed reassociation, cast placement, reduction trees, fusion and layout choices are
deterministic implementation effects unless the declared protocol randomizes or
nondeterministically selects them. Their effects may contribute to the average or
state-conditioned terms. They are not runtime variance merely because they involve
floating point.

Every subject must also classify each potentially active variability source. State
and batch/token sampling belong to the state-population design; algorithmic RNG,
execution nondeterminism, autotuning selection and stochastic rounding may create
within-state randomness only when the declared protocol actually permits them;
fixed eager/compiled realizations are deterministic implementation effects.
Unmeasured sources remain `UNKNOWN`, not zero. Sampling uncertainty remains a
separate consequence of observing finitely many state-sampling units.

## 6. Semantic endpoint gate

For a binary event `E`, report both:

```text
directional shift = Pr(E_C=1,E_R=0) - Pr(E_C=0,E_R=1)
disagreement      = Pr(E_C != E_R)
```

For categorical or set-valued events, report the probability of inequality plus a
predeclared event-specific distance or cost. For stochastic sampling, compare the
selection laws first; a single sampled token is only a coupling-specific outcome.
Continuous scalar impact requires an oriented effect plus a declared cost. Vector
update impact requires its geometry and distance/cost; no scalar direction is
invented unless the endpoint itself predeclares a meaningful projection.

Every event result must include the reference margin/boundary geometry when a
natural margin exists. An event rate without boundary exposure is incomplete as an
explanation, although it may still be a valid compatibility endpoint.

## 7. Attribution gate

Repair and injection produce causal effects only for the intervention actually
executed. An operator-level causal claim additionally requires:

1. treatment execution and realization identity;
2. preservation of non-target compilation choices, or inclusion of those choices
   in the treatment definition;
3. anchor parity with the original eager and monolithic compiled programs;
4. repair, injection and interaction reporting;
5. replication on a held-out state/checkpoint stratum.

If fusion, layout, scheduling or graph partition changes, report
`intervention-dependent attribution`. Do not name a unique operator root cause.
Source-of-discrepancy, propagation and boundary-conversion roles are separate and
may belong to different operators.

## 8. Scoped completion levels

| Level | Evidence required | Permitted claim |
|---|---|---|
| L0 mechanics | state and candidate identity, refusal behavior | valid measurement machinery |
| L1 exact conformance | independently specified exact fields and controls | covered exact-transition verdict |
| L2 discrepancy | paired continuous observables and repeat protocol | implementation-relative profile |
| L3 impact | predeclared semantic endpoint and confirmation bank | scoped compatibility/impact verdict |
| L4 numerical/stochastic conformance | independent envelope/law and decision rule | scoped correctness verdict |
| L5 attribution | valid interventions, anchors, interactions and confirmation | scoped causal or intervention-dependent attribution |
| L6 population | operational `Q`, probability/cluster design and coverage | population-level rate or risk |
| L7 long-run | transition-to-trajectory assumptions or separate validation | long-run consequence, not local effect by default |

Levels are not a single maturity score. A subject can reach L4 correctness without
L3 application impact, or L3 impact while L4 remains uninstantiated.

## 9. Current evidence under this standard

| Subject | Exact | Discrepancy | Impact | Numerical correctness | Attribution | Population |
|---|---|---|---|---|---|---|
| BERT materialized SGD | covered `ACCEPT` plus stale-counter `REJECT` control | instantiated | strict prediction compatibility `ACCEPT` on finite bank | `UNINSTANTIATED` | operator coverage 0/9 R4 families; segmented layer-0 attribution complete; monolithic parity failure blocks original-program cause | deterministic finite banks |
| Qwen3 causal-LM/AdamW | covered `ACCEPT` plus stale-counter `REJECT` control | instantiated | strict greedy compatibility `REJECT` on prompt-disjoint finite bank | `UNINSTANTIATED` | not run for this subject | deterministic finite banks |
| Qwen3 GRPO clipping | tracked identity and self gates across two new trajectories | log-prob shift, state/checkpoint heterogeneity, zero observed repeat variability | valid strict finite-bank clipping compatibility `REJECT`; grad-enabled bank has two stable one-direction events | `UNINSTANTIATED` | one preselected grad-context branch repair has valid context-specific update effect; no operator root cause, injection, interaction or held-out claim | 20 deterministic rollout clusters for event bank; one event-conditioned repair witness; no prevalence claim |

This table is the authoritative answer to “is the Oracle complete?”: the decision
procedure is defined, several ledgers are validated, and the real-model numerical,
operator-causal and population-wide instantiations remain incomplete.

## 10. Completion and kill rules

A scoped subject may be called complete only after its required level set is named
in advance and every required ledger has non-missing authoritative evidence.

Kill or narrow the claim when:

- a pass depends on a tolerance estimated from candidate outputs;
- state distribution, randomness protocol or endpoint changes after scoring;
- a semantic event is transported from a measurement execution context to a
  transition context without reproducing its value and decision there;
- row-level resampling ignores checkpoint/prompt/trajectory clustering;
- semantic ranking adds no information beyond a predeclared raw-delta baseline;
- repair/injection effects disappear under intervention-integrity controls;
- the result requires calling eager truth without an independent specification;
- a finite stress-bank frequency is relabelled as deployment prevalence;
- long-run trajectory divergence is used as the local compiler-effect estimator.
