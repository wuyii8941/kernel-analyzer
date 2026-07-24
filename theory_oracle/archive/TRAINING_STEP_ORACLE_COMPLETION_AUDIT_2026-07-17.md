# Training-Step Oracle Completion Audit — 2026-07-17

## Audit conclusion

The Training-Step Oracle is **decision-procedure complete for the currently declared
ledgers**, and its exact, numerical-refusal, numerical-correctness, deterministic
discrepancy and finite-bank impact behaviors all have concrete validation evidence.

This is now executable rather than document-only: `src/forkcert/training_step_decision.py`
enforces the query/evidence/result separation and fail-closed aggregation.
The current regression suite contains 56 passing mechanics/counterexample and
evidence-backed integration tests. The grad-enabled Qwen event bank and its v0.9
controlled branch-repair query now add a positive real training-control path.

It is **not a complete universal DL-training correctness Oracle**. Real-model
numerical contracts, stochastic training laws, probability-sampled population
claims and realization-preserving operator causality remain open.

This distinction follows the normative
`TRAINING_STEP_ORACLE_COMPLETENESS_STANDARD_V0_2_2026-07-17.md`.

## Requirement-by-requirement evidence

| Requirement | Authoritative evidence | Audit status |
|---|---|---|
| complete matched-state schema | Training-Step v0.1 definition plus BERT/Qwen contract records | defined and exercised for scoped subjects |
| candidate execution identity | tracked graph hashes/invocations in BERT, Qwen and analytic linear runs | validated for covered runs |
| exact transition relation | BERT and Qwen correct arms plus independent stale-counter controls | validated |
| numerical refusal when no independent envelope exists | BERT/Qwen results retain `UNINSTANTIATED` despite nonzero deltas | validated |
| numerical accept using independent truth | analytic linear exact rational reference and a priori error envelope | validated on scoped transition |
| independent-authority structure | kind/source/scope plus candidate-independence and frozen-rule declarations are mandatory for transition/operator terminal verdicts | executable refusal validated; scientific truth of a cited authority remains externally reviewable |
| legal numerical difference not automatically rejected | reverse-reduction analytic control | validated on scoped transition |
| numerical wrong-program rejection | drop-last analytic negative control | validated on 31 applicable witness states |
| discrepancy decomposition | Qwen finite-bank average shift, state heterogeneity and repeat profile with explicit observable, comparison, aggregation unit and geometry | validated descriptively |
| runtime variability distinct from state heterogeneity | exact repeat signatures with cross-state effect variation | validated under deterministic protocols |
| sampling uncertainty distinct from runtime variability | Qwen cluster audit refuses population inference | validated refusal |
| predeclared application impact | Qwen prompt-disjoint greedy compatibility contract | validated finite-bank `REJECT` |
| actual training-control event | new prompt-disjoint Qwen GRPO v0.2 trajectories; 20 valid tracked rollout states | strict finite-bank compatibility `REJECT`; four balanced-direction events |
| actual training-control update consequence | grad-enabled v0.4 bank plus frozen v0.9 A/B/C branch repair and independent safetensors audit | valid single-state controlled-update effect; branch contribution is intervention-dependent, not operator cause or natural-optimizer impact |
| stochastic law endpoint | multinomial/sampling-law Oracle records | operator/application layer validated; full stochastic training step open |
| operator conformance/coverage ledger | semantic-operator-only R4 terminal verdicts, independent contract authority, explicit covered/uncovered counts | BERT family audit validly reports 0/9 R4-covered and remains `PARTIAL`; no region evidence is borrowed |
| operator/region attribution integrity | attribution contract plus unified BERT layer-0 four-arm result | segmented intervention attribution complete; monolithic parity failure correctly blocks operator/root-cause language |
| population inference | no operational probability-sampled multi-checkpoint `Q` | missing |
| state-population identity | every unified query declares population kind, provenance, reference/candidate/external/synthetic anchor, independent enrichment status, inclusion design and aggregation rule | executable validity gate complete; real target-population coverage remains missing |
| long-run bridge | definition explicitly separates local transition and trajectory outcome | theory boundary defined; predictive bridge missing |
| shared executable decision core | v0.3 query/evidence/result engine plus 51 tests, seven unified result cards and one frozen branch-repair query | validated for declared mechanics; it does not validate scientific authorities by itself |
| variability-source separation | explicit state/batch sampling, algorithmic RNG, execution nondeterminism, autotuning, stochastic rounding and fixed implementation classifications | validated on deterministic, categorical-impact and stochastic-sampling records |

## Current subject matrix

| Subject | Exact | Numerical | Impact | Attribution | Population claim |
|---|---|---|---|---|---|
| analytic linear SGD | `ACCEPT`; structure covered | independently `ACCEPT/REJECT` capable | not in scope | not in scope | frozen finite bank only |
| BERT SGD | `ACCEPT` plus counter `REJECT` | `UNINSTANTIATED` | finite-bank prediction `ACCEPT` | operator coverage `PARTIAL` at 0/9 R4 families; layer-0 segmented attribution `COMPLETE`; original-program cause blocked | deterministic banks |
| Qwen causal-LM/AdamW | `ACCEPT` plus counter `REJECT` | `UNINSTANTIATED` | finite-bank greedy `REJECT` | not instantiated | deterministic prompt banks |
| Qwen GRPO grad-enabled v0.4/v0.9 | tracked identity on 20 grad-enabled rollout states plus exact static-to-dynamic specialization reconstruction | `UNINSTANTIATED` | finite-bank clipping `REJECT`; one preselected clipping event has controlled-update `REJECT` | valid branch-functional repair effect at `INTERVENTION_DEPENDENT`; injection/interactions/held-out and operator source remain missing | deterministic two-checkpoint bank plus one event-conditioned witness; no prevalence claim |

No row is silently promoted to a stronger claim by borrowing another row's evidence.

## Remaining gates in priority order

### Gate 1 — population and natural-transition confirmation

The grad-enabled event and controlled-update mechanics are now complete for the
declared finite bank and one preselected witness. The next missing claim is not
another clipping example: it is transport to a probability/cluster-sampled,
multi-checkpoint state population and to the natural optimizer/scaler transition.
Stress enrichment and operational prevalence remain separate.

### Gate 2 — real-model numerical relation

Choose one tractable sub-transition with an independent authority: for example a
documented optimizer map with preserved state, a high-precision normalization or
reduction sub-transition, or a confirmed wrong-code relation. Do not extrapolate the
linear envelope to transformer operators.

### Gate 3 — operator treatment integrity

Require the hybrid treatment to preserve the original monolithic compiled context,
or explicitly define the whole context change as the treatment. Report source,
propagation and boundary-conversion roles separately. Repair disappearance alone is
insufficient.

### Gate 4 — population claim

Define operational state populations over checkpoint, prompt/batch and trajectory.
Use their actual selection units for uncertainty. Only then report prevalence or
risk rather than finite-bank proportions.

### Gate 5 — stochastic and long-run extensions

Open one randomness source at a time and compare laws. Treat long-run training as
separate validation unless stability/coupling/coverage assumptions justify a local
to global inference.

## What “finished” may defensibly mean next

The next scoped completion target should be:

> a Qwen GRPO population and natural-step validation of the already-defined
> multi-ledger Oracle, with probability/cluster-sampled checkpoint strata,
> compiler-history identity, boundary-conditioned event and update endpoints,
> explicit repeats, cluster-aware uncertainty, and correctness abstention where
> no numerical specification exists.

It should not be “a universal compiler bug Oracle.” That claim would remain false
even if the next experiment finds more clipping events.
