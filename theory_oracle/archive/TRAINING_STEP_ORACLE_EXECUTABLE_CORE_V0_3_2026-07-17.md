# Training-Step Oracle Executable Decision Core v0.3 — 2026-07-17

## Outcome

The normative Training-Step Oracle now has one shared fail-closed decision core:

```text
src/forkcert/training_step_decision.py
```

It consumes a predeclared query contract and a separate evidence record, then emits
one structured result. It does not run a model, choose tolerances or certify that a
claimed authority is scientifically sound. Those remain responsibilities of the
subject contract and measurement executor.

The core's job is narrower and necessary: the same evidence state must receive the
same ledger meaning across subjects, and missing evidence must never become an
implicit pass.

## Input separation

The query declares, before evidence interpretation:

- complete state, reference/candidate, observable, population and randomness
  contracts;
- population kind, stratum provenance, trajectory/data anchor, selection design
  enrichment status and aggregation rule;
- required, optional and out-of-scope ledgers;
- independent authorities for exact, numerical and stochastic correctness;
- predeclared impact endpoints and boundary geometry;
- discrepancy estimands and the actual sampling unit;
- semantic-operator coverage rule when local conformance is in scope;
- attribution intervention and requested claim level;
- covered scope and explicit non-claims.

The evidence record separately supplies validity gates, candidate identity,
measurements, tests and intervention integrity. Evidence for an unregistered impact
endpoint invalidates the query instead of silently adding a post-hoc endpoint.

A transition or terminal operator authority is a structured audit record, not a
nonempty citation string. It names the authority kind, source, covered scope,
independence from candidate measurements and whether its acceptance rule was
frozen. Candidate-calibrated tolerances therefore remain `UNINSTANTIATED`. The core
checks these declarations structurally; it does not certify that the cited science
is true.

## Derived output

The output preserves these ledgers independently:

```text
validity and candidate identity
exact / numerical / stochastic transition
one record per impact endpoint
discrepancy structure
sampling uncertainty
variability-source classification
operator conformance and coverage
attribution and eligible claim level
correctness claim
subject-instantiation completeness
```

Each result also embeds the frozen state, implementation, observable, population,
randomness and claim-scope contracts under `declared_context`. A result card is
therefore not interpretable as a free-floating bias/variance number after being
separated from its query file.

Correctness is aggregated only from required transition-conformance ledgers.
Consequently:

- exact `REJECT` survives numerical `UNINSTANTIATED`;
- impact `REJECT` does not become correctness `REJECT`;
- nonzero discrepancy does not create a correctness verdict;
- missing numerical authority produces `UNINSTANTIATED`, not `ACCEPT`;
- invalid state/candidate evidence produces `INVALID`;
- local operator `ACCEPT/REJECT` records remain diagnostic and are not OR'ed into
  whole-step correctness;
- a terminal operator verdict requires semantic-operator identity, R4
  correspondence, an independent operator contract and evidence; region/kernel
  records cannot be relabelled as operators;
- incomplete operator-causal integrity downgrades attribution to
  `INTERVENTION_DEPENDENT`.

`subject_instantiation=COMPLETE` means every ledger marked `REQUIRED` has a valid
terminal record for that declared subject. It does not mean universal DL-training
coverage.

## Semantic and discrepancy guards

For binary endpoints, terminal evidence must include directional shift,
disagreement and natural-boundary exposure when applicable. If counts are supplied,
the core verifies that probabilities, both direction counts and the denominator are
consistent.

An estimated discrepancy record must separately name:

- `average_implementation_relative_shift`;
- `state_conditioned_heterogeneity`;
- `within_state_runtime_variability`.

Naming alone is not sufficient. Every predeclared discrepancy estimand must also
fix its observable, candidate/reference comparison, aggregation unit and geometry.
Its evidence must contain an estimate plus a scope-or-uncertainty record. Missing,
post-hoc or unregistered discrepancy components invalidate the ledger. This
prevents a scalar with unspecified direction or weighting from being relabelled as
"bias", and prevents cross-state spread from being relabelled as runtime noise.

Top-level unqualified `bias`, `variance`, `compiler_bias`,
`floating_point_variance` and `total_error` keys are rejected. Sampling uncertainty
is a separate ledger and its sampling unit must match the population contract.

The variability-source ledger separately classifies state sampling, batch/token
sampling, algorithmic RNG, execution nondeterminism, autotuning selection,
stochastic rounding and fixed implementation differences. In particular, a fixed
implementation difference cannot be submitted as within-state randomness, and
state sampling cannot be relabelled as runtime variability.

## Validation cases

`tests/test_training_step_decision.py` contains 43 mechanics/counterexample tests:

- complete correctness acceptance;
- exact rejection with numerical refusal;
- impact/correctness separation;
- missing state, ledger, scope, authority and evidence refusal;
- candidate-calibrated or otherwise non-independent correctness authority refusal;
- optional ledgers may be absent, but supplied invalid optional evidence prevents a
  `COMPLETE` result card;
- unregistered endpoint rejection;
- binary direction/count/boundary consistency;
- bias/variance terminology guard;
- operational definition, evidence scope and post-hoc registration guards for
  every discrepancy estimand;
- sampling-unit consistency;
- population-stratum provenance/anchor and aggregation-rule requirements, so
  reference, candidate, external and stress state banks cannot be silently pooled;
- trajectory/data origin and event/boundary enrichment are orthogonal fields, so a
  reference-anchored event witness cannot be reported as a natural reference-state
  prevalence sample;
- exhaustive predeclared variability-source coverage and classification guards;
- an `UNKNOWN` source remains `INDETERMINATE` and prevents subject completion;
- stochastic single-draw refusal;
- categorical and set-valued endpoint rules without fabricated scalar direction;
- continuous endpoints require a signed effect and cost, while vector endpoints
  require an explicit geometry and cost without inventing a scalar direction;
- semantic-operator versus region separation, R4 identity, coverage accounting and
  non-aggregation into whole-step correctness;
- full coverage with an indeterminate local contract remains `INDETERMINATE`, not
  falsely `RECORDED/COMPLETE`;
- operator attribution downgrade and full-gate eligibility;
- unknown attribution claim levels are invalid rather than silently weakened;
- failed monolithic anchor parity retains only an explicitly scoped
  intervention-dependent claim, while unidentified intervention treatment is
  invalid;
- attribution effects require predeclared endpoints and complete total, repair,
  injection and interaction contrasts.

`tests/test_training_step_decision_records.py` adds seven evidence-backed integration
records plus one frozen pre-execution Qwen branch-repair query audit. All 51 tests
pass under the project CUDA Python environment using standard
library `unittest`.

## First unified records

### Analytic linear correct candidate

The exact and independently bounded numerical ledgers both `ACCEPT`; correctness is
`ACCEPT` and the declared exact/numerical subject is `COMPLETE`.

```text
results/training_step_oracle/analytic_linear_correct_v0_1/
  unified_oracle_result_v0_3.json
```

### Analytic linear drop-last negative control

Exact structure `ACCEPT`s while numerical conformance `REJECT`s on 31 witness
states. Correctness is `REJECT`; the one silent state where the omitted product is
zero is retained.

```text
results/training_step_oracle/analytic_linear_drop_last_v0_1/
  unified_oracle_result_v0_3.json
```

### Qwen3 GRPO training-control confirmation

Candidate/state validity is `VALID`, clipping impact is `REJECT`, directional shift
is zero, discrepancy is estimated, numerical correctness is `UNINSTANTIATED`, and
population inference is unlicensed. The subject is therefore `INCOMPLETE` for its
declared numerical/population requirements.

```text
results/training_step_oracle/qwen3_grpo_training_control_confirmation_v0_2/
  unified_oracle_result_v0_3.json
```

This record is a retrospective structural mapping of already predeclared evidence.
It does not introduce a new threshold or confirmation claim.

### Qwen3 greedy categorical compatibility

The exact transition core `ACCEPT`s, strict greedy compatibility `REJECT`s on one
stable token transition, numerical correctness remains `UNINSTANTIATED`, and no
scalar directional shift is invented for token categories.

```text
results/training_step_oracle/qwen3_impact_confirmation_v0_1/
  unified_oracle_result_v0_3.json
```

### Qwen3 categorical sampling law

Deterministic model execution, nonzero implementation-relative TV/JS, much larger
algorithmic draw variability and state-level uncertainty are all reported
separately. With no independent acceptable-law region, stochastic/numerical
correctness and sampling compatibility remain `UNINSTANTIATED`.

```text
results/oracle_sampling/confirmation/unified_oracle_result_v0_3.json
```

### BERT layer-0 segmented attribution

The four-arm total, repair-residual, injection and interaction effects are complete
for the declared segmented-program attribution subject. The candidate segmented
endpoint fails monolithic anchor parity, so the core retains
`INTERVENTION_DEPENDENT`, refuses unique necessity/sufficiency and root-cause
claims, and leaves correctness `UNINSTANTIATED`.

```text
results/oracle_region/layer0_confirmation/unified_oracle_result_v0_3.json
```

### BERT operator-family coverage audit

Nine semantic operator families are inventoried, but every family has only R1
source presence and zero families have R4 in-program correspondence. The operator
ledger is therefore valid `PARTIAL`, all local verdicts remain `UNINSTANTIATED`,
and the subject is `INCOMPLETE`. The stable 134-node R3 region is not promoted to
an operator.

```text
results/training_step_oracle/bert_operator_coverage_v0_1/
  unified_oracle_result_v0_3.json
```

## Remaining scientific gates

The executable core makes refusal and aggregation consistent; it cannot fill
missing scientific authorities. The current real-model gaps remain:

- an independent Qwen/BERT numerical transition contract;
- a probability/cluster design supporting a target-population claim;
- population confirmation of the new Qwen grad-enabled event bank and its
  single-state branch-repair effect; the v0.9 witness validates a controlled
  intervention but not prevalence or natural-optimizer impact;
- realization-preserving injection, interaction analysis and held-out replication
  before any operator-causal claim;
- a stochastic full-training-step law and any justified local-to-long-run bridge.

The frozen Qwen branch-functional query at
`QWEN3_GRPO_BRANCH_REPAIR_UNIFIED_QUERY_V0_3.json` was executed and correctly
refused.  Its raw reloader omitted Accelerate's output-FP32 wrapper; after that
context was diagnosed and restored, the selected `no_grad` compiled event did not
survive the grad-enabled specialization.  The unified result is `INVALID`, and no
vector-update or attribution claim is emitted.  This negative case validates the
core's endpoint-realization gate rather than completing the Qwen impact ledger.

The replacement grad-enabled v0.4 bank and its preselected v0.9 branch repair now
provide a valid positive path.  The v0.9 unified result is complete for the
declared single-state impact/branch-functional ledgers and deliberately leaves
correctness uninstantiated.  Its failed predecessors additionally established
that tensor geometry, Accelerate wrapping, compiler specialization history and
resource lifetime are validity conditions, not interchangeable noise terms.
