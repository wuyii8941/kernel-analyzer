# Training-Step Oracle Active Plan — 2026-07-16

## Goal

Turn the validated Operator Oracle decision mechanics into a scoped DL-training-step Oracle without treating operator bits, raw parameter delta or long-run trajectory divergence as interchangeable evidence.

The authoritative definition is `TRAINING_STEP_ORACLE_V0_1_DEFINITION_2026-07-16.md`.

## 1. First subject

Use the existing controlled BERT no-momentum SGD step as the first contract subject. It is selected because the full transition is tractable, not because it produces the most forks.

Qwen teacher-forced training is now selected as the second, external subject in
`QWEN3_TRAINING_STEP_CONTRACT_V0_1_2026-07-17.md`. Because historical optimizer
moments were not preserved, its v0.1 transition starts a newly initialized
AdamW state from the frozen checkpoint. Historical GRPO/Adam replay remains
excluded until optimizer, scaler, RNG and batch state can all be reconstructed.

## 2. Required contract record before new scoring

Freeze one record containing:

```text
complete S fields and hashes
candidate execution identity
source/API authority for every exact obligation
complete Z fields
step relation T(S)
numerical geometry and independently sourced envelope, or UNINSTANTIATED
impact maps and any application margin
state-bank strata and weights
randomness/repeat protocol
verdict and missing-data rules
```

No tolerance may be selected from the confirmation outputs.

## 3. Evidence reuse

Reuse existing BERT discovery/confirmation state banks and transition measurements only for:

- state/candidate validity checks;
- relative gradient/update discrepancy;
- state heterogeneity and repeatability calibration;
- candidate endpoints for later attribution.

Do not retrospectively label those numerical rows pass/fail. Their earlier contract had no independent numerical acceptance set.

Current audit result: `BERT_TRAINING_STEP_CONTRACT_RECORD_V0_1_2026-07-16.md` classifies the old artifact as a valid matched-gradient/derived-update measurement but `UNINSTANTIATED` for complete step conformance, because no optimizer step or complete next state was materialized.

## 4. Validation sequence

### Gate A — complete-state validity

Verify every declared state field, candidate path, graph stability and exact repeat. Missing optimizer/RNG/scaler/buffer state invalidates the step comparison.

### Gate B — exact transition core

Check structure, option branches, mutation/alias behavior, counters, skip/overflow state and gradient metadata. Use confirmed option/metadata bugs and fixed/non-trigger controls.

### Gate C — numerical transition contract

Instantiate the documented SGD update relation. If gradient/rounding propagation cannot yield a defensible envelope, keep numerical acceptance `UNINSTANTIATED`; do not use observed eager/compiled spread as tolerance.

### Gate D — operator ledger

Attach covered operator contracts to real step operands. Report uncovered and fused/unidentified instances. Do not compute the step verdict by OR'ing local bits.

### Gate E — impact

Measure prediction/loss/update events separately. Test all four conformance/impact combinations rather than assuming a violation is harmful or a legal difference harmless.

### Gate F — confirmation

Freeze fresh state strata and controls, then score verdict, refusal behavior, coverage and baselines. Only after this gate may the scoped BERT step contract be called validated.

## 5. Required outputs

The first complete study must produce:

1. a BERT training-state contract record;
2. per-state structured result cards;
3. operator coverage and realization-level ledger;
4. exact/numerical/impact verdict summary;
5. discrepancy decomposition over reference, candidate and external/stress state strata;
6. comparison with raw loss, gradient norm, parameter delta and default allclose;
7. invalid/uninstantiated/indeterminate accounting;
8. an explicit statement of what is not covered.

## 6. Stop conditions

- If complete state cannot be reconstructed, stop at relative transition measurement.
- If no independent numerical envelope exists, validate only the exact core and refusal behavior.
- If operator realization is fused/unidentified, report region/step evidence rather than operator causality.
- If the result only reproduces raw parameter-delta ranking, narrow the claimed incremental value.
- Do not start free-running long-horizon validation until the one-step contract and state-population claim are fixed.

## 7. Current execution status

- normative Training-Step definition: complete;
- old BERT evidence reclassification: complete;
- materialized deterministic BERT/SGD executor: `materialized_training_step_oracle.py` added;
- CUDA smoke: passed on two states/two repeats; actual optimizer transition, state reset, candidate identity and exact-core output schema verified;
- numerical transition: deliberately `UNINSTANTIATED` because no independent update envelope exists;
- full mechanics validation contract: frozen in `BERT_MATERIALIZED_STEP_VALIDATION_CONTRACT_V0_1_2026-07-16.md`;
- full discovery/confirmation mechanics validation: passed on 128+128 states; see `BERT_MATERIALIZED_STEP_VALIDATION_FINDINGS_2026-07-16.md`;
- existing exact positive/fixed component controls mapped in `TRAINING_STEP_EXACT_CONTROL_MAPPING_V0_1_2026-07-16.md`;
- operator coverage/identity ledger complete in `BERT_TRAINING_STEP_OPERATOR_COVERAGE_LEDGER_V0_1_2026-07-16.md`; it correctly finds R3 whole-region evidence but no constituent R4 verdicts;
- full-step exact counter control frozen in `BERT_FULL_STEP_EXACT_COUNTER_CONTROL_CONTRACT_V0_1_2026-07-16.md`;
- full-step exact positive/fixed arms: passed; stale counter rejected while identical parameter/loss/prediction measurements remained unchanged;
- strict prediction-impact contract frozen on untouched SST-2 rows `[256,384)` in `BERT_TRAINING_STEP_IMPACT_CONTRACT_V0_1_2026-07-16.md`;
- impact bank extracted and executed: 128/128 valid states preserve prediction identity; covered strict compatibility `ACCEPT`;
- first scoped BERT phase complete: exact core, materialized step, refusal behavior, full-step exact positive, operator identity ledger and prediction impact are all exercised;
- numerical transition correctness remains abstained unless an independent envelope is obtained; broader optimizers/AMP/stochastic training are new contract strata.

## 8. Modern decoder-only extension — 2026-07-17

- Qwen3-0.6B selected for mechanism coverage rather than newest-version status;
- causal-LM/AdamW contract frozen before scoring in
  `QWEN3_TRAINING_STEP_CONTRACT_V0_1_2026-07-17.md`;
- executor added as `qwen3_training_step_oracle.py`;
- checkpoint and 8-row sample-bank hashes frozen;
- syntax, discrepancy helpers, AdamW state materialization and strict CUDA
  refusal behavior verified;
- initial direct-sandbox scoring was unavailable because that namespace exposes
  no `/dev/nvidia*` devices and its `nvidia-smi` cannot communicate with the
  driver;
- host user-systemd CUDA execution subsequently confirmed 14 available T4s;
- correct smoke passed: one state, two stable repeats, exact `ACCEPT` and valid
  compiled identity;
- stale-counter negative control passed: identical numerical/prediction
  measurements with exact `REJECT`;
- 8-state bank completed with 8 exact accepts and stable repeats;
- finite-bank shift, state heterogeneity and runtime repeatability decomposed;
- one stable greedy-token disagreement discovered, but impact remains
  `NOT_INSTANTIATED` pending an independent confirmation contract;
- post-discovery characterization localized it to a zero-margin eager tie at
  token position 57; this supports boundary conditioning but is not confirmation;
- target-population sampling uncertainty is `INDETERMINATE` because 8 responses
  are nested within only 2 prompt clusters;
- see `QWEN3_TRAINING_STEP_FINDINGS_2026-07-17.md`.

### Independent greedy-impact confirmation

- confirmation contract frozen before bank extraction in
  `QWEN3_GREEDY_IMPACT_CONFIRMATION_CONTRACT_V0_1_2026-07-17.md`;
- selected 32 prompt-disjoint states, one response per prompt/rollout cluster;
- all 32 candidate identities valid, exact core accepted and repeats stable;
- one stable token transition (`19 -> 422`) confirmed near the argmax boundary;
- strict finite-bank greedy compatibility: `REJECT`;
- numerical correctness remains `UNINSTANTIATED` and compiler correctness has
  no verdict;
- no population disagreement rate is claimed from deterministic file-order
  selection;
- see `QWEN3_GREEDY_IMPACT_CONFIRMATION_FINDINGS_2026-07-17.md`.

## 9. Completeness correction and next gates — 2026-07-17

The phrase “complete Oracle” is now governed by
`TRAINING_STEP_ORACLE_COMPLETENESS_STANDARD_V0_2_2026-07-17.md`.
Decision-procedure completeness and complete instantiation for one subject are
different claims.

Current evidence reaches exact, discrepancy and finite-bank impact layers on BERT
and Qwen. It does **not** yet provide all of the following on one real-model subject:

- an independent numerical transition envelope;
- a stochastic training-law contract;
- population transport and natural-optimizer confirmation of the now-valid
  grad-enabled training-control impact;
- realization-preserving operator repair/injection;
- a probability-sampled multi-checkpoint population claim.

Retained Qwen3 GRPO clipping/update evidence is mapped, without retrospective
upgrading, in
`QWEN3_GRPO_TRAINING_CONTROL_EVIDENCE_MAPPING_2026-07-17.md`. It supplies a strong
worked training-impact witness but not a compiler-correctness or operator-root-cause
verdict.

The next validation order is:

1. instantiate numerical correctness on a deliberately tractable complete training
   step using an independent high-precision or analytic transition relation;
2. freeze a new Qwen training-control confirmation population and endpoint before
   observing signed crossings;
3. compare boundary-conditioned impact ranking against raw numerical-delta ranking;
4. attempt operator attribution only with a treatment that preserves original
   monolithic realization identity; otherwise retain intervention-dependent labels;
5. add multi-checkpoint probability/cluster sampling before any population-rate
   claim.

Adding a third model without filling one of these gates is not a priority.

The third item now has retrospective finite-bank construct evidence: on the full
v0.4 bank, the two stable events rank 7/49 by raw absolute delta and 1/2 after
sign-specific boundary conditioning. This proves that paired semantic crossing
uses boundary and direction information absent from raw magnitude. Because the
diagnostic also uses the observed candidate delta, its AP is algebraic construct
evidence, not a predictive result. Held-out ranking is required only for a future
prospective risk score that does not consume held-out candidate outcomes.

### Numerical-ledger progress

The first item is now complete for a scoped analytic linear SGD transition. Its
exact rational reference and predeclared IEEE-754 error envelope accepted normal and
reverse-order compiled reductions on 32/32 states, while rejecting a compiled
drop-last negative control on every state where the omitted product was nonzero.
See `ANALYTIC_LINEAR_STEP_CONTRACT_V0_1_2026-07-17.md` and
`ANALYTIC_LINEAR_STEP_FINDINGS_2026-07-17.md`.

This validates the numerical decision mechanics but does not instantiate the Qwen
or BERT numerical transition. The grad-enabled v0.4 event bank and v0.9 controlled
branch repair now complete a single-state positive training-control path. The next
active gates are probability/cluster population transport, natural optimizer/scaler
transition, and then a real-model numerical sub-transition.

The current requirement-level audit is
`TRAINING_STEP_ORACLE_COMPLETION_AUDIT_2026-07-17.md`.

### Executable decision-core progress

The normative aggregation rules are now implemented once in
`src/forkcert/training_step_decision.py`, rather than being left to per-experiment
scripts. It validates predeclared query/evidence separation, missing-contract
refusal, endpoint registration, bias/heterogeneity/runtime/sampling terminology,
binary/categorical/set/law endpoints, explicit variability-source classification,
operator coverage/conformance and attribution integrity. Forty-three counterexample
tests, seven evidence integration records and one frozen branch-repair query audit
pass. See
`TRAINING_STEP_ORACLE_EXECUTABLE_CORE_V0_3_2026-07-17.md`.

Unified result cards now show the intended separation:

- analytic-linear correct: correctness `ACCEPT`, declared subject `COMPLETE`;
- analytic-linear drop-last: correctness `REJECT`, declared subject `COMPLETE`;
- Qwen GRPO v0.2: clipping impact `REJECT`, correctness `UNINSTANTIATED`,
  population inference unlicensed and subject `INCOMPLETE`;
- Qwen greedy: categorical impact `REJECT` without a fabricated scalar direction;
- Qwen sampling: deterministic model execution, algorithmic draw randomness,
  implementation-law discrepancy and finite-state uncertainty remain separate.
- BERT layer-0 attribution: total, repair, injection and interaction effects are
  complete for the segmented intervention, while monolithic anchor failure blocks
  operator-causal promotion.
- BERT operator coverage: the stable R3 graph region does not establish any of the
  nine R4 semantic-operator families; the required ledger remains `PARTIAL` rather
  than inventing local verdicts.

### Qwen GRPO training-control progress

- v0.1 all-state confirmation was correctly `INVALID` because the last two shape
  specializations per trajectory did not invoke tracked Inductor after Dynamo's
  default recompile limit;
- v0.2 used new prompt-disjoint strata and fixed candidate identity without changing
  the endpoint;
- 20/20 rollout states and 10,240 rows passed identity/self/finite/token gates;
- four clipping disagreements were confirmed among 4,608 applicable tokens;
- directions balanced 2 versus 2, so disagreement is nonzero while directional
  shift is zero;
- strict finite-bank training compatibility is `REJECT`; correctness and population
  prevalence remain unclaimed;
- the selected-event state reconstructed exactly;
- the first one-step branch-repair attempt is invalid because its compiled scorer
  failed endpoint anchor parity;
- the unexecuted v0.2 correction was withdrawn during static audit: evaluating
  `clamp(compiled_ratio)` inside the clipping interval did not reproduce the
  reference path's flat clipped branch;
- v0.3 freezes the branch **functional form**, exact Trainer endpoint anchors,
  full-batch B/C score hashes, compiled graph identity and an independent result
  audit;
- the old no-grad v0.3 witness remains correctly invalid;
- the replacement grad-enabled v0.4 bank contains two stable clipping events;
- the preselected step-14 event was reconstructed with exact scorer hashes;
- cold compilation failed to reproduce the candidate, proving shape-specialization
  history is an outcome-relevant matched-state component;
- v0.9 reconstructs the Accelerate wrapper and static-to-dynamic compiler history,
  completes A/B/C under per-arm resource isolation, and passes independent
  safetensors distance audit;
- the unified v0.9 record is complete for its declared clipping-impact,
  controlled-update and branch-functional attribution ledgers, while correctness,
  natural optimizer effect, population prevalence and operator cause remain open.

See `QWEN3_GRPO_TRAINING_CONTROL_CONFIRMATION_FINDINGS_2026-07-17.md`.

### External-completeness gate order (2026-07-18)

The next empirical work is governed by
`ORACLE_EXTERNAL_COMPLETENESS_GATES_V0_1_2026-07-18.md`. Old step-5/11/14
snapshots retained around known effects are stress states rather than probability
samples; they cannot be relabelled as external validation. Gate P uses
independently selected held-out states. Gate T separately restores the full natural
optimizer/scaler/RNG transition. Attribution is attempted only for an endpoint
that survives the relevant transport and transition gates. Correctness remains a
separate authority requirement.

The fixed-start Gate-P transport bank is now frozen before execution. Its B run
also captures the independently selected step-29 full pre-minibatch state and the
ten prior scorer input packs needed to reconstruct compiler specialization
history. The subsequent Gate-T contract explicitly compares an eager scorer
transition with a compiled-scorer-forward transition while holding GRPO loss,
backward, AMP, clipping, optimizer and scheduler fixed. It must not be described
as compilation of the entire training program.
