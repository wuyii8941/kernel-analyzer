# Qwen3 Training-Step Oracle Findings — 2026-07-17

## Outcome

The second Training-Step Oracle subject passed its predeclared mechanics gates
on the frozen 8-state bank:

- one stable 2,510-node compiled graph;
- candidate execution identity valid for every scored arm;
- 8/8 exact-core `ACCEPT`;
- 8/8 states stable across two repeats;
- no observed within-state runtime variability in the reported signatures;
- numerical transition remains `UNINSTANTIATED`;
- semantic impact remains `NOT_INSTANTIATED`.

This is evidence that the Oracle mechanics transfer from BERT/SGD to a modern
decoder-only Qwen3/AdamW transition. It is not evidence that all numerical
differences are legal, that eager is ground truth, or that the result
generalizes to a target population of DL training states.

## Frozen subject

- model: local Qwen3-0.6B checkpoint;
- task: teacher-forced causal-LM loss on retained response tokens;
- implementation pair: eager core versus full-graph Inductor core;
- precision: float32 weights with CUDA float16 autocast;
- optimizer: new non-fused/non-foreach AdamW state in both arms;
- state bank: 8 response sequences nested within 2 rollout-batch prompt
  clusters;
- repeats: 2 per state;
- target GPU: Tesla T4.

The model checkpoint comes from prior GRPO training, but the optimizer state is
new. These results are not historical GRPO optimizer replay.

## Candidate identity

```text
backend compiles: 1
compiled graph nodes: 2510
graph SHA-256:
2af7d8483aae80b619917e7a052f5e58e13a229be5352d648d09af4ddbb26fc2
runtime invocations: 17 = 1 warmup + 16 scored arms
```

The same graph identity appeared in correct-control, stale-counter and full-bank
runs. No CPU fallback or graph substitution was observed.

## Exact transition and negative control

The correct one-state smoke produced exact `ACCEPT` in both repeats. The
predeclared stale-counter control preserved the same graph and all reported
floating and prediction measurements but produced exact `REJECT` in both
repeats.

This validates a narrow but important property: the exact-state ledger detects
a transition violation that raw loss, parameter delta, optimizer-moment delta
and greedy prediction do not distinguish. It does not validate the currently
uninstantiated numerical acceptance relation.

## Discrepancy decomposition on the frozen bank

### Average relative shift

The finite-bank mean signed loss difference `(compiled - eager)` was negative,
but individual state means had both signs:

```text
negative states: 6
positive states: 2
zero states: 0
```

This is a descriptive average over the frozen bank. It is not a universal
compiler bias and not a correctness direction.

### State-conditioned heterogeneity

Loss shift, next-model L2 discrepancy and next-AdamW-moment L2 discrepancy all
varied across states. The optimizer-moment discrepancy varied much more across
states than its minimum value, and the largest model-state L2 case was not the
largest signed-loss-shift case.

Therefore this subject is not adequately described by a single scalar
`compiler error`. The implementation-relative effect depends on the token
sequence even with model weights, optimizer initialization and runtime protocol
fixed.

### Within-state runtime variability

For all three reported continuous summaries, the maximum repeat range and mean
within-state sample variance were exactly zero under two deterministic repeats.

This supports deterministic repeatability for these observed states and this
protocol. It does not prove runtime variability is absent under different GPU
scheduling, kernels, shapes or nondeterministic settings.

### Sampling uncertainty

`INDETERMINATE` for target-population inference. The 8 rows are multiple
responses nested within only 2 prompt/rollout clusters. Treating 8 rows as
independent in a t interval or row-wise bootstrap would be pseudo-replication.

The finite-bank mean and heterogeneity are exactly describable for this bank,
but uncertainty about a broader Qwen or DL-training state distribution is not
identified by this experiment.

## Semantic witness

One state had one greedy next-token disagreement, stable across both repeats:

```text
state: grpo_000000_26432b1cc437
sequence disagreement states: 1 / 8
token disagreements: 1
```

This is a discovered witness, not a predeclared impact failure. The mechanics
bank did not freeze a token-level margin, direction, endpoint acceptance rule or
independent confirmation population. The correct label is therefore
`impact = NOT_INSTANTIATED`, not `REJECT`, “bug”, or a population fork rate.

A post-discovery descriptive rerun added decision details without changing any
verdict:

```text
token position: 57
teacher-forced target token: 3745
eager top-1 token: 1372
eager top-1/top-2 margin: 0.0
compiled top-1 token: 3745
compiled top-1/top-2 margin: 0.015625
```

Both repeats produced the same record. The eager result is exactly tied at the
reported top-1/top-2 boundary under the materialized precision, so this is a
boundary-degenerate witness. The rerun was exploratory and post-selected; it
does not establish a directional population effect, and the fact that the
compiled token equals the teacher-forced target does not make compiled a
mathematical reference.

The witness does show why semantic impact cannot be inferred only from the sign
of average loss shift: the disagreement occurred in one negative-shift state,
while other states with larger-magnitude loss shifts had no greedy disagreement.

## What this changes in the Oracle theory

The BERT and Qwen results now support the following separation empirically:

1. exact transition conformance can be decided when a rule is independently
   specified;
2. average relative shift is state-bank dependent and may hide sign changes;
3. state heterogeneity can exist with zero observed runtime variability;
4. semantic events can occur while numerical correctness is uninstantiated;
5. finite-bank description and population inference require different evidence.

This is stronger than saying “bias and variance exist”. It identifies which
objects were observed and which remain unidentified.

## Confirmation status and remaining gaps

The greedy-impact confirmation requested below was subsequently instantiated on
32 non-overlapping prompt clusters and produced strict compatibility `REJECT`
with one stable event. See
`QWEN3_GREEDY_IMPACT_CONFIRMATION_FINDINGS_2026-07-17.md`. It completed an
independent prompt-level bank, decision-detail recording and a predeclared
strict-disagreement endpoint.

The following remain necessary for broader claims:

1. add multiple checkpoint clusters rather than holding weights fixed;
2. define a target population and probability sampling design before reporting
   population uncertainty or a general disagreement rate;
3. distinguish prompt-, response-, checkpoint- and within-state effects;
4. preserve `UNINSTANTIATED` numerical correctness unless an independent
   transition envelope is obtained;
5. study actual training-control events separately from greedy inference impact;
6. require realization-preserving interventions before repair/injection
   attribution is called operator causal evidence.

## Evidence

- contract: `QWEN3_TRAINING_STEP_CONTRACT_V0_1_2026-07-17.md`;
- executor: `qwen3_training_step_oracle.py`;
- correct smoke: `results/training_step_oracle/qwen3_smoke_v0_1`;
- exact negative control:
  `results/training_step_oracle/qwen3_counter_stale_v0_1`;
- full bank: `results/training_step_oracle/qwen3_bank_v0_1`;
- post-discovery decision detail:
  `results/training_step_oracle/qwen3_decision_detail_v0_1`;
- decomposition: `results/training_step_oracle/qwen3_bank_v0_1/decomposition.json`;
- decomposition script: `summarize_qwen3_training_step_bank.py`.
