# Qwen3 Greedy-Impact Confirmation Findings — 2026-07-17

## Verdict

```text
strict greedy compatibility: REJECT
compiler correctness: NO CLAIM
numerical transition: UNINSTANTIATED
population inference: NOT CLAIMED
```

The predeclared strict compatibility rule rejected because one independently
selected prompt state produced one stable eager/compiled greedy-token
disagreement. Every state passed the candidate-identity and exact-transition
gates, and all repeats were stable.

## Confirmation bank

- source: `data/phase0_grpo_samples.jsonl`;
- discovery prompts excluded exactly by `prompt_ids`;
- selection: first eligible response for each unique prompt in source order;
- states: 32;
- unique prompt tuples: 32;
- unique rollout-batch clusters: 32;
- repeats per state: 2;
- bank SHA-256:
  `f8c43e386dce9d55157e84633f4595f92552b6bae62e786b4ab6a8f16cbb1736`.

The bank is independent of the 8-state mechanics discovery with respect to
prompt identity. It is not a random sample from a declared Qwen or DL-training
population.

## Mechanics gates

```text
candidate identity valid: 32 / 32
exact-core ACCEPT: 32 / 32
repeat-stable states: 32 / 32
backend compiles: 1
compiled graph nodes: 2510
runtime invocations: 65 = 1 warmup + 64 scored arms
```

The compiled graph SHA matches the discovery and control runs:

```text
2af7d8483aae80b619917e7a052f5e58e13a229be5352d648d09af4ddbb26fc2
```

## Confirmed compatibility event

Both repeats produced the same record:

```text
state: grpo_000002_80fe31334599
token position: 51
teacher-forced target token: 22

eager:
  top-1 token: 19
  top-1 logit: 22.234375
  top-2 token: 422
  top-2 logit: 22.21875
  margin: 0.015625

compiled:
  top-1 token: 422
  top-1 logit: 22.25
  top-2 token: 19
  top-2 logit: 22.234375
  margin: 0.015625
```

Neither chosen token equals the teacher-forced target, so this event provides no
single-token quality direction. The transition direction is `19 -> 422`; argmax
categories do not have a meaningful scalar “up” or “down” direction.

## Finite-bank quantities

```text
sequence-disagreement states: 1 / 32
token disagreements: 1
finite-bank sequence-disagreement proportion: 0.03125
```

The last number describes this deterministic bank only. A binomial confidence
interval would incorrectly assume a random sampling design that was not used.

## Error / bias / variance interpretation

### Average relative shift

The finite-bank mean signed loss delta was close to zero relative to its
cross-state spread. Individual state means again had both signs:

```text
negative states: 19
positive states: 13
```

This does not support a universal fixed-direction “compiler bias”. It supports
a bank-relative average plus substantial state-conditioned effects.

### State heterogeneity

Loss, next-model L2 and AdamW-moment L2 discrepancies varied across states. The
confirmed disagreement state had the largest positive loss shift and largest
optimizer-moment L2 in this bank, but other large discrepancies did not change
greedy tokens. Therefore raw magnitude alone is not the semantic Oracle.

### Runtime variability

The recorded paired signatures had zero within-state repeat range under this
deterministic protocol. The compatibility event is persistent for its matched
state, not observed runtime noise.

### Sampling uncertainty

Not identified for a broader target distribution. The bank has one state per
prompt cluster, which avoids the discovery bank's response-level
pseudo-replication, but deterministic file-order selection still does not define
a probability sampling design.

## Theoretical consequence

The result realizes two separate ledger combinations:

```text
exact transition: ACCEPT
strict application compatibility: REJECT
numerical correctness: UNINSTANTIATED
```

This is precisely why whole-step conformance, numerical discrepancy and
semantic impact cannot be collapsed into one bit. It also supplies a concrete
counterexample to “global average shift is the Oracle”: a small bank-average
signed loss shift coexists with a stable boundary event, while many larger local
continuous discrepancies do not cause an event.

## What remains unproven

- No mathematical wrong-code bug is established.
- No universal Qwen or DL-training disagreement rate is estimated.
- No training-control event such as clipping, overflow or step skip changed.
- No long-run convergence or quality effect is established.
- No operator, region or kernel cause is identified.
- No claim is made for Qwen3.6, other model families, BF16 or newer GPUs.

## Evidence

- contract:
  `QWEN3_GREEDY_IMPACT_CONFIRMATION_CONTRACT_V0_1_2026-07-17.md`;
- bank manifest:
  `data/external_datasets/qwen3_impact_confirmation_32.manifest.json`;
- result directory:
  `results/training_step_oracle/qwen3_impact_confirmation_v0_1`;
- scored compatibility:
  `results/training_step_oracle/qwen3_impact_confirmation_v0_1/impact_evaluation.json`;
- discrepancy decomposition:
  `results/training_step_oracle/qwen3_impact_confirmation_v0_1/decomposition.json`.

