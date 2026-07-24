# BERT Materialized-Step Validation Findings — 2026-07-16

Contract: `BERT_MATERIALIZED_STEP_VALIDATION_CONTRACT_V0_1_2026-07-16.md`.

## 1. Bottom line

The frozen discovery and untouched confirmation banks both pass the materialized-step **mechanics** gate:

- actual SGD next states were materialized from reset matched states;
- every measured candidate call reached one stable compiled graph;
- the covered exact transition core accepted all 256 states across both banks;
- every state reproduced exactly across two repeats;
- eager/compiled next parameters differed numerically;
- numerical transition conformance correctly remained `UNINSTANTIATED`.

This validates execution, exact-core and refusal behavior for the declared deterministic BERT/SGD subject. It does not establish numerical transition correctness.

## 2. Bank results

| Field | Discovery | Confirmation |
|---|---:|---:|
| states | 128 | 128 |
| measured rows | 256 | 256 |
| backend compiles | 1 | 1 |
| graph nodes | 134 | 134 |
| graph hash | `1aed318f…cae943` | `1aed318f…cae943` |
| candidate identity valid | 128/128 states | 128/128 states |
| exact-core accept | 128/128 | 128/128 |
| exact-core reject | 0 | 0 |
| repeat-stable states | 128/128 | 128/128 |
| mean next-state L2 discrepancy | `1.787218e-07` | `2.188472e-07` |
| maximum coordinate discrepancy | `1.192093e-07` | `1.192093e-07` |
| prediction disagreement | 0 | 0 |
| numerical verdict | `UNINSTANTIATED` | `UNINSTANTIATED` |

Each bank contains the expected 256 JSONL rows. No crash, missing or nonfinite row was dropped.

## 3. Exact-core interpretation

The accepted fields are limited to the frozen exact relation:

```text
state structure/names/shapes/dtypes
gradient presence/shape/dtype
SGD option and empty-state structure
nonfloating next-state fields
eval-mode buffers
coupled next RNG state
```

The result is meaningful: an ignored option, missing gradient, mutated buffer, RNG mismatch or fallback would have rejected/invalidated the covered row instead of being hidden in a norm.

It does not say all possible exact training semantics were covered. Schedulers, GradScaler, dropout and Adam state are outside this subject.

## 4. Numerical interpretation

The candidate and reference produce different materialized parameters even when exposed loss and prediction agree. The discrepancy magnitude changes across state banks, while the maximum coordinate scale reproduces.

Neither observation supplies an accuracy boundary. Therefore:

```text
small discrepancy       does not imply ACCEPT
held-out reproduction   does not imply correctness
zero prediction changes does not imply transition equivalence
```

The numerical verdict remains undefined by design, not because the experiment failed.

## 5. Contract-gate audit

| Frozen gate | Result |
|---|---|
| candidate identity and stable graph | pass |
| complete row count | pass |
| covered exact core | pass |
| numerical refusal preserved | pass |
| exact repeats | pass |
| no post-confirmation rule change | pass |

## 6. Scope and next requirement

The first scoped materialized Training-Step Oracle mechanics are now validated for eval-mode BERT with deterministic no-momentum SGD. Before a broader or correctness claim, the project still needs:

1. exact positive/fixed step-level controls;
2. an operator coverage/identity ledger attached to real step operands;
3. an independently justified numerical gradient/update envelope, or permanent numerical abstention for this subject;
4. an application impact contract;
5. separate contracts for train-mode dropout, gradient clipping, AMP/GradScaler, Adam/AdamW and Qwen/GRPO.

Long-run training remains outside this local transition validation.
