# BERT Training-Step Contract Record v0.1 — 2026-07-16

## 1. Record purpose

This record audits the existing controlled BERT transition evidence against `TRAINING_STEP_ORACLE_V0_1_DEFINITION_2026-07-16.md`. It determines what can be reused and what is still missing before the first true training-step verdict.

Evidence source:

- `matched_transition_oracle.py`;
- `ORACLE_NEXT_STAGE_CONTRACT_2026-07-16.md`;
- `TRANSITION_ORACLE_FINDINGS_2026-07-16.md`.

## 2. Intended first subject

```text
model: BERT-tiny SST-2 classifier
input unit: one frozen tokenized example and label
mode: eval/dropout disabled
parameters: float32
forward arithmetic: CUDA float16 autocast, logits/loss promoted to float32
candidate: torch.compile/Inductor full graph with tracked backend invocation
proposed update: full-parameter SGD, lr=1e-5, no momentum, no weight decay
randomness: deterministic algorithms and fixed global seed
```

This is a controlled supervised-step subject. It is not production Qwen/GRPO training.

## 3. Existing-state coverage

| Required state field | Existing evidence | Status for a complete step |
|---|---|---|
| model parameters | both paths share one loaded model object; artifact hashes recorded | strong operand matching, but no per-state full tensor hash |
| mutable buffers | implicitly shared model object | buffer before/after values not enumerated or audited |
| batch/tokens/label | frozen dataset row, tokenization and state id recorded | covered for relative measurement |
| optimizer configuration | learning rate/no-momentum/no-weight-decay declared | no optimizer object or state instantiated |
| optimizer state | none required for ideal no-momentum SGD | structural empty-state relation not materialized/audited |
| algorithmic RNG | global seed and deterministic mode set | RNG state is not snapshotted/restored and next RNG state is not observed |
| AMP/loss scale | autocast fp16 declared | no GradScaler; this is fine only if explicitly outside subject |
| schedules/counters | absent | outside current measurement; must be explicitly empty or added |
| train/eval mode | `model.eval()` | covered, but this is not ordinary dropout-enabled training mode |
| compiler/hardware config | environment and compile audit recorded | covered for named run |

## 4. Existing observable coverage

| Required observable | Existing evidence | Status |
|---|---|---|
| loss and logits | observed and finite | covered measurement |
| gradients | full trainable-parameter gradients cloned and compared | covered measurement |
| gradient metadata | presence/support partly recorded | incomplete exact metadata contract |
| update vector | computed algebraically as `-lr * gradient` | derived, not executed |
| next parameters | only eager/candidate distance derived from gradient difference | not materialized or fieldwise checked |
| next optimizer state | absent | not observed |
| buffers/counters/RNG next state | absent | not observed |
| exception/skip/overflow | finite checks; no scaler/skip mechanism in subject | partial/excluded |

The script itself records `optimizer_step_applied: false` and calls the optimizer “hypothetical.” This field is decisive: the artifact is not an executed training-state transition.

## 5. Existing evidence verdict

### Relative gradient/update measurement

```text
validity: VALID for the named deterministic matched-gradient protocol
candidate identity: tracked full-graph compiled invocation
result: nonzero, state-conditioned eager/candidate gradient discrepancy
runtime variability: zero in the observed exact-state repeats
claim: implementation-relative controlled gradient and derived-SGD discrepancy
```

### Training-step conformance

```text
bit: undefined
verdict: UNINSTANTIATED
```

Reasons:

1. no actual optimizer transition was executed;
2. complete next state was not observed;
3. no independent numerical gradient/update envelope was supplied;
4. exact transition obligations were not frozen as a complete product relation.

The correct result is not `ACCEPT`, `REJECT` or `INVALID`. The measurement is valid for its narrower object, while the broader transition contract is uninstantiated.

## 6. Reusable evidence

Reuse without rerunning:

- frozen discovery/confirmation state identities;
- model/data artifact records;
- compiled-path and graph-stability evidence;
- full-gradient discrepancy geometry;
- state heterogeneity and exact-repeat calibration;
- prediction/loss endpoints as impact measurements;
- confirmation that raw loss delta cannot reconstruct gradient discrepancy.

Do not reuse as correctness labels:

- nonzero gradient distance;
- derived next-parameter distance;
- zero repeat variability;
- prediction agreement;
- held-out reproduction of discrepancy magnitude.

## 7. Requirements for the first executable step contract

Before a training-step verdict, the next manifest must add:

1. exact enumeration/hash of initial parameters and mutable buffers;
2. explicit empty or instantiated optimizer state and all option values;
3. actual optimizer/update execution for both paths from isolated identical copies;
4. materialized next parameters, optimizer state, buffers, counters and RNG state;
5. exact structural/option/mutation relations;
6. a numerical update envelope or an explicit `UNINSTANTIATED` numerical field;
7. a state-reset proof between eager/candidate/repeat arms;
8. separately declared impact endpoints;
9. operator coverage ledger with fused/unidentified cases preserved.

## 8. Scope decision

The first executable subject remains BERT with deterministic no-momentum SGD and no GradScaler. Dropout-enabled training, Adam/AdamW, gradient clipping, AMP scaling and Qwen/GRPO are separate later strata.

This narrow start is not a shortcut around the whole-training goal: it is the smallest state transition whose complete contract can be audited without inventing missing historical state.
