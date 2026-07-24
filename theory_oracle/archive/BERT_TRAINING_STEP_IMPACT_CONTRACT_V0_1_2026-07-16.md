# BERT Training-Step Impact Contract v0.1 — 2026-07-16

> Frozen before extracting or executing the new impact bank.

## 1. State population

Use SST-2 validation source rows `[256,384)` as a new external state stratum:

```text
Q_impact = 128 consecutive validation examples not used by the earlier
           discovery [0,128) or confirmation [128,256) banks
```

The bank is not selected by eager/compiled discrepancy or prediction outcome.

## 2. Subject

Use the same materialized BERT/SGD subject and execution-validity rules as `BERT_MATERIALIZED_STEP_VALIDATION_CONTRACT_V0_1_2026-07-16.md`:

- eval-mode BERT-tiny SST-2;
- float32 parameters, CUDA float16 autocast;
- tracked full-graph compiled candidate;
- actual no-momentum SGD step, learning rate `1e-5`;
- two exact repeats per state.

## 3. Impact endpoint

The endpoint is the exposed classifier prediction:

```text
E_i(S) = argmax(logits_i(S), dim=-1)
```

The contract is strict paired reproducibility on the covered bank:

```text
impact_fail(S) = 1 when E_eager(S) != E_compiled(S)
impact_fail(S) = 0 otherwise
```

Bank verdict:

```text
INVALID  if any state/candidate identity or repeat gate fails
REJECT   if any valid covered state has impact_fail=1
ACCEPT   if all 128 valid covered states have impact_fail=0 in both repeats
```

This is an S4 compatibility/impact contract, not a mathematical correctness claim and not a population-general accuracy guarantee.

## 4. Non-endpoints

- Loss delta is descriptive; no loss margin is declared.
- Parameter/update discrepancy is descriptive; no application cost/margin is declared.
- Numerical transition conformance remains `UNINSTANTIATED`.
- Agreement with the gold SST-2 label is task accuracy, not eager/compiled impact, and is not used in this verdict.

## 5. Confirmation rule

The contract fields, source range and zero-disagreement boundary cannot change after outputs are observed. Report every invalid/missing/nonfinite row and preserve discovery/confirmation/impact strata separately.
