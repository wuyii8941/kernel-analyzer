# BERT Full-Step Exact Counter Control Findings — 2026-07-16

Contract: `BERT_FULL_STEP_EXACT_COUNTER_CONTROL_CONTRACT_V0_1_2026-07-16.md`.

## Result

Both arms began with `step_counter=7`, executed the same compiled BERT forward/backward graph and materialized SGD step from matched state.

| Arm | Eager next counter | Candidate next counter | Candidate identity | Exact verdict | Numerical verdict |
|---|---:|---:|---|---|---|
| correct negative | 8 | 8 | valid | `ACCEPT` | `UNINSTANTIATED` |
| stale positive | 8 | 7 | valid | `REJECT` | `UNINSTANTIATED` |

Each arm reproduced exactly across two repeats. Both used the same 134-node graph hash and had the same parameter discrepancy:

```text
next-state L2 discrepancy: 7.796134e-09
max parameter delta:        3.725290e-09
prediction disagreement:    false
```

## Meaning

The exact result changes while raw parameter/loss/prediction measurements remain unchanged. The Oracle therefore detects a missing state transition that a numerical endpoint threshold cannot see.

The mutation validates full-step exact-state mechanics. It is a synthetic independently labeled positive, not evidence that Inductor naturally corrupts a training counter.

The numerical verdict remains uninstantiated in both arms because the exact counter witness neither supplies nor requires a floating update envelope.
