# BERT Full-Step Exact Counter Control Contract v0.1 — 2026-07-16

> Frozen before executing the stale-counter mutation arm.

## Subject

Use one frozen BERT discovery state and the materialized deterministic SGD executor. Both eager and candidate begin with identical external `step_counter=7`. The declared transition relation requires `next_step_counter=8`.

## Arms

```text
negative control: candidate-counter-mode=correct
positive control: candidate-counter-mode=stale
```

The stale mode is an independently labeled synthetic implementation mutation: it changes only the candidate's next counter after the same compiled forward/backward and SGD step. Initial state, inputs, optimizer options and compiled graph remain matched.

## Verdict

```text
INVALID  if candidate compiled identity or state reset fails
ACCEPT   exact core when both next counters equal 8 and all other exact fields pass
REJECT   exact core when candidate next counter is not 8
```

Numerical transition conformance remains `UNINSTANTIATED` in both arms. Parameter discrepancy magnitude cannot alter the exact counter verdict.

## Expected role

This validates a full-step exact-state positive/fixed pair. It is not a natural compiler bug, prevalence estimate or operator attribution result.
