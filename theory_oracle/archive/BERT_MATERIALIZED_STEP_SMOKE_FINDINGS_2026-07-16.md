# BERT Materialized Training-Step Smoke Findings — 2026-07-16

## 1. Purpose

Validate the first executable gate of `TRAINING_STEP_ORACLE_ACTIVE_PLAN_2026-07-16.md`: actually materialize eager and compiled SGD next states from identical BERT states while preserving refusal semantics for the uninstantiated numerical envelope.

Artifact:

```text
results/training_step_oracle/bert_smoke_v0_1/
```

The smoke used two frozen SST-2 states, two exact repeats per state, float32 parameters, CUDA float16 autocast and SGD with learning rate `1e-5`, no momentum and no weight decay.

## 2. Execution validity

```text
backend compiles:             1
unique graph hash:            1
compiled runtime invocations: 5 (one warmup + four measured calls)
candidate identity valid:     4/4 measured rows
```

Both arms restored the same baseline model state and coupled CPU/CUDA RNG snapshot before each measured step. No CPU fallback was admitted.

## 3. Exact transition core

Both states passed the covered exact core in both repeats:

```text
state structure:              equal
gradient presence/shape/dtype: equal
optimizer structure/options:  equal
nonfloating next-state fields: equal
mutable buffers:              exact equal
next RNG state:               exact equal
exact verdict:                ACCEPT on 2/2 states
```

This is a scoped exact-core acceptance, not a numerical whole-step acceptance.

## 4. Materialized numerical transition

The optimizer step was actually executed for each arm. The next parameter states differed:

| State | Next-state L2 difference | Max coordinate difference | Repeat stability |
|---|---:|---:|---|
| `sst2-000000` | `7.7961e-09` | `3.7253e-09` | exact across two repeats |
| `sst2-000001` | `1.3270e-07` | `2.9802e-08` | exact across two repeats |

Aggregate:

```text
mean next-state L2 difference: 7.02498e-08
maximum coordinate difference: 2.98023e-08
```

No independent gradient/update error envelope was supplied. Therefore:

```text
numerical transition verdict: UNINSTANTIATED
```

The nonzero values are implementation-relative measurements, not bugs and not passes.

## 5. Impact

Loss was exactly equal at the exposed float32 scalar and predictions agreed on both states. No application margin was instantiated, so the structured result records impact as `NOT_INSTANTIATED` rather than “no impact proved.”

## 6. What changed relative to old evidence

The earlier `matched_transition_oracle.py` derived `-lr * gradient` and explicitly recorded `optimizer_step_applied: false`. The new executor materializes the optimizer step, next parameters, optimizer structure, buffers and RNG state.

This closes the execution/materialization gap for a deterministic no-momentum SGD smoke. It does not close the numerical contract gap.

## 7. Remaining gates

- freeze a full discovery/confirmation manifest rather than reuse smoke states for scoring;
- add an independently justified numerical update envelope or retain `UNINSTANTIATED`;
- instantiate impact margins;
- add the operator conformance/coverage ledger;
- validate exact positive/fixed controls at the step level;
- expand beyond eval-mode BERT/no-momentum SGD only after this scoped contract passes.
