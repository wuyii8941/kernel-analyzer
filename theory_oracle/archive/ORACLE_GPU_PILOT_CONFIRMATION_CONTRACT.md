# Oracle GPU Pilot — Held-out Confirmation Contract

Frozen before the confirmation run on 2026-07-15.

## Purpose

Test whether the structural conclusions of `deterministic_20260715_v1` reproduce on independently generated matched states. This is confirmation of the measurement structure on the same controlled subject, not external generalization.

## Frozen protocol

- subject and tensor shapes: unchanged from `oracle_gpu_pilot.py`;
- state seed: `20260716`, not used in calibration;
- state population: 64 `near`, 64 `far`, and 64 `natural` states, 16 cases per state;
- repeats: 3 under deterministic CUDA execution;
- implementation pair: eager versus tracked, full-graph Inductor;
- self-pairs: eager--eager and compiled--compiled;
- bootstrap unit: state, 5,000 resamples;
- GPU/cache/output: isolated from calibration;
- clip threshold: `3.9`.

The clip threshold is the calibration reference median gradient norm (`3.888574...`) rounded to one decimal place. Candidate values and semantic disagreements were not used to choose it. The confirmation clip endpoint remains a controlled sensitivity result, not an application-derived acceptance rule.

## Predeclared checks

### Measurement validity

1. Every candidate call must reach the tracked compiled callable; otherwise fail closed.
2. There must be no silent graph proliferation under fixed shapes.
3. Deterministic self-pair numerical and semantic discrepancies must be zero.
4. Same-state/case repeat variability should be zero under this deterministic protocol.

### Numerical and conditional structure

1. Mean absolute class-0/class-1 margin delta should be nonzero and within a factor of two of calibration (`0.00020238`).
2. No direction is preregistered for the global signed mean. A confidence interval crossing zero is compatible with state/case-conditional discrepancy rather than persistent global shift.
3. Far-boundary semantic disagreement should not exceed near-boundary disagreement.
4. Any binary class-0/class-1 flip must satisfy the deterministic geometric condition that the reference boundary distance is no larger than the absolute paired margin shift.

### Semantic and transition profile

1. Directional shift and paired disagreement are reported separately even if both are zero.
2. Class-0/class-1, argmax, and top-2 endpoints remain separate; none substitutes for the others.
3. One-step update L2 discrepancy should be nonzero and its relative mean should be within a factor of two of calibration (`0.00024982`).
4. Gradient-clipping rates and disagreement are reported, but a zero disagreement is not evidence that clipping is generally insensitive; the state count may be underpowered for a rare boundary crossing.

## Forbidden claims

- compiler correctness failure;
- natural-workload fork rate;
- arbitrary model/operator generalization;
- application harm or long-run training impact;
- operational equivalence without an externally justified tolerance.

## Post-launch protocol audit

The harness was audited after launch and `--seed` was found to control both the base parameter tensors and the per-state input/bias/target generators. Consequently, seed `20260716` changes both the model-parameter instance and the sampled cases. The run remains a valid held-out comparison from the same controlled program family, but it is **not** a conditional-on-identical-weights replication and cannot isolate finite input-state sampling from parameter-state heterogeneity.

This interpretation change was recorded before reading the confirmation outputs. A future application-facing harness must expose separate `parameter_seed` and `state_seed` fields and sample multiple parameter/checkpoint clusters explicitly.
