# Qwen3 GRPO Grad-Branch Repair Contract v0.9 — 2026-07-17

## Status

Frozen before v0.9 execution. v0.8 remains `INVALID`: its C arm failed during
backward with CUDA OOM and no partial result is accepted.

## Resource-lifetime correction

v0.8 retained each `Accelerator` in a process-global list. Accelerate retains its
prepared model, so completed A/B arms remained live on GPU when C began. v0.9
does not change the endpoint, scorer, graph history, branch treatment, optimizer
probe, witness, hashes, or acceptance thresholds. It changes only arm teardown:

1. after each arm, release all retained Accelerator/model references;
2. run Python garbage collection;
3. reset Dynamo's compilation cache;
4. empty the CUDA caching allocator before loading the next arm.

Every arm is still recomputed from the same saved parameters. No v0.8 partial
weight or gradient is reused.

## Unchanged hard gates and claim boundary

The v0.8 Accelerate-native realization, `[4,168] -> [4,167]` specialization
history, native `[4,128]` scorer hashes, original graph-code/node sequence, B/C
identity, branch derivative gate and independent saved-weight distance audit all
remain mandatory.

A valid result estimates only a single-state controlled branch-intervention
effect. It does not establish correctness, operator cause, natural optimizer
impact, population prevalence, or long-run harm.
