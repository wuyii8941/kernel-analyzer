# Qwen3 named-operator barrier batch contract v0.1

## Scope

Frozen Qwen3-0.6B GRPO step-29 state, selected-token log-probability observable.
The batch covers every still-uncovered representative with a stable named module
boundary:

- layers 0, 14 and 27 for q/k/v/o and gate/up/down Linear projections;
- layer-0 input RMSNorm;
- layers 1, 14 and 27 for the repeated cross-layer input RMSNorm class;
- layers 0, 14 and 27 for post-attention, q and k RMSNorm;
- the singleton token embedding.

Final RMSNorm and `lm_head` are excluded because they already have valid
original-candidate four-arm evidence.  Functional operations such as bmm,
softmax, rotary application, SiLU, multiply and residual add require a separate
treatment design and are not claimed here.

## Fixed-boundary estimands

All 35 named targets are wrapped in persistent graph-break boundaries.

- Reference baseline: eager outer body, all targets eager.
- Injection for target j: eager outer body, only target j compiled.
- Barrier candidate: compiled outer body, all targets compiled.
- Repair for target j: same compiled outer body, only target j eager.

For each target, report both injection and repair contrasts.  Two exact repeats
are required for every arm.  Switching modes must not recompile the outer body.

## Interpretation

Passing results receive `BARRIER_CONDITIONED` coverage for the selected
invocation only.  They receive no original-candidate root-cause credit unless
the all-compiled barrier candidate reproduces the frozen candidate anchor.
Results do not transport across layers or states without later validation.
