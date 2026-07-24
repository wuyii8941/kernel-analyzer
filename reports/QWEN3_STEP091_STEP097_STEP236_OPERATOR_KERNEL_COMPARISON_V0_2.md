# Three-state check: Qwen3 layer-15 operator/kernel slice

The same declared slice was independently replayed on step091, step097, and
step236.  Each state has its own snapshot, realization contract, eager/compiled
baseline transitions, boundary map, result, and independent audit:

- `results/operator_oracle/qwen3_step091_layer15_attention_mlp_slice_v0_1/`
- `results/operator_oracle/qwen3_step097_layer15_attention_mlp_slice_v0_1/`
- `results/operator_oracle/qwen3_step236_layer15_attention_mlp_slice_v0_4/`

All three audits are valid.  The three candidate contracts have the same
two-graph family, and every run passes no-op, repeatability, exact call count,
weight-storage provenance, transport-layout, cross-artifact, and no-recompile
checks.

## Cross-state result

The intermediate generated kernel
`triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12` has a
same-input post-norm discrepancy in all three states.  Its kernel-only repair
changes the continuous scorer in all three states, but changes no clipping
decision.  This supports a reproducible numerical-production signal without
supporting the claim that this kernel is the source of the observed clipping
fork.

The layer-15 MLP/layer-exit contextual replacement is conditional:

- step091: no clipping decision changes;
- step097: one of two fork coordinates changes;
- step236: the selected fork changes from on to off.

Attention replacement changes the continuous scorer but no clipping decision
in all three states.  Thus the same local numerical mechanism can have a
state-dependent semantic consequence, and a kernel producer is not equivalent
to a semantic mediator.

## Strongest supported claim

> This pipeline can reproducibly identify a generated-kernel output discrepancy
> under identical local inputs, test that kernel in isolation for endpoint
> mediation, and separately identify a state-conditioned composite MLP
> mediation effect.  It does not assign a unique root cause to an ATen op,
> fusion group, or compiler pass.

The results are implementation-relative and forward-only.  They should be used
as evidence for conditional localization and for selecting further replay
contexts, not as a global operator ranking or correctness proof.
