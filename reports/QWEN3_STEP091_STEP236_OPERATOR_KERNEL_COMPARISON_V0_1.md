# Cross-state check: Qwen3 layer-15 kernel slice

Two independent matched states were run through the same declared layer-15
attention/MLP and intermediate generated-kernel slice:

- step091: `results/operator_oracle/qwen3_step091_layer15_attention_mlp_slice_v0_1/`
- step236: `results/operator_oracle/qwen3_step236_layer15_attention_mlp_slice_v0_4/`

Both runner reports and independent audits are valid.  The candidate graph
family hashes are the same across the two realization contracts, and each run
passes no-op, repeatability, call-count, weight-storage, layout, provenance,
and no-recompile gates.

## Stable and state-dependent parts

The intermediate generated kernel
`triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12` shows a
same-input post-norm discrepancy in both states.  Thus the kernel-level
production signal is reproducible under these two states.

Kernel-only repair changes the continuous scorer in both states but produces no
semantic clipping change in either state.  This is evidence that the kernel's
local numerical discrepancy is not, by itself, the observed clipping fork in
these cases.

The layer-15 MLP contextual replacement is state-dependent: it changes the
clipping decision in step236, but not in step091.  Attention replacement also
does not change clipping in either state.  This is exactly why a single-state
repair result cannot be promoted to a global operator ranking or a compiler
root-cause claim.

## Current claim

The repeated evidence supports a narrower but useful conclusion:

> The pipeline can distinguish a stable kernel-local numerical producer from a
> state-conditioned semantic mediator.  In the two tested states, the former
> is the intermediate kernel's post-norm output, while the latter appears at
> the layer-15 MLP/layer-exit boundary only for step236.

The MLP result is therefore a conditional mediation signal, not a persistent
operator bias.  More states are required before making any population-level
statement about operator contributions.
