# Layer-16 generated-kernel mediation across states

The same generated-kernel call was analyzed in two independent matched
states:

- [step236 audit](</data1/tzh/forkcert/results/operator_oracle/qwen3_step236_layer16_attention_mlp_slice_v0_2/audit.json>)
- [step097 audit](</data1/tzh/forkcert/results/operator_oracle/qwen3_step097_layer16_attention_mlp_slice_v0_1/audit.json>)

Both calls are runtime call 16 of
`triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12`, mapped by
the exact `model.layers.16.post_attention_layernorm.weight` storage identity.
Both audits pass all production, mediation, provenance, transport, repeat, and
no-recompile gates.

In both states:

- the kernel has a reproducible same-input post-norm discrepancy;
- replacing only that kernel output changes the continuous scorer;
- the kernel-only repair changes at least one clipping decision.

The direction is state-dependent: step236 changes one fork off→on, while
step097 changes one fork on→off.  This is a useful result, not a contradiction:
it shows a stable call-level semantic mediation capability without implying a
fixed global numerical bias direction.

The neighboring layer15 call in the same kernel family has local production in
three states but no kernel-only clipping mediation.  Therefore the analysis is
now sensitive to both runtime call identity and state-conditioned endpoint
effect; grouping by generated-symbol name or ranking by raw delta would lose
this distinction.

The strongest claim remains “generated-kernel call implicated under the declared
fixed suffix and state distribution”.  The fused kernel still contains several
post-fusion operations, so no individual `add`, `mean`, `rsqrt`, cast, or
`mul` can yet be called the root cause.
