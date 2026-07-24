# Qwen3 step236: adjacent generated-kernel localization

The same live generated kernel family was analyzed at two adjacent runtime
calls, with independent audits:

- layer15: [audit](</data1/tzh/forkcert/results/operator_oracle/qwen3_step236_layer15_attention_mlp_slice_v0_5/audit.json>)
- layer16: [audit](</data1/tzh/forkcert/results/operator_oracle/qwen3_step236_layer16_attention_mlp_slice_v0_2/audit.json>)

Both calls belong to
`triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12` and are
mapped by exact `post_attention_layernorm.weight` storage identity.  The
inventory maps the calls to kernel IDs ending in `:411` (layer15) and `:437`
(layer16); both use the same recorded live compile-tasks module and pass all
provenance/context gates.

The local same-input post-norm discrepancy is reproducible at both calls.  The
kernel-only repair has different endpoint behavior:

| call | same-input production | kernel-only continuous change | clipping change |
|---|---|---|---|
| layer15 / `:411` | observed | observed | none |
| layer16 / `:437` | observed | observed | one fork changes off→on |

This is the first direct evidence in the project that the same generated
kernel family can contain a call whose own output discrepancy mediates a
semantic event, while a neighboring call in the same family does not.

The layer16 result is therefore a valid **call-level generated-kernel
mediation** claim in this fixed state and suffix.  It is not an individual
ATen-op claim: the inventory associates the fused kernel with several
post-fusion nodes (`add`, `mean`, `pow`, `rsqrt`, `mul`, conversion and view),
and the current intervention does not distinguish those constituents.

The appropriate next refinement is constituent-op isolation inside this fused
kernel, with the same-input and kernel-only controls retained.  Until that is
done, the claim must remain “kernel call implicated”, not “the reduction/add/
cast op is the root cause”.
