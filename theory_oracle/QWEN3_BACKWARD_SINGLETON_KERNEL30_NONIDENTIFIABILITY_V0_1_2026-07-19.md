# Qwen3 backward singleton kernel-30 non-identifiability note v0.1

`triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_slice_backward_sum_view_30`
executes once, but its three outputs are not the three eager reductions.  Each
output is a compiler-private six-way partial-reduction buffer with physical
shape `[1,1,1024,6]`.  A later selected call of the repeated
`triton_per_fused_sum_34` family combines those partials.

Eager semantics specifies only the final reductions over batch and token axes;
it does not specify how their values should be partitioned among six internal
buffers.  Consequently there is no unique eager value for kernel 30's direct
output interface.

Writing an eager total into one partial slot and zeros into the other five
would reproduce the downstream total, but that intervention is consumer-aware.
Replacing kernel 30 together with the selected `sum_34` call would be a region
treatment.  Neither is a standalone operator-level eager repair.

Therefore kernel 30 is runtime-observed but operator-level repair is
`UNINSTANTIATED_NONIDENTIFIABLE_INTERNAL_ABI`.  It receives no causal credit.
This is not evidence of zero effect or correctness; it is a limit on the chosen
operator-level estimand.
