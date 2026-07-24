# Qwen3 backward singleton-repair contract v0.2

This revision applies only to the `silu_mul` and `silu_mul_backward`
treatments.  Revision 1 failed before producing an endpoint because the
generated source and destination buffers used different tensor views of the
same logical coordinate set.  Revision 2 predeclares a reshape-only view
mapping before copying the eager-operation result.  It does not change the
arithmetic semantic boundary, selected state, endpoints, or validity gates.

The failed revision-1 run is retained and receives no causal evidence credit.
All interpretation limits from the v0.1 contract continue to apply.
