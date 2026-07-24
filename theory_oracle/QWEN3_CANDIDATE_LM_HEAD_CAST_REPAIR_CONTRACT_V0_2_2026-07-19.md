# Qwen3 lm-head weight-cast generated-kernel repair contract v0.2

V0.1 is invalid for this family.  Although its live call and restoration gates
passed, it reshaped the contiguous `[151936,1024]` source into the logical
`[1024,151936]` destination.  The destination is a non-contiguous transpose view
with strides `[1,1024]`; reshaping changed element correspondence and produced
an enormous artificial scorer effect.

V0.2 keeps the same frozen state and singleton family
`triton_poi_fused__to_copy_t_17`.  It writes the eager FP16 cast of the source
through `destination.t()`, whose shape and contiguous strides match the source.
It requires exact transposed shape/stride compatibility before writing.  No
effect or direction from v0.1 is used as evidence.

All original-candidate anchor, graph, one-call/one-repair, repeat,
no-recompilation and restoration gates remain required.  A passing result covers
only this generated cast plus transposed-view boundary at one state and supplies
no correctness claim.
