# Qwen3 original-candidate fused SiLU-multiply repair contract v0.2

The subject, selected state, calls 0/14/27, endpoints, fail-closed gates and
claim limits are unchanged from v0.1.  V0.1 failed before producing a repaired
output because the two equal-numel live buffers carried different logical view
shapes.

V0.2 predeclares one treatment correction: require equal element counts and
reshape the non-mutated buffer to the mutated buffer's logical shape before
performing FP16 SiLU materialization and FP16 elementwise multiplication.  This
matches the generated kernel's flat `xindex` correspondence.  It does not relax
any gate and no target effect or direction was observed before this revision.
