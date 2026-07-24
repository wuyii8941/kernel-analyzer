# Step236 kernel provenance strengthening

The v0.5 artifact is the same valid layer-15 attention/MLP/kernel slice as
v0.4, with one additional provenance invariant:

- every intermediate kernel call record stores the live
  `torch._inductor.runtime.compile_tasks.*` module name;
- that name is present in the recorded live-module inventory;
- the inventory contains 28 ordered instances of the declared generated
  kernel family, and call 15 is the mapped `post_attention_layernorm` call;
- the original attention buffer is unchanged across the kernel call, so the
  kernel-only repair targets only the post-norm output.

The independent audit remains `valid: true`.  This strengthens the claim from
“same generated-symbol family” to “a replayed call of the recorded live
generated module”, while still forbidding a unique compiler-pass or source-op
root-cause claim.
