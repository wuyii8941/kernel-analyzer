# vLLM 0.16.0 LoRA shrink review — 2026-10-19

The active LORA contract covers the complete A-projection output of the installed vLLM 0.16.0 `_lora_shrink`
call for finite physical f16 or bf16 inputs and weights with a float32 output buffer. The LoRA paper fixes the
low-rank decomposition, while the retained installed source fixes the call-specific choices: the supplied scaling
is applied once, every token selects one LoRA id or `-1`, and all weight slices are evaluated.

The output pre-state is part of the task. If any token has an active LoRA, the wrapper first zeros the complete
output and then writes active rows, so every `-1` row is zero. If all mappings are `-1`, the wrapper returns before
zeroing and preserves every old output coordinate. The former draft omitted this distinction, did not fix a release
or physical dtypes, and remains under `contracts/draft` rather than serving as a fallback.

Sixteen contract examples cover the projection, scaling, multiple slices, mapping and both state rules; five wrong
implementations are rejected. Twelve independent cases use a separate exact rational element loop and a public
PyTorch formula without calling the adapter implementation. Complete-output checks require float32 dtype, the full
shape and every coordinate. Missing fields, inconsistent dimensions, invalid mappings and values not representable
in the declared input dtype are rejected.

The installed official Triton wrapper was also run on seven GPU cases spanning f16 and bf16, one and several LoRAs,
rank 3 through 8, hidden size 9 through 32, one to three slices, mixed active and `-1` mappings, and two all-inactive
calls. Against an independent float32-accumulation PyTorch formula, the maximum absolute difference was
`4.77e-7`; mixed-call `-1` rows were exactly zero, all-inactive outputs exactly preserved their pre-state, and x,
weights and mapping remained unchanged.

This profile does not cover LoRA expand, the complete B*A update, fused-MoE LoRA, embeddings, arbitrary metadata,
other output dtypes, special values, split-K execution details, gradients, layout or alias effects.
