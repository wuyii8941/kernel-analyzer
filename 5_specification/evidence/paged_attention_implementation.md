# vLLM 0.16.0 paged-attention specification review — 2026-10-19

The active PAGEDATTN contract covers the complete logical output of the installed vLLM 0.16.0
`unified_attention` call on its causal, unquantized, two-dimensional path. The retained installed source fixes the
paged layout, packed-query positions, contiguous GQA grouping, causal mask, left-window convention, score order and
the conditions selecting the two-dimensional path. PyTorch SDPA and FlashAttention documentation independently
support the scaled-softmax task, contiguous GQA, bottom-right causal alignment, inclusive window and softcap formula.

The former draft followed an unverified main-branch snapshot and accepted bidirectional calls, although the installed
0.16.0 wrapper asserts `causal`. It also allowed K and V dimensions that do not match the installed cache layout,
omitted physical dtype checks and could not check output dtype and shape. That version and its adapter remain under
`contracts/draft/` and cannot be selected as a runtime fallback.

The reviewed profile accepts finite physical f16, bf16 or f32 query and logically read cache values, physical f32
scale and softcap, signed-int32 packed-sequence metadata and a rectangular block table. A cache slot independently
proved unread may be represented as null, so irrelevant tail bytes cannot invalidate or influence the task. Every
logical read must name an in-range populated slot. Bidirectional attention, a right window other than -1, quantized
caches, descales, parallel-softmax buffers, sinks, ALiBi, query-query bias and multimodal prefix ranges are refused.

Eighteen contract examples cover block permutation, bottom-right positions, inclusive window, scale, softcap,
contiguous GQA, all three dtypes and invalid boundaries. Eight wrong formulas are rejected. Seven independent cases
use a separate 768-bit Arb element loop; three multi-sequence cases also use a public dense PyTorch formula. Complete
output checks require every coordinate, the exact shape and dtype. A retained partial-attention audit additionally
checks a 33-token paged task whose unused cache tail has independently proved irrelevance.

The installed official vLLM wrapper was run on five GPU cases: f16 paged GQA prefill, f16 decode with a window, f16
multi-token prefill with a window, bf16 with softcap and a short f32 call. Against an independent dense gathered
formula, f16/bf16 maximum absolute differences were at most `4.88e-4`; the f32 case differed by at most `2.84e-4`
because the device dot path may use reduced multiply precision. All outputs were finite and every input was unchanged.
The multi-token window case did not reproduce the suspected value-mask error and is not recorded as a bug.

This profile does not claim a task for every paged-attention implementation. Another release, cache layout, mask,
window convention, output set or optional branch requires its own source match and binding.
