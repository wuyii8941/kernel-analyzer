# Gradient value clipping implementation review — 2026-10-19

The active CLIP contract covers the complete post-state of one participating gradient buffer in
`torch.nn.utils.clip_grad_value_`. The selected tensor is a positive rank-1 through rank-4 physical finite f16, bf16,
f32 or f64 tensor, and the positive clipping bound is finite and exactly representable in that dtype.

The retained PyTorch 2.10 page states that gradients are modified in place and clipped to
`[-clip_value, clip_value]`. The retained 2.10 source applies in-place `clamp_min(-clip_value)` followed by
`clamp_max(clip_value)` to every gradient; `foreach` only selects the foreach or loop implementation. These sources
fix the selected post-state without using a tested Triton result.

The former active draft combined value clipping with total norms, norm clipping, epsilon scaling, several public
calls, multiple gradient buffers and non-finite behavior. It remains in `contracts/draft/CLIP_extended_20261019.yaml`;
none of those additional tasks is admitted by this review.

The reviewed adapter validates the tensor dtype, positive rank and dimensions, finite exactly representable physical
values, a positive finite exactly representable bound, and a `false`, `true` or `null` foreach choice before clamping
every coordinate. It refuses norm clipping, missing or unaccounted gradient buffers, special values, signed-zero
distinctions, nonpositive or unrepresentable bounds, empty tensors and unselected ranks or dtypes.

Validation comprises 19 contract examples and six incorrect formulas. Sixteen generated cases cover all four
floating dtypes and ranks one through four, all three foreach choices, boundary values and both clipped and unchanged
coordinates. An independent coordinate clamp and PyTorch 2.10 agree on every checked value, dtype and shape.
