# Liger FP32 order training consequence (1024 steps)

This paired run tests the same implementation change as the FP32 order
intervention: the original sequential chunk order versus an even-then-odd
chunk order. Both conditions used the same freshly initialized small GPT-2,
the same character-level training batches, the same validation batch, and the
same FP32 AdamW schedule. The model and all optimizer state were FP32.

Both conditions completed 1024 steps with finite losses. The validation-loss
difference (original minus even-then-odd) was:

| step | difference |
|---:|---:|
| 0 | `0` |
| 256 | `0` |
| 512 | `+2.38e-7` |
| 768 | `-2.38e-7` |
| 1024 | `0` |

The largest per-step training-loss difference was `9.54e-7`. This is a real
same-FP32 paired training execution, but the measured quality difference is
too small to support a material loss claim. It complements the fixed-state
profile result: a change in FP32 reduction order can alter the signed update
component without implying measurable short-horizon quality degradation.

The complete machine record is
`results/property/liger_fp32_chunk_order_v1/fp32_order_training_1024.json`.
