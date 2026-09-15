# Liger FP32 chunk-order intervention (length 256)

This is a real GPU run on the Qwen3-1.7B Liger fused linear cross-entropy
implementation. Every variant uses FP32 inputs, FP32 contribution tensors and
the same forward computation. The only ordinary-variant change is the order in
which the chunk contributions are added. The Kahan variant additionally keeps
an FP32 compensation tensor for that outer sum.

The run completed all 32 declared states without execution or non-finite
errors. Forward loss and hidden-state gradients were bitwise equal across all
variants. The measured differences therefore first appear in the parameter
gradient accumulation, not in the forward loss or upstream hidden gradient.

## Results

The comparison uses the original sequential order as candidate and each
variant as a separate reference. The profiles use the fixed 8192-coordinate
measurement geometry and the 16-state confirmation half.

| reference variant | gradient total RMS / reference RMS | update total RMS / reference RMS | update additive | update aligned | update residual |
|---|---:|---:|---:|---:|---:|
| reverse | `2.41e-7` | `4.74e-9` | confirmed `4.66e-10` | confirmed `5.57e-10` | confirmed `7.41e-11` |
| even-then-odd | `1.67e-7` | `1.70e-8` | not confirmed | not confirmed | not confirmed |
| frozen permutation | `1.96e-7` | `1.07e-8` | not confirmed | confirmed negative aligned effect | not confirmed |
| FP32 Kahan | `2.27e-7` | `1.70e-8` | confirmed `5.21e-11` | confirmed `-4.59e-10` | confirmed `2.66e-11` |

At the raw gradient-vector level, even-then-odd and Kahan reduced the mean
candidate-to-reference L2 difference to about `0.74` of the reverse-order
comparison. After the nonlinear zero-moment AdamW response probe, however, the
total update RMS was not reduced. This is an informative intervention result:
changing the accumulation order can remove the reproducible update-direction
component while increasing unstructured update energy.

## Interpretation

The source explanation is the non-associativity of FP32 addition in the Liger
chunk accumulation. The length-256 run extends the earlier length-64 and
length-128 observations to a disjoint sequence-length condition. It is not a
new operator problem and does not establish a training-loss consequence.

The Kahan prediction was not confirmed as an update-level repair. The
even-then-odd schedule is a useful post-selection contrast because its three
update profile intervals cross zero, but its larger total update RMS means it
must not be called a uniformly better implementation. A long paired training
run would be needed before making any claim about loss.

The complete machine record is
`results/property/liger_fp32_chunk_order_v1/length256_order_variants_intervention.json`.
