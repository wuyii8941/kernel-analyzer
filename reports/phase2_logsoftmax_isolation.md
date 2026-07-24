# Phase 2 Log-Softmax Isolation

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- same model forward supplies both log-softmax paths: PASS
- token and vocabulary dimensions identical: PASS
- two independent measured calls per path: PASS
- canonical autocast output dtype recorded: PASS
- CUDA exp/log ULP contract established: FAIL / pending
- large-vocabulary kernel reduction order established: FAIL / pending

## Delta Self Control
Logits self equal: True; FP16 log-softmax self equal: True; FP32 log-softmax self equal: True.

## Summary
| samples | positions | vocabulary_outputs | fp16_differs_rounded_fp32 | rounded_equality_rate | target_delta_mean | target_delta_p99 | target_delta_max | max_fp16_vs_fp32_abs | logits_dtype | half_input_output_dtype | float_input_output_dtype | logits_self_equal | fp16_logsoftmax_self_equal | fp32_logsoftmax_self_equal | analytic_legal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 512 | 77791232 | 77603774 | 0.0024097574389874686 | 1.0199898348162151e-07 | 4.76837158203125e-07 | 4.76837158203125e-07 | 3.814697265625e-06 | torch.float16 | torch.float32 | torch.float32 | True | True | True | False |

## Interpretation
Under canonical autocast, half-input log_softmax returns FP32, so this is an input-dispatch comparison rather than final FP16 output rounding. Kernel reduction order and CUDA exp/log ULP contracts are still required for an analytic legal bound.

## External Validity
Measured with FP16 autocast on Tesla T4. It isolates the local output behavior of this installed PyTorch build only; native BF16 requires separate measurement and bounds.
