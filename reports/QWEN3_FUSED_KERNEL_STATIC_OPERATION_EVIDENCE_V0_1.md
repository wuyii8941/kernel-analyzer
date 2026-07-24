# Static operation evidence for the layer-16 fused kernel

The generated-code file recorded in the inventory is:

`results/operator_oracle/qwen3_step236_whole_model_trace_v0_1/compiled_1/raw/inductor_trace/torchinductor/model__1_forward_4.1/output_code.py`

For `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12`, the
code contains the following fused stages:

- FP16 attention-output load converted to FP32;
- residual plus attention-output add;
- square and reduction sum (`tl.sum`);
- division by the hidden width and epsilon add;
- reciprocal square root (`libdevice.rsqrt`);
- normalization and weight multiplication;
- final conversion/store to FP16.

The independent verifier checks this file's SHA-256 against the inventory and
records the operation evidence as static provenance.  It does not infer that
any one stage is causal.  Runtime experiments establish that the fused kernel
call can mediate the endpoint, but distinguishing reduction, cast placement,
rsqrt, or multiplication requires a further internal intervention that keeps
the other stages fixed.
