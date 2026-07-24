# Qwen3 backward singleton source-graph repair contract v0.4

This campaign predeclares two singleton generated-family repairs:

1. `final_norm_backward`: reconstruct the tail slice-backward, residual sum and
   final RMSNorm derivative with the source graph's FP32 operations and final
   FP16 conversion.
2. `embedding_norm_backward_prep`: reconstruct the embedding RMSNorm derivative,
   valid-token mask, and clamped index preparation that precede the downstream
   embedding-gradient accumulation.

Both treatments replace the complete generated invocation and write every
declared output/mutated buffer.  They do not split constituent ATen operations
into separate causes.  Validity requires the frozen scorer and candidate graph
identity, one backward hook, one resolved generated module, one family call and
one repair.  Complete clipped-gradient and update vectors are retained.

These are selected-state repair estimands.  The embedding treatment does not
replace the downstream sort/scatter accumulation, and neither treatment grants
injection, necessity, sufficiency, population, long-run, root-cause or
correctness credit.
