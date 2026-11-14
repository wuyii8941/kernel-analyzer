## Summary

Retain the normalization key-sum state Z and denominator in FP32 across normalized linear-attention calls. With constant unit Q/K/V, the BF16 Z state currently stops growing at 256 decode steps and FP16 Z at 2048, while the FP32 KV state continues to grow. Constant values consequently stop being preserved.

The helper is shared by chunk, fused-chunk, and fused-recurrent normalized paths. This patch requests FP32 cumsum output, evaluates the denominator in FP32, and casts the attention output to the incoming `o.dtype`.

## Interface decision (RFC required before this PR)

- Returned Z changes from the activation dtype to FP32; its storage doubles for FP16/BF16 activations. Cache consumers must retain that dtype. Previously rounded state cannot recover lost increments.
- Output is explicitly cast to `o.dtype`. With the usual low-precision/implicit state this retains the activation output dtype. **With an explicitly supplied FP32 Z, the old helper could promote the output to FP32; this proposal returns the activation dtype instead.** This behavior change requires agreement; it is not described as fully dtype compatible.
- Public argument and state-tuple structure are unchanged. The affected option is `normalize=True`.

See the separate RFC draft for the precision-policy decision. This PR body is prepared for use after that agreement and user review, not as a substitute for the RFC.

## Validation

On RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0:

- The four new decode cases run beyond each rounding boundary. Original source: two implicit-state failures and two explicit-FP32-state controls passing. Modified source: four passes.
- The affected-path selection reports 41 passes: normalized split/resume, Q/K/V and initial-state gradients, and layer padding/cache cases are included. The exact command and JUnit accompany this review.
- Changed files pass the repository Ruff configuration and whitespace checks.
- An initial full-file attempt had four CUDA OOM failures under the runner's 16% allocator cap and was interrupted after 22 passes. The first two OOMs retained large reference graphs; two following small tests also hit the cap. The first small case passes independently both before and after. This is preserved as a resource-limited attempt, not a numerical regression or a full-suite pass.
- One expanded diagnostic attempt terminated with SIGSEGV while a periodic stack-dump timer was active. Its cause is unknown. The independent FP16 layer case passes both original and modified source; the final 41-case check without that extra timer passes. Earlier failed attempts remain in the review records.
- Normalization-helper timings and allocation are recorded for dense T=1/T=128 and packed T=128. They are limited operator measurements, not model latency claims.

```bash
python -m pytest tests/ops/test_linear_attn.py -k 'not test_fused_recurrent and not test_chunk and not test_fused_chunk'
```

The excluded tests are separate unnormalized algorithm comparisons; the normalized tests still parameterize all three implementations. The complete current full-file matrix has not passed in this run. No downstream pretrained-model quality or NCU result is claimed. Related [#268](https://github.com/fla-org/flash-linear-attention/issues/268) adds Z state, while this proposal addresses precision of its repeated updates.

Pinned base: [fla-org/flash-linear-attention@2705ab8ee9a9086bcfa6d079858e4281e01215a1](https://github.com/fla-org/flash-linear-attention/commit/2705ab8ee9a9086bcfa6d079858e4281e01215a1) (`main`). Environment: RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.
