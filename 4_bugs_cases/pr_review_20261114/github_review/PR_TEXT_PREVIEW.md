# 拟提交 PR 文字：五份合并预览

均为审核稿，尚未提交给上游。FLA 先走 RFC，Trident 当前没有可接收 PR 的原仓库入口。

## B049 · linkedin/Liger-Kernel

**Title:** fix(kl_div): preserve mixed-precision input gradients

## Summary

Keep the KL backward temporary in at least FP32 and cast to the student dtype after upstream scaling and reduction. Currently it inherits the teacher dtype: uniform FP16 targets for a `(32, 32768)` mean loss can produce an all-zero FP32 student gradient instead of `-2**-35` per element.

Save the student dtype in the autograd context and promote the temporary with the student/target types. This changes storage and post-kernel scaling, not the kernel's existing FP32 target load/exp arithmetic; it does not establish FP64 kernel accuracy. The public signature and forward implementation stay the same. Extra temporary storage for low-precision targets is an explicit correctness trade-off.

## Testing Done

- The 144 new GPU parameterizations cover student/teacher dtype combinations, all reductions, both target conventions, and two upstream scales. Original source: 22 failures and 122 passes. Modified source: 144 passes.
- Full `test/transformers/test_kl_div.py` at the recorded current base: 248 passes.
- Repository-wide Ruff lint and format checks use the repository-pinned Ruff 0.14.11; results are in the evidence directory.
- FP64 references use the same physical targets, with zero absolute gradient tolerance. A representable small gradient cannot pass by disappearing.
- Bounded forward/backward timing and incremental allocation are recorded for FP16 and FP32 teachers. On `(32, 32768)`, the FP16-teacher incremental peak rises from 2,098,176 to 4,195,328 bytes. No performance gain is claimed.

```bash
python -m pytest test/transformers/test_kl_div.py
```

Hardware: NVIDIA RTX A6000; PyTorch 2.10.0+cu128; Triton 3.6.0. The full repository `make test`, convergence matrix, other hardware backends, and NCU profiling were not run. The change does not claim pretrained-model quality improvement.

Pinned base: [linkedin/Liger-Kernel@8fbf19f47e391b6e7f5d60eb09fb2b0a49ddcf81](https://github.com/linkedin/Liger-Kernel/commit/8fbf19f47e391b6e7f5d60eb09fb2b0a49ddcf81) (`main`). Environment: RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.


## B047 · BobMcDear/attorch

**Title:** Fix L1 gradients at zero residual

`L1Loss` currently returns a positive derivative when input equals target; PyTorch selects zero at this point. For example, a scalar bias added to a zero tensor has zero mean L1 loss but receives an approximately +1 bias gradient.

Return zero for exact equality and preserve the existing branches for other residuals, including their NaN behavior. Add six tests for exact and mixed residuals, mean/sum/unreduced losses, non-unit upstream gradients, and both input and target derivatives.

Validation on RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0:

- All six added GPU cases fail on the original source and pass with this change.
- The L1 selection of the existing p-loss tests reports 348 passes. Some existing FP16/no-AMP parameterizations return early; this is a pytest count, not 348 distinct GPU executions.
- Changed files pass basic Ruff correctness checks and `git diff --check`.
- A bounded `(256, 1, 28, 28)` mean-L1 forward/backward timing is included in the review evidence; it does not establish a performance improvement.

```bash
python -m pytest tests/test_p_loss_layers.py -k 'L1Loss and not SmoothL1Loss'
```

Related issue [#13](https://github.com/BobMcDear/attorch/issues/13) and PR [#19](https://github.com/BobMcDear/attorch/pull/19) concern SmoothL1/Huber thresholding; this change is confined to L1 equality. The complete p-loss matrix and model training were not rerun.

Pinned base: [BobMcDear/attorch@67c2772baf33913ef51c2d1064f746ca17b63749](https://github.com/BobMcDear/attorch/commit/67c2772baf33913ef51c2d1064f746ca17b63749) (`main`). Environment: RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.


## B038 · flagos-ai/FlagGems

**Title:** fix: keep undropped softmax probabilities in attention backward

### PR Category

Operator, OP Test

### Type of Change

Bug Fix

### Description

Keep undropped softmax probabilities in the Q/K softmax derivative. For `P = softmax(S)`, dropout factor `R`, and `U = dO Vᵀ`, the derivative is `dS = P * (R * U - sum(P * R * U))`. Applying R to the outer P introduces an extra dropout factor. Use a separate dropped probability only where dV needs it. Correct the dense and variable-length DQ/DKV loops; the shared attention-bias derivative receives the same correction.

The direct-core regression uses identical keys, making dQ exactly zero for a fixed input-independent mask, and Q=0, making dK zero. It also checks dV against the analytic same-mask result. FP16/BF16, dense/variable-length, and dropout 0/.5 give eight GPU cases: four fail on the original source and all eight pass after the change.

The mask is reconstructed using this backward core's RNG indexing. This isolates the derivative formula; it does not establish compatibility with a different forward's RNG convention.

### Validation

RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0. Changed files pass Black, isort, project-configured Flake8, and whitespace checks.

The existing attention test invocation reports 522 passes, 300 skips, and 1 failure(s). The BF16 sliding-window case at `(B=2,H=4,M=N=512,D=64)` also fails on the unmodified source with the same mismatch: one element, absolute difference `0.0001678466796875` versus allowed `0.0001`. This run is not a clean full-suite pass; its command and JUnit are retained. Tests/tolerances were not relaxed.

```bash
python -m pytest tests/test_flash_attention_dropout_derivative.py
```

Bounded same-mask dropout-backward timings are provided separately. No speedup or general performance-neutrality claim is made; non-NVIDIA backends and end-to-end forward/backward dropout integration remain outside this validation.

### Issue

Related: [#8748](https://github.com/flagos-ai/FlagGems/issues/8748) reports dropout-gradient failures in a Hygon forward/backward pairing. This PR isolates a same-mask derivative error and must not automatically close [#8748](https://github.com/flagos-ai/FlagGems/issues/8748).

### Progress

- [ ] Maintainer review
- [x] GPU regression tests for the corrected shared formula
- [ ] Broader hardware and forward/backward RNG integration validation

Pinned base: [flagos-ai/FlagGems@714859074155592820cac77d08a006f2c48eb4fc](https://github.com/flagos-ai/FlagGems/commit/714859074155592820cac77d08a006f2c48eb4fc) (`master`). Environment: RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.


## B007 · fla-org/flash-linear-attention

**Title:** [Fix] Keep normalized linear-attention key sums in FP32

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


## B054 · kakaobrain/trident

**Title:** Fix masked softmax with independent tensor strides

Masked softmax passes score strides to mask reads, and backward passes output strides to cotangent reads. Legal tensor views with independent strides are therefore interpreted incorrectly. Non-dense scores can also disagree with the newly allocated output's layout.

Materialize scores and mask before forward and cotangents before backward, so the existing shared-stride kernels receive matching layouts. This small repair adds copies for affected views. Passing independent strides into kernels could be evaluated separately.

Add 36 regressions spanning independent score/mask/cotangent layouts, both reduction dimensions, and input immutability. On RTX A6000, the complete masked-softmax file reports 43 passes after the change. A safe baseline subset with contiguous scores reports six failures and two passes. The full non-dense-score baseline was not run because its output-layout mismatch can write outside the allocation.

```bash
python -m pytest tests/test_masked_softmax.py
```

The device tests use PyTorch 2.10.0+cu128 / Triton 3.6.0 and an API-name/signature compatibility adapter for this legacy repository. The adapter does not alter masked-softmax numerical code and is attached to the review evidence; native execution on the historical supported Triton stack was not repeated. Black and whitespace checks pass. Bounded contiguous/stride-2-mask timing is provided, with no performance-improvement claim.

**Routing status:** `kakaobrain/trident` is archived. This is a validated local patch and proposed wording, not a PR that can currently be opened against that repository. Submission needs the repository to reopen or an explicitly agreed maintained destination. No arbitrary fork is presented as the upstream project.

Pinned base: [kakaobrain/trident@6c429c6b1a2d33711d964a9d643c2b4b1610afdf](https://github.com/kakaobrain/trident/commit/6c429c6b1a2d33711d964a9d643c2b4b1610afdf) (`main`). Environment: RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.


## FLA RFC

Normalized linear-attention decode stores its running key sum Z in the activation dtype. Constant unit keys cause BF16 Z to stall at 256 steps and FP16 Z at 2048 while the FP32 KV state continues to grow, breaking constant preservation. Explicit FP32 initial Z avoids that failure.

Would the maintainers prefer FP32 Z by default for `normalize=True`, or an explicit state-precision policy? The attached implementation keeps cumsum/Z and denominator in FP32. Four new GPU regressions pass, including existing explicit-FP32-state controls; the PR draft contains the affected-path checks and resource-limited full-suite attempt.

Two interface details need agreement:

1. Returned Z becomes FP32, doubling that state's FP16/BF16 storage. Cache users must preserve it. Old lost increments cannot be recovered from a rounded cache.
2. The proposed helper returns attention output in its incoming `o.dtype`. The old helper could promote output to FP32 when given explicit FP32 Z. Should that promotion remain part of the interface, or should normalized output consistently follow activation dtype?

The signature and `(KV, Z)` tuple structure do not change. The proposal affects the normalized path, not the default unnormalized profile. Limited dense/packed helper timing and incremental allocation are included in the review package; no model performance or quality result is claimed.

This RFC follows the repository's requirement to agree precision changes before opening their implementation PR. Related #268 adds the state itself; this proposal concerns precision of repeated updates. The implementation patch is ready for review, with the dtype policy explicitly pending maintainer agreement.

