# 五项提交文字：按独立审核修正

未向上游提交 issue、PR 或评论。

## B049

**Title:** fix(kl_div): preserve mixed-precision input gradients

## Summary

Keep the KL backward temporary in at least FP32 and cast to the student dtype after upstream scaling and reduction. The current teacher-typed temporary can erase representable FP32 student gradients: uniform FP16 targets with a `(32, 32768)` mean loss produce zeros instead of `-2**-35` per coordinate.

Save the student dtype in the autograd context and promote temporary storage accordingly. Forward and public signatures are unchanged. The kernel's existing FP32 load/exp arithmetic is unchanged; this does not establish FP64 kernel accuracy.

## Testing Done

- [Original mixed-dtype matrix](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B049_before_v1/junit.xml): 22 failures/122 passes. The [complete modified KL file](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B049_latest_full/junit.xml) passes 248 cases, including all 144 additions (student/teacher dtype combinations, reductions, target conventions and upstream scales).
- FP64 references use the same physical targets and zero absolute gradient tolerance, so complete loss of a representable gradient cannot pass.
- [`make checkstyle`](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/style/followup/liger_make_checkstyle.log) passes with pinned Ruff 0.14.11; all 500 Python files remain unchanged.
- [Bounded timing/allocation](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/PERFORMANCE_v1.md): the FP16-teacher incremental temporary peak on `(32, 32768)` rises from 2,098,176 to 4,195,328 bytes. This is a correctness trade-off; no speedup claim is made.

```bash
python -m pytest test/transformers/test_kl_div.py
```

- Hardware Type: NVIDIA RTX A6000 (48 GB); PyTorch 2.10.0+cu128; Triton 3.6.0
- [ ] run `make test` to ensure correctness
- [x] run `make checkstyle` to ensure code style
- [ ] run `make test-convergence` to ensure convergence

An optional broader unit run was [interrupted after 1,311 passes and 100 skips](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/liger_broad_attempt/junit.xml), so whole-repository validation is not claimed. Mini-model convergence, other hardware backends and NCU were not run; no training-quality claim is made.

Tested base: [`8fbf19f47e391b6e7f5d60eb09fb2b0a49ddcf81`](https://github.com/linkedin/Liger-Kernel/commit/8fbf19f47e391b6e7f5d60eb09fb2b0a49ddcf81).


## B047

**Title:** Fix L1 gradients at zero residual

`L1Loss` currently returns a positive derivative at exact equality, whereas PyTorch selects zero. A scalar bias added to a zero tensor can consequently receive an approximately +1 gradient despite zero mean L1 loss.

Return zero only for exact equality, preserving the other existing branches, including their NaN behavior. Six regressions cover mean/sum/unreduced losses, mixed residuals, non-unit cotangents and input/target gradients.

Validation: the [six new cases fail on the original source](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B047_before_v1/junit.xml) and [pass after the change](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B047_after_v1/junit.xml). The corrected command below [passes 354 parameterizations and includes all six new cases](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B047_corrected_public_command/junit.xml). Some existing FP16/no-AMP parameterizations return early, so this is a pytest count rather than 354 distinct kernel executions.

```bash
python -m pytest tests/test_p_loss_layers.py -k 'l1_zero_residual_gradient or (L1Loss and not SmoothL1Loss)'
```

Changed files pass basic Ruff correctness and whitespace checks. [Bounded same-device timing/allocation](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/PERFORMANCE_v1.md) is provided without a speedup claim. Tested base: `BobMcDear/attorch@67c2772baf33913ef51c2d1064f746ca17b63749`, RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0. The complete p-loss matrix and model training are not claimed.

Related [issue #13](https://github.com/BobMcDear/attorch/issues/13) and [PR #19](https://github.com/BobMcDear/attorch/pull/19) concern SmoothL1/Huber thresholding; this patch changes only L1 equality.


## B007

**Title:** [Fix] Keep normalized linear-attention key sums in FP32

## Summary

Preserve the normalization key sum Z across BF16/FP16 decode calls. With unit Q/K/V, Z currently stalls at 256/2048 steps while the FP32 KV state keeps increasing, so the output no longer preserves constant values. Keep Z and its denominator in FP32 and return attention output in the incoming activation dtype.

This affects `normalize=True`: it is the default for `chunk_linear_attn` and `fused_chunk_linear_attn`, but not `fused_recurrent_linear_attn`. It also affects `LinearAttention(do_feature_map_norm=True)` and the normalized chunk/fused-chunk paths of Based/ReBased. The precision policy needs an RFC agreement before this implementation PR; the separate RFC draft includes the interface decision.

## Test plan

- New decode regressions: original 2 numerical failures / 2 explicit-FP32-state controls passing; patched 4 passes. Numeric assertions precede the state-dtype assertion.
- Complete `tests/ops/test_linear_attn.py`: **58 passed**, including all three implementations, gradients, state resume, padding, cache and empty sequences.
- `tests/layers/test_based.py`: **43 passed, 8 skipped**; skips are the existing unsupported odd expanded dimensions in 16-bit chunk kernels.
- ReBased integration: **6 passed**, covering both affected modes and three dtypes, independent quadratic attention output comparisons from physical feature-map values, and finite input/parameter gradients. This does not claim an independent ReBased gradient oracle.
- Changed files pass the repository Ruff rules and whitespace checks.

Hardware/software: RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0, `FLA_CI_ENV=0`. The complete-file run used an 80% allocator cap on an otherwise idle GPU; the earlier 16% cap caused OOMs. Earlier failed/diagnostic attempts remain in the evidence notes. No test tolerance was relaxed.

```bash
FLA_CI_ENV=0 python -m pytest tests/ops/test_linear_attn.py
FLA_CI_ENV=0 python -m pytest tests/layers/test_based.py
```

## Benchmark / NCU

Same-device normalization-helper forward timings on RTX A6000 (BF16, H=2, D=64), three warm samples; descriptive medians, not a speedup claim:

| Workload | Before ms | After ms | Incremental peak bytes before → after |
| --- | ---: | ---: | ---: |
| dense T=1 | 0.016908 | 0.023263 | 1,536 → 2,048 |
| dense T=128 | 0.025306 | 0.034516 | 98,304 → 197,632 |
| packed T=128, lengths 64/64 | 0.365197 | 0.420195 | 131,584 → 230,912 |

Baseline timing revision: `aa0f314b97628ef0f5ba1a62c733cb1ba16752ae`. Current patch base: `2705ab8ee9a9086bcfa6d079858e4281e01215a1`; the relevant normalization/cumsum source is unchanged between these revisions. Candidate is the attached patch, SHA-256 `4734e2a369f6be05cabcba8c372fc51eb6c61712546556f5c98c9d8a1bf46141`. [The bounded timing/allocation procedure and samples](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/PERFORMANCE_v1.md) are publicly linked; these are helper timings, not model latency. The FP32 calculation costs more time and memory in these measurements. NCU and pretrained-model quality were not evaluated.

## Breaking changes

Returned Z becomes FP32, doubling its FP16/BF16 storage. Cache consumers must preserve FP32; lost increments in an old rounded cache cannot be reconstructed. Signatures and state-tuple structure do not change. When the caller explicitly supplies FP32 Z, the old helper could also promote attention output to FP32; this proposal returns the activation dtype instead. That output policy and default FP32 state need maintainer agreement in the RFC.

## Checklist

- [x] I have read [CONTRIBUTING.md](https://github.com/fla-org/flash-linear-attention/blob/2705ab8ee9a9086bcfa6d079858e4281e01215a1/CONTRIBUTING.md) and follow its conventions (code style, docstrings, commit prefixes).
- [x] I have read [AGENTS.md](https://github.com/fla-org/flash-linear-attention/blob/2705ab8ee9a9086bcfa6d079858e4281e01215a1/AGENTS.md) and, where my change matches its scope, the relevant skill under [.agents/skills](https://github.com/fla-org/flash-linear-attention/tree/2705ab8ee9a9086bcfa6d079858e4281e01215a1/.agents/skills).
- [ ] Dependent tests pass locally or in CI, and new behavior is covered by tests where applicable (tick as N/A for changes with no testable code, e.g. docs-only).
- [x] Kernel changes include same-hardware before/after benchmark numbers, dense + varlen where applicable (tick as N/A when no kernel code changed).
- [ ] This PR is minor/cosmetic-only (typo, formatting, style-only tweaks) — tick only if it is, and justify below.
- [ ] I understand this PR exceeds 500 changed lines and have explained its single purpose and why it should stay together below.

The size exception is not applicable: this patch is below 500 changed lines. The dependency helper returns a broad import graph; the tests above cover the changed normalization helper, its three operator callers and its direct layer integration. Unrelated backend/model suites are not claimed as executed.

Current gating item: agree the RFC output/state dtype policy and assert both implicit and explicit-FP32 output dtypes before opening the implementation PR. The unchecked dependent-test item reflects that pending interface gate; it does not negate the [58-case numerical run](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B007_complete_file_cap80/junit.xml). [All prior failed attempts and limits](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/README.md) remain available.


## B038

**Title:** fix: correct dropout derivatives in variable-length attention backward

### PR Category

Operator, OP Test

### Type of Change

Bug Fix

### Description

Correct the outer probability factor in variable-length attention's Q/K dropout derivative. Keep undropped P in `P * (R * U - sum(P * R * U))` and use dropped P only for dV.

**Dependency / submission base:** [PR #8969](https://github.com/flagos-ai/FlagGems/pull/8969) already implements the dense correction. This patch changes only the two varlen functions and is prepared against its unmerged head `69caccc0f8ca58c58a213be4fc8b0193333c87b8`. It must not be presented as a standalone, fully fixed patch on uncorrected master. Coordinate the shared fix with that PR's author/maintainers, then rebase on the agreed final base before opening this implementation PR.

The [16-case NVIDIA suite](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/pr8969/test_flash_attention_dropout_derivative.py) has eight invariant controls and eight nonzero-Q/K/V cases, dense/varlen, FP16/BF16 and dropout 0/.5. dQ/dK use an independent FP64 autograd oracle. dV uses the same oracle with an explicit budget for its preexisting probability-format conversion; inputs, masks and dQ/dK tolerances were not relaxed. [The protocol states the exact budget and limits](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/pr8969/nonzero_regression_protocol_v2.json).

GPU results: [PR head: 4 failures / 12 passes](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B038_8969_before_nonzero_v2/junit.xml); [varlen patch: 16 passes](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B038_8969_after_nonzero_v2/junit.xml). [Reverting only varlen DKV](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B038_8969_dkv_reverted_v2/junit.xml) leaves the old Q=0 controls passing but fails both new p=.5 varlen cases specifically at dK (about 99% of coordinates). [Source-intervention audit](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/pr8969/audit.json) confirms that only that function is reverted.

dBias is not GPU-validated by this matrix. The tests replay this backward core's mask; they do not validate another forward's RNG convention or Hygon integration. The initial BF16 dV comparison failures, including p=0, and their separate numerical-format explanation are [preserved](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/README.md).

```bash
python -m pytest tests/test_flash_attention_dropout_derivative.py
```

Tested on RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0. Black/isort/Flake8 and whitespace checks pass; [style records](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/style/followup/summary.json).

### Issue

Depends on / coordinates with [#8969](https://github.com/flagos-ai/FlagGems/pull/8969). Related: [#8748](https://github.com/flagos-ai/FlagGems/issues/8748); do not auto-close its forward/backward RNG integration report.

### Progress

- [ ] Change is properly reviewed (1 reviewer required, 2 recommended).
- [ ] Change is responded to an issue.
- [x] Change is fully covered by a UT.

The checked UT item covers the changed derivative and its DKV-only rollback control. It does not assert every attention feature/backend is validated.

### Performance

[Before](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B038_8969_benchmark_before/benchmark.json) and [after](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B038_8969_benchmark_after/benchmark.json) same-device, same-mask dense/varlen core samples are provided for the exact dependency base and varlen delta. These are bounded FP16 measurements with three warm samples, excluding mask setup, not a speedup or performance-neutrality claim. Dense code is unchanged in this follow-up.


## B054

**Title:** Fix masked softmax with independent tensor strides

## 🙏 Describe the pull request

Masked softmax passes score strides to mask reads, and backward passes output strides to cotangent reads. Legal tensor views with independent strides are therefore interpreted incorrectly. Non-dense scores can also disagree with the newly allocated output's layout.

Materialize scores and mask before forward and cotangents before backward, so the existing shared-stride kernels receive matching layouts. This small repair adds copies for affected views. Passing independent strides into kernels could be evaluated separately.

Add 36 regressions spanning independent score/mask/cotangent layouts, both reduction dimensions, and input immutability. On RTX A6000, the complete masked-softmax file reports 43 passes after the change. A safe baseline subset with contiguous scores reports six failures and two passes. The full non-dense-score baseline was not run because its output-layout mismatch can write outside the allocation.

```bash
python -m pytest tests/test_masked_softmax.py
```

The device tests use PyTorch 2.10.0+cu128 / Triton 3.6.0 and an API-name/signature compatibility adapter for this legacy repository. The adapter does not alter masked-softmax numerical code and is attached to the review evidence; native execution on the historical supported Triton stack was not repeated. Black and whitespace checks pass. Bounded contiguous/stride-2-mask timing is provided, with no performance-improvement claim.

**Routing status:** `kakaobrain/trident` is archived. This is a validated local patch and proposed wording, not a PR that can currently be opened against that repository. Submission needs the repository to reopen or an explicitly agreed maintained destination. No arbitrary fork is presented as the upstream project.

Pinned base: [kakaobrain/trident@6c429c6b1a2d33711d964a9d643c2b4b1610afdf](https://github.com/kakaobrain/trident/commit/6c429c6b1a2d33711d964a9d643c2b4b1610afdf) (`main`). Environment: RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.

## 💬 Additional context

The original upstream is archived. No issue number is invented and no `Resolves #` placeholder is left in the proposed text. Submission routing still needs an agreed maintained destination or reopening.

## ✅ Checklist

- [x] Code follows the project's coding conventions and style.
- [x] Tests have been added or updated to cover the changes.
- [x] Documentation has been updated, if necessary.

Documentation item: N/A for an implementation-only restoration of existing tensor-view semantics; copy/performance and runtime-adapter limitations are stated above. These checks do not override the archived-repository routing blocker.

Public evidence: [patched 43-case run](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B054_after_v1/junit.xml), [original safe subset](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B054_before_v1/junit.xml), [legacy API adapter](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/original_trident_runtime.py), [bounded copy-cost measurements](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/PERFORMANCE_v1.md). Per the user review, this remains an archived patch and will not be sent to the archived original repository or an arbitrary fork.


## FLA RFC

Normalized decode stores cumulative key sum Z in the activation dtype. Unit Q/K/V cause BF16 Z to stop increasing at 256 steps and FP16 Z at 2048 while the FP32 KV state continues growing, breaking constant preservation. Explicit FP32 initial Z avoids the numerical failure; [minimal reproducer in the first reviewed commit](https://github.com/wuyii8941/kernel-analyzer/blob/1d2d480ba995178d96c462e62070ff5b0638a2bf/4_bugs_cases/pr_review_20261114/github_review/B007/issue_draft.md).

Would maintainers prefer FP32 Z by default, or an explicit state-precision policy? The candidate implementation retains Z/denominator in FP32 and casts attention output to its incoming activation dtype.

Two interface decisions need agreement:

| Incoming state | Current returned Z | Proposed Z | Current attention output | Proposed output |
| --- | --- | --- | --- | --- |
| implicit / low-precision Z, low-precision activations | activation dtype | FP32 | activation dtype | activation dtype |
| explicit FP32 Z, low-precision activations | FP32 | FP32 | may promote to FP32 | activation dtype |

Keeping the old promotion would also need a rule for subsequent calls receiving FP32 Z returned by the new default. After agreement, the regression should assert output dtype for both implicit and explicit-FP32 inputs. The current regression asserts it only for the implicit case; it does not settle the final interface policy.

The affected option is `normalize=True`: chunk/fused-chunk default to true, fused-recurrent defaults to false. `LinearAttention` enables it with `do_feature_map_norm=True` (that layer flag defaults false). Based/ReBased chunk modes also enable it explicitly.

### Measured cost

The persistent-state type changes from 2 to 4 bytes per logical Z element for BF16/FP16 state; explicit-FP32 state remains 4 bytes. This is separate from temporary peak allocation and is not a claim that the whole model cache doubles. Cache consumers must retain the agreed state dtype; already-lost increments cannot be recovered.

Same-device RTX A6000 normalization-helper forward samples (BF16, H=2, D=64; three warm samples, descriptive medians):

| Workload | Before ms | After ms | Change | Incremental temporary bytes before → after |
| --- | ---: | ---: | ---: | ---: |
| dense T=1 | 0.016908 | 0.023263 | +38% | 1,536 → 2,048 |
| dense T=128 | 0.025306 | 0.034516 | +36% | 98,304 → 197,632 |
| packed T=128, lengths 64/64 | 0.365197 | 0.420195 | +15% | 131,584 → 230,912 |

These are limited helper measurements, not model slowdowns of the same percentages. [Raw baseline samples](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B007_benchmark_before_v1/benchmark.json), [raw candidate samples](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B007_benchmark_after_v1/benchmark.json), [measurement limits](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/PERFORMANCE_v1.md). The numerical helper/cumsum sources are unchanged between timing base aa0f314 and current proposal base 2705ab8.

### Validation and limits

The [four added decode tests pass](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B007_latest_after_regression_final/junit.xml), including explicit-FP32-state controls; the [original cases show two numerical failures and two controls passing](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B007_latest_before_regression_final/junit.xml). A subsequent [complete linear-attention file run passes 58 cases](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B007_complete_file_cap80/junit.xml) with an 80% allocator cap, including supported gradients and cache behavior. [Based](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B007_based_caller/junit.xml) and [ReBased](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B007_rebased_caller/junit.xml) caller checks are separate controls.

Earlier OOM and unexplained diagnostic SIGSEGV attempts are [retained](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/README.md). This passing complete-file run does not establish the cause or repair of that process crash, nor model quality or whole-model performance. Related [#268](https://github.com/fla-org/flash-linear-attention/issues/268) introduces Z itself; this discussion concerns repeated-update precision.

This is an RFC first, following [CONTRIBUTING.md](https://github.com/fla-org/flash-linear-attention/blob/2705ab8ee9a9086bcfa6d079858e4281e01215a1/CONTRIBUTING.md). The implementation PR waits for the state/output policy decision and its corresponding explicit dtype assertions.

## FlagGems 协调评论备稿

Your dense DQ/DKV correction in this PR overlaps the same-mask formula issue we reproduced on NVIDIA. The two varlen loops are unchanged here and retain the dropped outer-P factor.

We prepared a [varlen-only patch](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/pr8969/varlen_followup.patch) against `69caccc0f8ca58c58a213be4fc8b0193333c87b8` and [16 NVIDIA regression cases](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/pr8969/test_flash_attention_dropout_derivative.py), including nonzero Q/K/V. The [patched run passes all 16](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B038_8969_after_nonzero_v2/junit.xml); [reverting only varlen DKV](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B038_8969_dkv_reverted_v2/junit.xml) fails the new cases specifically at dK. The dV comparison separately accounts for its probability-format rounding; it is not an RNG-integration claim.

Would you prefer extracting a shared formula fix as a small prerequisite, or a varlen follow-up after this PR lands? We would rebase on that agreed final base and keep #8748 as Related rather than claim the mixed forward/backward routing issue is resolved.

