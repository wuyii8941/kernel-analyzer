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
