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
