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
