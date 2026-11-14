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
