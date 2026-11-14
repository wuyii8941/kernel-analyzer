# Applying and validating each patch

Each patch targets a separate upstream repository and the exact base recorded in UPSTREAM_BASES.json. It includes the source change and its tests. Run in a clean local checkout of that repository:

```bash
git switch --detach <recorded-base-sha>
git switch -c <per-fix-branch>
git apply --check /path/to/this-review/Bxxx/proposed.patch
git apply /path/to/this-review/Bxxx/proposed.patch
```

These are local preparation commands, not push/PR commands. Install the repository's dependencies in its normal supported environment before invoking pytest. The recorded run used PyTorch 2.10.0+cu128 and Triton 3.6.0; FLA additionally required its declared Transformers dependency. The public review contains no compiled caches or environment installation.

| Repository | GPU command after patch |
| --- | --- |
| Liger | `python -m pytest test/transformers/test_kl_div.py` |
| Attorch | `python -m pytest tests/test_p_loss_layers.py -k 'l1_zero_residual_gradient or (L1Loss and not SmoothL1Loss)'` |
| FlagGems | `python -m pytest tests/test_flash_attention_dropout_derivative.py` |
| FLA | `python -m pytest tests/ops/test_linear_attn.py -k 'not test_fused_recurrent and not test_chunk and not test_fused_chunk'` |
| Trident | `python -m pytest tests/test_masked_softmax.py` on the original supported stack, or the legacy-name adapter below on Triton 3.6 |

For Trident on the recorded modern stack, place the attached `B054/original_trident_runtime.py` on PYTHONPATH and execute pytest under its context manager:

```python
import pytest
from original_trident_runtime import original_trident

with original_trident():
    raise SystemExit(pytest.main(["tests/test_masked_softmax.py"]))
```

The original numerical modules are unchanged by this adapter. Inspect it before reuse. Native execution on the historical supported stack was not repeated here.

GPU jobs use isolated processes, two CPU threads, repository-local compilation caches, and a 16% PyTorch allocator cap. The first FLA full-file run exceeded that cap in a large naive reference, retaining failed-test graphs and producing later OOMs. It was interrupted; it must not be represented as an upstream numerical failure. The smaller affected-path selection preserves all test tolerances and includes all three normalized implementations, initial-state gradients, and layer/cache cases.

The original-source comparison imports the exact original numerical module but uses the same new regression tests. The complete numerical module was recovered from the pinned git base, not reimplemented in Python. For the newer FLA/Liger heads, that module was checked byte for byte against the saved original. Compact declarations, JUnit, and numerical-module hashes are in evidence/runs.json and per-case validation.json. Tests with complete process failures or old test-policy mistakes are preserved and labeled in VALIDATION_NOTES.md.

## Complete FLA follow-up

`FLA_CI_ENV=0 python -m pytest tests/ops/test_linear_attn.py -x` passes the complete 58-case file with an 80% allocator cap. The 16% cap above describes earlier attempts, not this complete-file follow-up. `tests/layers/test_based.py` was also executed. The attached `B007/test_rebased_caller.py` is a separate integration check: outputs use an independent quadratic attention reference from physical producer values; gradients are checked for finiteness, not against a separate VJP oracle.

## FlagGems dependency base

The current patch is based on unmerged PR8969 head `69caccc0f8ca58c58a213be4fc8b0193333c87b8`, not master. Fetch `refs/pull/8969/head`, create an isolated checkout at that SHA, and apply B038/proposed.patch. [Original module](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/pr8969/pr8969_original_backward.py) and [DKV-only rollback module](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/pr8969/pr8969_dkv_reverted.py) are supplied for isolated comparisons with the same final test file. They are complete numerical modules; neither is a replacement GPU implementation.

The original PR head plus the new test file yields four varlen failures; the patch passes 16; the DKV-only rollback fails two new dK checks. dQ/dK compare to ideal FP64 autograd at unchanged project tolerances; dV additionally includes the explicitly declared probability-format allowance. No dBias or RNG-integration verification is claimed.
