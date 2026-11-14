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
| Attorch | `python -m pytest tests/test_p_loss_layers.py -k 'L1Loss and not SmoothL1Loss'` |
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
