# Masked softmax reads independent tensor views with the wrong strides

The forward operation reuses score strides for mask reads; backward reuses output strides for cotangent reads. The independent expected operation is `torch.where(mask.bool(), -inf, scores).softmax(dim)`.

```python
import torch
import trident

scores = torch.zeros((2, 128), device="cuda", requires_grad=True)
backing = torch.zeros((2, 256), dtype=torch.bool, device="cuda")
mask = backing[:, ::2]
mask[:, ::2] = True
actual = trident.function.masked_softmax(scores, mask, 1)
expected = scores.masked_fill(mask, float("-inf")).softmax(1)
print(actual.masked_select(mask).sum(), expected.masked_select(mask).sum())
```

Expected: masked positions carry exactly zero probability. Original source reads a different physical mask layout. Independent cotangent and score layouts expose related violations of the shared-stride assumption.

Materializing scores/mask before forward and cotangents before backward restores the required layouts. Six of eight safe original cases fail; all 43 tests in the patched masked-softmax file pass. Execution on Triton 3.6 uses the attached legacy API compatibility adapter, with unchanged numerical kernels. The full non-dense-score original matrix is avoided because its output layout can write out of bounds.

The repository is archived, so this issue/PR text is preserved for review and cannot currently be posted to the original upstream. Native validation on the historical supported stack and an agreed submission destination remain separate requirements.

Pinned base: [kakaobrain/trident@6c429c6b1a2d33711d964a9d643c2b4b1610afdf](https://github.com/kakaobrain/trident/commit/6c429c6b1a2d33711d964a9d643c2b4b1610afdf) (`main`). Environment: RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.
