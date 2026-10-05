# [inductor] scatter_add / index_add / index_put(accumulate) with a provably constant index give wrong results on CUDA (unmasked atomic_add; reader fused before the atomics)

### 🐛 Describe the bug

When the index of an accumulating scatter is provably constant (a target with a single element along the scattered
dim, whose index Inductor simplifies to 0, or an index tensor built as a constant in the graph), Inductor (1) emits the atomic store without a mask and (2) may fuse readers of the
target into the scatter kernel, ahead of the atomics. Both give silently wrong results on CUDA.

```python
import torch

idx = torch.zeros(5, dtype=torch.long, device="cuda")
f = lambda idx: torch.zeros(1, device="cuda").scatter_add(0, idx, torch.ones(5, device="cuda"))
print(f(idx), torch.compile(f)(idx))   # tensor([5.]) tensor([8.])
```

**(1) Unmasked atomic_add.** The generated kernel is

```
xnumel = 5
...
tl.atomic_add(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp2, None, sem='relaxed')
```

`TritonKernel.store` takes the mask from the index (`indexing.mask_str`); the constant index has no iteration
variable, so the mask is `None`. That is harmless for a plain store but not for `atomic_add`: the padding lanes
(`xindex >= xnumel`) add too. A source loaded from memory is 0 there, but a computed source (`ones`, `x + 1`,
`x.exp()`, `x.cos()`, ...) is not, so the result counts `ceil(N / XBLOCK) * XBLOCK` contributions: N = 5 -> 8,
30 -> 32, 1000 -> 1024, 5000 -> 5120. Affected: `scatter_add`, `index_add`, `index_put(accumulate=True)`,
`scatter_reduce("sum")`, `index_reduce("mean")` whenever the index is provably constant: a one-element target, or
an index tensor built as a constant in the graph (`torch.zeros(n, dtype=torch.long)`, `torch.full`,
`torch.arange(n) // big`) into a target of any size, e.g. `torch.zeros(4).index_put((torch.zeros(30, dtype=torch.long),), x + 1, accumulate=True)`
adds 2 too much at index 0 on CUDA. Adding the iteration mask to the atomic
store (patched at runtime) fixes all of these.

**(2) Reader fused ahead of the atomics.** With the index constant, the scheduler treats the scatter's write and a
later read of the target as the same index and fuses the reader into the scatter kernel; the load is emitted before
the `tl.atomic_add`, so it sees the pre-scatter value:

```python
idx = torch.zeros(1, dtype=torch.long, device="cuda")
s, t = torch.tensor([2.0], device="cuda"), torch.tensor([6.0], device="cuda")
def g(idx, s, t):
    c = torch.ones(1, device="cuda").scatter_add(0, idx, torch.ones_like(s))
    return t / c, t.gather(0, idx) / c.gather(0, idx)
print(g(idx, s, t), torch.compile(g)(idx, s, t))   # (3., 3.) vs (6., 6.)
```

This is what breaks the backward of `scatter_reduce` on one-element tensors (the count
`scatter_add(ones, 0, idx, ones)` is read by `where`/`div`/`gather` in the same kernel):
`x.scatter_reduce(0, idx, src, "mean", include_self=True)` with one element gives gradients `[1.0, 1.0]` instead of
`[0.5, 0.5]`, and `"amax"`/`"amin"` give `inf`/`nan`. `aot_eager` is correct in every case. These OpInfo samples
(0-dim `scatter_reduce`) are in the database, but `test_torchinductor_opinfo.py` runs `scatter_reduce.*` on one sample.

This is also why `test_torchinductor_opinfo.py` disables `check_gradient` for `index_reduce.amax` / `index_reduce.amin`
on CUDA ("Gradient contains non-finite entries"): the non-finite entries appear only on the one-element samples
(4 of 8 for amax, 2 of 8 for amin), the eager gradient is finite there (e.g. 1.0 vs `inf` compiled), and the backward
kernel reads the count and then `tl.atomic_add`s into the one-element count buffer in the same kernel, so it divides by 0.

**Impact.** Any compiled code that accumulates computed values into a single bucket. One concrete case is PyG's
scatter-mean (`torch_geometric.utils.scatter(..., reduce="mean")`, used by `global_mean_pool` and mean aggregation),
whose count is `src.new_zeros(dim_size).scatter_add_(0, index, src.new_ones(n)).clamp(min=1)`: with one graph in
the batch, the compiled `global_mean_pool` returns the mean times `N / ceil(N)` (7 nodes: x0.875, 30: x0.9375,
300: x0.781); with two graphs it is correct.

On CPU the count and mean-pool cases are correct, but `torch.zeros(1).index_add(0, idx, torch.ones(5))` fails to
compile (`AssertionError` in `cpp.py` store: the index is expected to be a vector).

### Suggested fix

- In `TritonKernel.store`, for `mode == "atomic_add"` always apply the iteration masks (or at least when the index
  does not depend on them).
- Do not fuse readers of an atomic (scatter) store's target into the same kernel based on the constant-folded index.

### Versions

torch 2.10.0+cu128, RTX A6000 (sm_86). The atomic_add branch of `TritonKernel.store` is unchanged on main
(2026-10-05); nightly result: see below.
