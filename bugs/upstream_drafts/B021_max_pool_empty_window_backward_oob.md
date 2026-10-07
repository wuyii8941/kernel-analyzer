### 🐛 Describe the bug

**`max_pool{1,2,3}d`: a window that samples only padding returns an out-of-range index, and the CPU backward (and the CUDA 3d backward) writes that window's gradient into another channel/sample — and past the end of the gradient buffer for the last ones.**

With dilation and padding, a legal window can sample only padding positions (the arguments below pass the `pad <= ((kernel_size - 1) * dilation + 1) / 2` check). The forward correctly outputs `-inf`, but stores an index past the end of the input plane, and the backward uses it unchecked.

```python
import torch, torch.nn.functional as F

x = torch.ones(2, 1, 1, requires_grad=True)          # batch 2, 1 channel, length 1
y, i = F.max_pool1d(x, kernel_size=2, stride=1, padding=1, dilation=2, return_indices=True)
y.backward(torch.tensor([[[5.]], [[7.]]]))
print(y.flatten(), i.flatten())   # tensor([-inf, -inf]) tensor([1, 1])   <- index 1 on a length-1 input
print(x.grad.flatten())           # CPU:  tensor([0., 5.])  sample 0's gradient lands in sample 1
                                  # CUDA: tensor([0., 0.])  (expected: no input is in any window)
```

The only window samples positions -1 and +1, both padding, so no input should receive gradient. With one sample and 4 channels (upstream 10..13):

| op (every spatial size 1) | returned index | CPU grad | CUDA grad |
|---|---|---|---|
| max_pool1d | 1 | [0, 10, 11, 12] | [0, 0, 0, 0] |
| max_pool2d | 2 | [0, 0, 10, 11] | [0, 0, 0, 0] |
| max_pool3d | 3 | [0, 0, 0, 10] | [0, 0, 0, 10] |

Channel `c`'s gradient lands in channel `c + d` (`d` = number of spatial dims); for the last `d` channels it is written past the end of the gradient buffer. With 6 channels the 2d/3d CPU cases end with `corrupted size vs. prev_size` / `free(): invalid next size (fast)`.

**Where (CPU).** `aten/src/ATen/native/cpu/MaxPoolKernel.cpp`: the forward initialises `maxindex` to the first sampled position after `while (iw0 < 0) iw0 += dilationW;`, which is already past the end when the window samples nothing (the inner loop does not run, so the out-of-range initial index is stored). The backward only checks `maxindex != -1` before `grad_input_ptr[maxindex] += grad_output_ptr[index]`. Storing `-1` for windows that sample no input (or skipping them in the backward) would fix both the index and the write. The CUDA 3d backward has the same effect; CUDA 1d/2d are fine.

**Heap corruption.** With 6 channels the 2d/3d CPU cases end with glibc reporting `corrupted size vs. prev_size` /
`free(): invalid next size (fast)`, and under `MALLOC_CHECK_=3` the 3d case aborts with `malloc(): memory corruption (fast)`.
Forward-mode AD on the same input also uses the out-of-range index: `torch.func.jvp` raises on CPU and hits a device-side
assert in the CUDA gather kernel (`scatter gather kernel index out of bounds`).

**Why tests do not see it.** The OpInfo max_pool samples use kernel 3, padding <= 1 and signal lengths 3 and 6, so every window touches at least one input; a padding-only window needs a very short axis together with dilation and padding. `test_max_pool1d_corner_cases` (test/nn/test_pooling.py) does check this exact geometry (input `[[1]]`, kernel 2, padding 1, dilation 2), but only the forward output (`-inf`) with `return_indices=False` - neither the returned index nor the backward.

### Versions

torch 2.10.0+cu128 (CPU, CUDA); nightly 2.15.0.dev20261005+cpu and 2.15.0.dev20260907+cu126 reproduce (CPU 1d/2d/3d, CUDA 3d); `MaxPoolKernel.cpp` on main unchanged as of 2026-10-07.

<!-- search record (for the submitter), 2026-10-07:
1. issues/PRs: "max_pool empty window gradient", "max_pool dilation padding -inf indices", "max_pool2d_with_indices_backward cpu out
   of bounds", "max_pool gradient wrong channel", "max_pool1d return_indices out of range": nothing; PR #191920 (open) is a CUDA
   out-of-bounds READ with overflow-scale dilation, a different defect.
2. tests: OpInfo samples never produce a padding-only window; test_max_pool1d_corner_cases checks the forward value only.
3. recent PRs touching aten/src/ATen/native/cpu/MaxPoolKernel.cpp: none changing the maxindex initialisation or the backward check.
4. nightly: 2.15.0.dev20261005+cpu and 2.15.0.dev20260907+cu126 reproduce (CPU 1d/2d/3d, CUDA 3d). Per SECURITY.md, out-of-bounds
   access from arguments is filed as a regular bug. -->
