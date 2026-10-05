# [inductor] Pooling backward lowerings crash on 1-element kernel_size / stride / padding / dilation (forward normalizes, backward asserts)

### 🐛 Describe the bug

Eager pooling accepts a 1-element sequence for `kernel_size`, `stride`, `padding` and `dilation` (it is broadcast to
every spatial dim), and the Inductor forward lowerings normalize it with `pad_listlike`. The backward lowerings
`avg_pool2d_backward`, `avg_pool3d_backward` and `max_pool2d_with_indices_backward` do not normalize and assert the
full length, so training under `torch.compile` fails:

```python
import torch, torch.nn.functional as F

x = torch.randn(2, 3, 9, 9, device="cuda", requires_grad=True)
for fn in (torch.nn.AvgPool2d(3, padding=(1,)),
           lambda t: F.max_pool2d(t, 3, 2, 1, dilation=(1,)),
           lambda t: F.avg_pool2d(t, (3,), (2,), (1,))):
    torch._dynamo.reset()
    fn(x).sum().backward()                       # eager: fine
    torch.compile(fn)(x).sum().backward()        # LoweringException: AssertionError
```

```
torch._inductor.exc.InductorError: LoweringException: AssertionError:
  target: aten.avg_pool2d_backward.default
  ...
  args[4]: [1]
```

The same happens for `F.avg_pool3d(x3d, (3,), (2,), (1,))` (`avg_pool3d_backward`). `max_pool3d` is fine (different
backward path). Note: reproduce with `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` if a previous successful compile of the
same graph is cached.

This sample is in the OpInfo database (`nn.functional.avg_pool2d`, padding `(2,)`), but
`test/inductor/test_torchinductor_opinfo.py` runs avg_pool1d/2d/3d (and several other ops) on one sample only
(`inductor_one_sample["cuda"]`), so it is never compiled in CI.

### Suggested fix

At the top of the three backward lowerings, normalize like the forward does:

```python
kernel_size = pad_listlike(kernel_size, 2)   # 3 for avg_pool3d_backward
stride = pad_listlike(stride, 2)
padding = pad_listlike(padding, 2)
dilation = pad_listlike(dilation, 2)         # max_pool2d_with_indices_backward
```

With this applied at runtime (wrapping the registered lowerings), all cases compile and the gradients match eager
bit for bit.

### Versions

torch 2.10.0+cu128; the same asserts are on main (`torch/_inductor/lowering.py`, 2026-10-05).
