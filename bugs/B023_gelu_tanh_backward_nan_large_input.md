# B023：`gelu(approximate='tanh')` 的反向在 |x| ≥ 约 1.85e19 时返回 NaN（float32 / bf16；eager CPU、CUDA 与 Inductor 分解共有）

日期：2026-10-07。对象：PyTorch 2.10.0+cu128（CPU、CUDA、Inductor）；nightly 2.15.0.dev20260907+cu126 与 2.15.0.dev20261005+cpu
同样复现；main 上三处公式未变。类：④（公式的计算次序使中间量溢出，得到错误的特殊值类别）。量级：只在 |x| ≥ 1.8447e19（float32
中 x·x 溢出）时出现，实际训练中极少见。

## 现象

```python
x = torch.tensor([1e10, 1.8e19, 1.9e19, 1e20, -1e20], requires_grad=True)
F.gelu(x, approximate="tanh").backward(torch.ones_like(x))
x.grad   # [1., 1., nan, nan, nan]；float64 与 erf 形式：[1., 1., 1., 1., 0.]
```

真实导数在 x > 0 时为 1、x < 0 时为 0（可表示、有限）。bf16 输入同样（opmath 为 float32）；float16 输入到不了这个范围。前向正确。

## 机制

三处实现写的是同一个公式（`aten/src/ATen/native/cuda/ActivationGeluKernel.cu::GeluBackwardCUDAKernelImpl`、
`aten/src/ATen/native/cpu/Activation.cpp` 的标量与向量化路径、`torch/_decomp/decompositions.py::gelu_backward`）：

```
x_sq = x * x;  inner = kBeta * (x + kKappa * x_sq * x);  tanh_inner = tanh(inner)
tanh_derivative  = 1 - tanh_inner * tanh_inner          # |x| 大时恰为 0
inner_derivative = kBeta * (1 + 3 * kKappa * x_sq)      # x_sq 溢出为 inf
right_derivative = left * tanh_derivative * inner_derivative   # (0.5x * 0) * inf = 0 * inf = NaN
```

|x| ≥ 1.8447e19 时 `x_sq` 在 float32 中溢出为 inf，而 `tanh_derivative` 早已精确为 0，乘积为 NaN。在 `tanh_derivative == 0` 时把
`right_derivative` 置 0（或先用不会溢出的形式写 inner_derivative）即可。

## 发现方式与方法对照

2b 激活家族预注册的高风险组合（gelu_tanh × huge，数值 1e30 级）：E64（同一实现的 float64 运行有限）在 eager CPU / CUDA float32、
nightly、Inductor 上都报类别不一致（198/198 个元素 NaN）；E（Inductor 对 eager）看不到（两者一致地 NaN）；预注册的奇偶恒等式
性质在 huge 数值上不检查（溢出前提）。模式 A 的 FR（Inductor 反向 kernel）报 K 与 K_R 的特殊值类别不一致 594/594（K = NaN，K_R 有限，且 K_R 的区间包含 float64 eager 的结果），即工具不需要规格也把它归为数值溢出造成的错误类别（`results/essential/phase2b/activations/fr_modeA.json`）。规格交付后 F 可直接判定。

## 证据

- `bugs/repro/B023_repro_gelu_tanh_backward_nan.py`，输出 `results/essential/phase2b/bugs/B023_repro_output.txt`（2.10、nightly CUDA
  20260907、nightly CPU 20261005）。
- 网格：`results/essential/phase2b/activations/analysis.json`（`acti_gelu_tanh_huge_contiguous`，E64 的 `da`）。

## 状态

检索未见报告（2026-10-07，GitHub API：「gelu tanh backward nan」「gelu approximate tanh nan gradient large」「gelu_backward tanh
overflow」「GeluBackward tanh NaN」；相邻的 #189234 处理 erf 形式中 1 + erf 的相消，不涉及 tanh 形式的反向）。实际影响小（需要
1.85e19 量级的激活）；草稿 `bugs/upstream_drafts/B023_gelu_tanh_backward_nan.md`，是否提交由用户决定。
