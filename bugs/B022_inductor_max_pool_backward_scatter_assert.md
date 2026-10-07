# B022：Inductor 中 max_pool 反向的 scatter_add 分解遇到前向存下的下标 −1 时触发设备端断言（nightly 与 main；2.10 正确）

日期：2026-10-07。对象：PyTorch nightly 2.15.0.dev20260907+cu126（本机驱动 535 能运行的最新 CUDA nightly）；main（2026-10-07 的
`torch/_decomp/decompositions.py`）代码相同；2.10.0 不受影响。类：编译后训练在合法输入上崩溃（CUDA 上下文丢失）。

## 现象

```python
x = torch.tensor([[[7.]], [[8.]]], device="cuda", requires_grad=True)     # 长度 1 的轴
f = lambda x: F.max_pool1d(x, 2, 1, 1, dilation=2)                          # 唯一窗口采样 −1 与 +1，都是 padding
y = torch.compile(f)(x)          # [-inf, -inf]（正确）
y.backward(torch.ones_like(y))   # nightly：Assertion `index out of bounds: 0 <= ...` -> CUDA error: device-side assert
                                 # 2.10：梯度 [0, 0]（正确）
```

`return_indices=True` 与 `False` 都崩溃（反向内部仍用下标）。本轮网格中 nightly Inductor 在池化的 2,328 个 seed 级运行里有 20 个
触发断言（14 个条件）：只采到 padding 的窗口（dilation 2、padding 1），以及含 ±∞ / NaN 的输入里真实值为 −∞、与 padding 并列的窗口；
2.10 的 Inductor 在这 20 个输入上全部正常运行。

## 机制

nightly / main 新增了全局分解 `torch/_decomp/decompositions.py::max_pool2d_with_indices_backward`（docstring：
"Decomposition of max_pool2d_with_indices_backward using scatter_add"，上游 #167318 一线），Inductor 的分解表包含该算子：

```python
grad_input_flat = grad_input_flat.scatter_add(1, indices_flat, grad_output_flat)
```

它按前向存下的下标散射梯度，不处理无效下标；只在确定性模式、XPU、MPS 下退回原生 kernel。Inductor 自己的前向对「没有真实最大者」
的窗口存下 −1（2.10 与 nightly 相同），散射到 −1 时 Inductor 的间接下标越界检查（`assert_indirect_indexing`）触发
`tl.device_assert`。2.10 没有这个分解，反向逐窗口比较下标，−1 永远不匹配，梯度为 0。

## 证据

- `bugs/repro/B022_repro_inductor_max_pool_backward_assert.py`（输出 `results/essential/phase1/bugs/B022_repro_output.txt`）：2.10
  两种写法梯度 [0, 0]；nightly 两种写法都在反向触发断言。独立复算：纯 Python 枚举窗口位置，没有窗口碰到输入，真实梯度为 0。
- 网格：`results/essential/phase1/classification_pool.json.gz` 中 `nightly_inductor_cuda32` 的报错条目（隔离重跑后的 20 个）。

## 与 B021、规格歧义的关系

同一类窗口在 eager CPU 上返回越界的正下标并写错位置（B021），在 Inductor 上返回 −1：2.10 正确，nightly 崩溃（本条）。真实 −∞ 与
padding 并列时 Inductor 返回 −1、eager 返回真实位置，这一点属于规格歧义（中期报告第 6 节），但无论怎样裁定，前向自己存下的下标
让自己的反向崩溃都是缺陷。

## 上游检索（W9，2026-10-07）

- issue / PR（`max_pool inductor device-side assert`、`max_pool2d_with_indices_backward index out of bounds inductor`、`max_pool -inf
  padding indices -1`、`max_pool2d_with_indices_backward scatter_add decomposition`）：未见报告。相关：#167318（引入 scatter_add 分解）、
  #195124（开着，为不重叠窗口加快速路径，未涉及 −1）、#195123。
- main：分解体与 nightly 20260907 逐行相同，未处理 −1。
- nightly：本机最新可运行的 CUDA nightly 为 20260907（cu126 通道之后没有新构建；cu128 通道现在依赖 CUDA 13 运行库，需要 580 以上的
  驱动，本机为 535）。
- 状态：新问题候选；草稿 `bugs/upstream_drafts/B022_inductor_max_pool_backward_scatter_assert.md`，是否提交由用户决定。
