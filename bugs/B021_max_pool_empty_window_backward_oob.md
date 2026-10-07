# B021：max_pool 的窗口只采到 padding 时，返回越界下标，CPU（1d/2d/3d）与 CUDA（3d）反向把梯度写到别的通道 / 样本，并越过梯度缓冲区末尾

日期：2026-10-07。对象：PyTorch 2.10.0（`ka_main`）eager CPU 与 CUDA；nightly 2.15.0.dev20261005（CPU）。类：反向语义错误 +
越界写（静默的梯度串扰；最后几个通道写到缓冲区之外，堆损坏）。

## 现象

合法参数（满足「padding 不超过有效核长的一半」：2·1 ≤ 2·(2−1)+1）下，长度为 1 的轴上 `kernel=2, stride=1, padding=1,
dilation=2` 的唯一窗口采样位置 −1 与 +1，都在 padding 里。按文档（padding 视为 −∞）输出为 −∞，没有任何输入参与，所有输入的梯度为 0。

```python
x = torch.ones(2, 1, 1, requires_grad=True)             # batch 2，通道 1，长度 1
y, i = F.max_pool1d(x, 2, 1, 1, dilation=2, return_indices=True)
y.backward(torch.tensor([[[5.]], [[7.]]]))
y, i          # [-inf, -inf]，下标 [1, 1]（越过长度为 1 的输入）
x.grad        # CPU: [0, 5]   —— 样本 0 的梯度写进了样本 1；样本 1 的梯度写到缓冲区之外
              # CUDA: [0, 0]
```

| 情形（每条轴长 1，N=1，C=4，上游 10..13） | 返回下标 | CPU 梯度 | CUDA 梯度 |
|---|---|---|---|
| max_pool1d | 1 | [0, 10, 11, 12] | [0, 0, 0, 0] |
| max_pool2d | 2 | [0, 0, 10, 11] | [0, 0, 0, 0] |
| max_pool3d | 3 | [0, 0, 0, 10] | **[0, 0, 0, 10]** |

通道 c 的梯度落到通道 c + d（d 为空间维数）；最后 d 个通道写到梯度缓冲区之外。2d/3d 在 6 个通道时 glibc 报
`corrupted size vs. prev_size` / `free(): invalid next size (fast)`（堆损坏）。

## 机制（CPU）

`aten/src/ATen/native/cpu/MaxPoolKernel.cpp`（main，2026-10-07）前向：

```cpp
while(iw0 < 0) { iw0 += dilationW; }                                    // -1 -> 1，已越过 input_width = 1
int64_t maxindex = id0 * input_height * input_width + ih0 * input_width + iw0;   // 初值是越界位置
for (... iw = iw0; iw < iw1; iw += dilationW)                            // iw1 = min(.., 1)：循环一次也不执行
...
indices_ptr[i] = maxindex;                                              // 存下越界下标
```

反向只检查 `maxindex != -1`，然后 `grad_input_ptr[maxindex] += grad_output_ptr[index]`，越界下标落在下一个通道的平面上，最后几个
通道越过缓冲区末尾。CUDA 的 1d/2d 反向不受影响（梯度为 0）；CUDA 的 3d 反向同样把梯度写到 c + 3。

## 证据

- 最小复现与不使用工具代码的独立复算：`bugs/repro/B021_repro_cpu_max_pool_empty_window_oob.py`（输出
  `results/essential/phase1/bugs/B021_repro_output.txt`）。独立复算用纯 Python 枚举窗口的采样位置，证明没有窗口碰到输入、真实梯度
  恒为 0。2.10 与 nightly CPU 2.15.0.dev20261005 结果相同（每个情形在子进程中运行，因为越界写会损坏堆）。
- 本轮网格：池化家族中只有一个这样的几何（`pool654` / `pool655`，1-D），eager CPU float64 / float32 的反向偏离规格（F），梯度
  总量性质被违反（P，4 / 1176 个 seed 级检查），eager CUDA 与 Inductor 相容。2d/3d 与 CUDA 3d 是复现时按机制推广得到的。

## 各组能否发现（W8）

| 组 | 能否发现 | 说明 |
|---|---|---|
| E（与同设备同 dtype 的 eager 比） | 否 | eager CPU 本身就是出错的一方 |
| FP64 参照（eager CPU float64） | 否 | 同一个 CPU kernel |
| CPU 对 CUDA 的设备差分 | 1d/2d 能，3d 不能 | 3d 两个设备一起错 |
| F（独立规格） | 是 | 规格：几何空窗口输出 −∞、没有可路由梯度的输入 |
| P（梯度总量 = 非空窗口的上游之和） | 是 | 不依赖参照 |

## 测试判据（W5 的预览）

OpInfo 的 max_pool 样例：kernel 3、padding ≤ 1、信号长度 3 和 6（2d/3d 某一维 dilation 为 2）——任何窗口都至少采到一个输入位置，
从不出现只采到 padding 的窗口；这种窗口需要长度为 1（或很小）的轴配合 dilation 与 padding。

## 上游检索（W9，2026-10-07）

- issue / PR 关键词（`max_pool empty window gradient`、`max_pool dilation padding -inf indices`、`max_pool2d_with_indices_backward
  cpu out of bounds`、`max_pool gradient wrong channel` 等）：未见报告。相邻：PR #191920（开着，CUDA max_pool2d/3d 在超大 dilation
  下的越界**读**，循环计数溢出），不是本问题。
- main 的 `MaxPoolKernel.cpp`：初值与反向检查未改。
- nightly：CPU 2.15.0.dev20261005 复现；CUDA 3d 部分待最新 CUDA nightly 核对（cu126 通道的最新构建为 20260907）。
- PyTorch 的 `SECURITY.md` 把「越界访问、崩溃」列为普通 bug，不走安全通告，所以草稿按普通 issue 写。
- 状态：新问题候选；草稿 `bugs/upstream_drafts/B021_max_pool_empty_window_backward_oob.md`，是否提交由用户决定。
