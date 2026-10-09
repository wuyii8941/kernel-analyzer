# B015：Inductor 的 avg_pool3d 反向在 ceil_mode 下也把越界窗口按整个 kernel 体积平均（已知 issue 认为 3d 没问题）

日期：2026-10-05。对象：PyTorch 2.10.0（`ka_main`）CUDA 与 CPU；PyTorch main（2026-10-05 的 `lowering.py`）代码相同。
类：④（反向公式与 eager 语义不符）。

## 现象与机制

`F.avg_pool3d(x, k, stride, ceil_mode=True)`（`count_include_pad` 取默认 True）在 `torch.compile` 下前向正确，
反向错误；`aot_eager` 与 eager 逐位一致。最小例子：`avg_pool3d(k=3, s=2, ceil_mode=True)` 作用于 6³，每维最后一个
窗口 [4,5] 只覆盖两个位置，eager 按 2×3×3 等裁剪后的大小平均，Inductor 一律除以 27（`grad[0,0,5,5,:]`：
eager `[0.083, 0.083, 0.167, 0.083, 0.208, 0.125]`，Inductor `[0.037, 0.037, 0.074, 0.037, 0.074, 0.037]`）。

`torch/_inductor/lowering.py::avg_pool3d_backward`：

```python
if divisor_override is not None:
    scale = divisor_override
elif count_include_pad or not had_padding:
    scale = kernel_size[0] * kernel_size[1] * kernel_size[2]
else:
    scale = compute_pool_size_without_padding(pd, ph, pw)
```

与 pytorch#198119（avg_pool1d/2d）同一机制：`count_include_pad` 时除数应是裁剪到"输入 + padding"范围后的窗口
大小，`ceil_mode` 让最后的窗口越出这个范围。#198119 写道 "avg_pool3d goes through the fallback kernel and is
right"——这只在反向窗口数超过 125（`window_size > 125`，例如 kernel 12、stride 2）时成立；常见的小 kernel 都由
上面这段 lowering 生成 Triton / C++ kernel。

## 发现过程（工具）

OpInfo × Inductor 筛查（`OPINFO_SELECT=onesample`；PyTorch 的 Inductor OpInfo 测试对 avg_pool3d 只跑第一个样例）。
用例 `oib_nn_functional_avg_pool3d_6`（OpInfo 自带样例：kernel (4,5,6)、stride (2,3,2)、padding 2、ceil_mode、
count_include_pad），9 个种子：反向 kernel `triton_poi_fused_avg_pool3d_backward_0`，e_sem 相对 RMS **0.306**，
20% 的坐标区间不含 0（最大区间宽度 2.8e-13），R2 检出"幅度被拉向 0"（除数偏大），e_num 8.2e-8。
同批的 avg_pool1d 三个反向用例（0.155 / 0.057 / 0.086）属于 #198119 已覆盖的 1d 情形，作为召回。

## 证据

`4_bugs_cases/bugs/repro/B015_repro_inductor_avg_pool3d_ceil_backward.py`（输出
`pre-reorg-20261009:results/tool_spec/final/inductor_pool_bwd/B015_repro_output.txt`），梯度相对误差（float64，CUDA 与 CPU 相同）：

| 配置 | aot_eager | Inductor |
|---|---|---|
| k3 s2 ceil，6³ | 0 | **0.378** |
| k3 s2 p1 ceil，6³ | 0 | **0.248** |
| k(2,3,3) s2 ceil，(2,3,7,7,7) | 0 | **0.316** |
| OpInfo 样例 k(4,5,6) s(2,3,2) p2 ceil | 0 | **0.161** |
| 对照：count_include_pad=False | 0 | 0 |
| 对照：ceil_mode=False | 0 | 0 |
| 对照：窗口数 > 125（k12 s2，走 ATen fallback） | 0 | 0 |

## 影响与状态

- 用 `nn.AvgPool3d(..., ceil_mode=True)` 的 3D 网络（视频、医学影像）在 `torch.compile` 训练时，边界上的梯度
  被缩小，无报错。
- 上游：#198119（开着，2026-09-22）只覆盖 1d/2d，且明确认为 3d 正确；检索未见针对它的修复 PR。若修复只改
  `avg_pool2d_backward`，3d 会被漏掉。上游材料写成该 issue 的补充评论（`upstream_drafts/B015_...md`）。
