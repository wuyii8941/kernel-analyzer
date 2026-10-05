# OpInfo × Inductor 的工具筛查（2026-10-05）

目的：系统地扫 `torch.compile`（Inductor）的 lowering，在 PyTorch 自己的测试没有覆盖到的地方找语义问题。
入口仍是 `scripts/tool_spec_check.py`（e_num = K − K_R，e_sem = K_R − f），用例由
`scripts/tool_spec_cases_opinfo.py` 从 PyTorch 的 OpInfo 数据库自动生成。

## 1. 方法

- 每个 OpInfo（float32 与 float64 在 CUDA 上都支持）的样例输入按选项分组（kwargs 与位置参数里的非张量部分），
  每组取元素数最多（≤ 20 万）的一个样例。
- 种子 s：在 `torch.manual_seed(s)` 下重新生成该算子的样例，取同一个序号且结构相同的那个。数值随种子变化，
  形状、选项和算子的定义域（log 取正数、acos 取 [−1, 1] 等）保持不变。
- 实现：`torch.compile(op, dynamic=False)` 作用于 float32 CUDA 张量。规格：同一个 op 在 float64 上 eager 求值
  （声明误差界 2⁻⁴⁰·max|f|）。反向用例对 OpInfo 标为需要梯度的输入求 Σ⟨g_i, y_i⟩ 的梯度，上游梯度 g 取 float32
  随机张量（在 float64 中精确）。
- 筛查用 9 个种子（3 个开发、6 个确认）。e_sem 以规则、检测器或区间证实（区间不含 0）判定检出，按相对 RMS 分档。

**选什么算子。** PyTorch 的 `test/inductor/test_torchinductor_opinfo.py` 对 113 个算子只在第一个样例上做编译与
eager 的比较（`inductor_one_sample["cuda"]`），其中包括 avg_pool1d/2d/3d、max_pool、scatter_reduce.*、
index_put、插值、各类损失、归一化。也就是说，这些算子的大部分样例（`ceil_mode`、`count_include_pad`、
`include_self`、0 维输入、1 元素参数……）从未在 CI 里编译过。`OPINFO_SELECT=onesample` 只选这些算子，每个算子
最多 12 组样例，前向与反向都做。另有一批针对放宽了容差或关掉梯度检查的算子（`index_reduce.*` 的
`check_gradient: False`、`cumprod`、`logcumsumexp` 等，以及相近的 gather / index_add / scatter_reduce）。

## 2. 规模

| 批次 | 用例 | 前向 / 反向 | 算子 | 输出 | 未检出 | 常数取整级 | 小 | 候选 | 混合 | 判不了 | 报错 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| onesample | 1440 | 732 / 708 | 81 | 1434 | 1136 | 226 | 46 | 8 | 4 | 14 | 6 |
| 放宽容差的算子 | 210 | 105 / 105 | 12 | 225 | 190 | 1 | 0 | 2 | 18 | 14 | 0 |

另有 344 个输出不是 Triton 写出的（ATen fallback），不评。全量前向筛查（每个算子 3 组样例，773 个用例）仍在进行。

## 3. 结果

**新问题（3 条，均已写入 `bugs/`）：**

- B015：avg_pool3d 反向在 `ceil_mode=True` 时把越界窗口按整个 kernel 体积平均（`oib_nn_functional_avg_pool3d_6`，
  e_sem 0.306，区间证实）。pytorch#198119 只覆盖 1d/2d，并认为 3d 正确；那只在反向窗口数大于 125、走 fallback 时成立。
- B016：scatter 的下标可证明为常数时（单元素目标、图内常量下标），原子写丢掉掩码，读取者被融合到原子写之前
  （`oib_scatter_reduce_mean_19` e_sem 0.5；amax/amin 出现 inf/NaN）。PyG 的 scatter-mean（`global_mean_pool`）
  在单图时编译后返回均值 × N/⌈N⌉。
- B014：avg_pool2d/3d、max_pool2d 的反向 lowering 不规整 1 元素的参数序列，训练编译直接崩溃
  （`oib_nn_functional_avg_pool2d_5` 的 LoweringException，随后静态扫描 lowering 找到另外两处）。

**已知问题的召回：** avg_pool1d ceil_mode 反向（三个样例，0.06–0.16，即 #198119）。`index_fill` 以需要梯度的
0 维 value、多个不同下标时编译反向触发 `aten._unique` 的 size 断言（fake 实现缺失，main 上已补）。

**解释为规格问题的候选：** `normalize` 的 0 维样例（x/|x|，精确导数为 0，float64 规格本身只有舍入噪声）、
单元素 `rms_norm`（梯度正比于 eps，而 eps 默认取 `finfo(dtype).eps`，float32 与 float64 差 1e9 倍）。

**"小"档（46 个）：** 插值（双线性、双三次、三线性，前向与反向）与 p-范数（`norm`、`vector_norm`、
`pairwise_distance`），1e-7 量级，来自 fp32 的坐标比例常数与 1/p 常数取整。

**判不了：** masked.* 系列（mask 参与的组合工具未建立参照）、`__rpow__` 反向。

## 4. 筛查中发现并修正的工具 / 用例问题

- 反向用例的上游梯度原先存成 float64、launch 时再转 float32，转出来的张量不在输入摘要里，被当成外来中间值
  （输出全部判为"混合"）；改为直接存 float32。
- 需要梯度的输入改为沿用 OpInfo 生成器的标记（nll_loss 的类别权重不可导）；生成器在 `requires_grad=True` 时
  结构不同的（masked.*）退回原做法。
- 输出为共享缓冲区的视图时，特殊值类别与参照完整度改为只统计本视图（`docs/tool_changes_20261005.md` 第 6 节）。
- 0 维输出的剖面。
- `as_strided` 依赖原始 storage、`_softmax_backward_data` 带 dtype 参数，不适合 clone / 改 dtype 的用例构造，排除。

## 5. 文件

`scripts/tool_spec_cases_opinfo.py`、`scripts/data/inductor_one_sample_cuda.txt`、`scripts/opinfo_triage.py`、
`scripts/run_tool_cases_chunked.sh`；结果在 `results/tool_spec/opinfo_onesample/`、`opinfo_loose/`、
`opinfo_screen/`（本地，不入库）。
