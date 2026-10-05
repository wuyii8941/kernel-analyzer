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

## 3b. `dynamic=True` 筛查（2026-10-06，直接差分预筛）

假设：PyTorch 的 Inductor OpInfo 测试只用静态形状编译（v2.10.0 的测试文件里没有 dynamic），符号尺寸路径（0/1 特化、
符号下标化简、自动 dynamic 重编译走的路径）覆盖较少。做法：同一批 2149 个用例改 `OPINFO_DYNAMIC=1`
（`torch.compile(dynamic=True)`），先用便宜约 60 倍的直接差分（`scripts/baseline_direct_diff.py`）预筛，再与静态结果比较
（`scripts/baseline_dynamic_diff.py`）。结果在 `results/baseline/dyn_*`。

静态通过、dynamic 失败的全部是**编译崩溃**，没有静默错误；检索后都是已知或 main 已修：

| 写法 | 错误 | 状态 |
|---|---|---|
| `F.binary_cross_entropy(..., weight=w)`（含 `nn.BCELoss(weight=w)`；默认编译下第二个 batch 大小即触发） | `_infer_size` 不接受 SymInt | main 已修（#180583，2026-04-17），2.10 仍有 |
| `F.cross_entropy` 概率标签 + `weight`（`dynamic=True`，类别维为符号尺寸） | `numel()` 遇到符号尺寸 | main 部分修复（#182004，2026-06-01），nightly 待测 |
| `F.interpolate` 双线性 / 双三次 / 三线性，`align_corners=True`，非整数 `scale_factor` | `OverflowError: float infinity to integer` | 已知 #169757（该 issue 称 CUDA 正常；2.10 上 CUDA 同样失败） |
| `adaptive_avg_pool2d` / `interpolate(mode='area')` | `cannot determine truth value of Relational ... > 25` | 已知 #159550 |
| `torch.quantile` / `nanquantile` 张量 q | `numel()` 遇到符号尺寸 | 已知 #179383 |
| `repeat` 含 0 的重复次数 | `reduce() of empty iterable` | 已知 #188217 |
| 直接调用 `aten.max_pool2d_with_indices_backward`（kernel_size 为单个 int） | SymInt 不能当 `int[2]` | 直接调用 aten 的写法，不报 |

另有 `binary_cross_entropy_with_logits_24`、`logcumsumexp_1` 在 dynamic 下 CI 式差分失败而静态通过，两者 eager fp32
本身与 fp64 相差 1e-4 量级（病态），是归约次序不同造成的数值差，不是语义错误。

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
