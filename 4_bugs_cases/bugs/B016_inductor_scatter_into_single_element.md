# B016：Inductor 在 scatter / index_add / index_put(accumulate) 的下标可证明为常数时结果错误（单元素目标、图内常量下标）

日期：2026-10-05。对象：PyTorch 2.10.0（`ka_main`），CUDA（错误结果）与 CPU（部分写法编译崩溃）；PyTorch main
（2026-10-05 的 `torch/_inductor/codegen/triton.py`）的相关代码未变，nightly 实测见"状态"。类：④（编译后的语义与
eager 不符），静默错误。

## 现象

```python
idx = torch.zeros(5, dtype=torch.long, device="cuda")
f = lambda idx: torch.zeros(1, device="cuda").scatter_add(0, idx, torch.ones(5, device="cuda"))
f(idx)                 # tensor([5.])
torch.compile(f)(idx)  # tensor([8.])
```

只要 scatter 的下标能被编译器证明是常数——目标在 scatter 维上只有一个元素（下标被区间推理化简为 0），或下标张量本身是图内构造的常量（`torch.zeros(N, dtype=long)`、`torch.full`、`arange(N) // 1000` 之类），目标可以有多个元素——且 src 是计算出来的值（常数 1、`x + 1`、`exp(x)`、`cos(x)` 等，
而不是直接读入的张量），`scatter_add`、`index_add`、`index_put(..., accumulate=True)`、`scatter_reduce(sum)`、
`index_reduce(mean)` 的编译结果都把 N 个贡献算成 ⌈N / XBLOCK⌉·XBLOCK 个（N=5→8，30→32，1000→1024，
5000→5120）。

补充（多元素目标，常量下标，CUDA）：

| 写法 | eager | compiled |
|---|---|---|
| `zeros(4).index_put((zeros(30, long),), x + 1, accumulate=True)` | [27.32, 0, 0, 0] | **[29.32, 0, 0, 0]** |
| `zeros(4).scatter_add(0, full((30,), 2), x.exp())` | [0, 0, 46.43, 0] | **[0, 0, 48.43, 0]** |
| `zeros(4).index_add(0, arange(30) // 1000, x.cos())` | [17.90, 0, 0, 0] | **[19.90, 0, 0, 0]** |
| 对照：下标由数据决定（`(x > 100).long()`） | [30, 0, 0] | [30, 0, 0] |

## 机制（两处）

1. **原子写丢了掩码。** 目标在该维的大小为 1 时，scatter 的间接下标被区间推理化简成常数 0。`TritonKernel.store`
   的掩码由下标里的迭代变量推出，常数下标没有迭代变量，`mask_str` 为 `None`。普通 store 无所谓（各通道写同一个值），
   `mode == "atomic_add"` 时却让填充通道（xindex ≥ xnumel）也执行原子加，加的是 src 在填充通道上的值：读入的张量
   在那里是 0（无害），计算出来的值则通常不是 0。生成的代码：

   ```
   tl.atomic_add(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp2, None, sem='relaxed')
   ```

   运行时给原子写补上迭代掩码（`xmask`）后，计数、`index_add`、PyG 式 mean pool、多元素目标的常量下标写法全部恢复正确
   （`4_bugs_cases/bugs/repro/B016_*` 与下文"证据"）。

2. **消费者被融合到原子写之前。** 同样因为下标化简成常数，调度器认为 scatter 的写和后续读取的是"同一下标"，
   把读取 scatter 结果的点算子融合进 scatter 的 kernel；读放在 kernel 开头，原子加在后面，读到的是 scatter
   之前的值。单元素 `scatter_reduce(mean, include_self=True)` 的反向（计数 `scatter_add(ones, 0, idx, ones)` 之后
   被 `where` / `div` / `gather` 读取）因此得到 1.0 而不是 0.5；amax / amin 的反向得到 inf / NaN。前向最小例子：
   `c = ones(1).scatter_add(0, idx, ones_like(s)); return t / c, t.gather(0, idx) / c.gather(0, idx)`，编译结果
   为 t / 1。补上掩码不能修好这一处。

**PyTorch 自己的测试因此关掉了一项检查（2026-10-06 补充）。** `test/inductor/test_torchinductor_opinfo.py`（v2.10.0）对
`index_reduce.amax` / `index_reduce.amin` 在 CUDA float16/32/64 上设 `check_gradient: False`，注释是
"Gradient contains non-finite entries"。逐样例核对：编译后的反向只在单元素样例上出现非有限值（amax 8 个样例中 4 个，
amin 8 个中 2 个，正是 0 维与长度 1 的样例），eager 梯度有限（如 1.0，编译后为 inf）。生成代码与第 2 处机制相同：
同一个 kernel 里先读计数、再对单元素计数缓冲区 `tl.atomic_add`，计数读到 0，`grad / count` 得 inf。也就是说，那条
被关掉的梯度检查报的就是本问题；修好后可以重新打开。工具在放宽容差批次里对这些用例报"特殊值类别不一致"（K 有
9 个 inf/NaN，f 有限）。

CPU（C++ 代码生成）上 `count` 与 mean pool 结果正确，`zeros(1).index_add(0, idx, ones)` 编译时
`AssertionError`（`cpp.py` store 断言下标是向量）。

## 发现过程（工具）

OpInfo × Inductor 筛查（`scatter_reduce.*` 在 PyTorch 的 Inductor OpInfo 测试里只跑第一个样例）：
`oib_scatter_reduce_mean_19`（OpInfo 自带的 0 维样例）两个梯度 e_sem 相对 RMS 0.5、区间证实；
`oib_scatter_reduce_amax_19` / `amin_18` 的 K 出现 9 个 inf/NaN 而 f 有限（特殊值类别不一致）。顺着
`aot_eager` 正确、Inductor 错误，读出反向图与生成代码，得到上面两处机制；第 1 处更普遍，随后用专门用例确认。

专门的工具用例（`pre-reorg-20261009:scripts/tool_spec_cases_scatter1.py`，96 个种子）：`sc1_mean_pool_one_graph`（PyG 式 scatter-mean，
一个图，30 个节点）e_sem 相对 RMS **0.067**，**100%** 的坐标区间不含 0，R2/R3 检出"幅度被拉向 0"，e_num 1e-7。
TTIR 里那条原子写本来就没有掩码，K_R 如实算进了填充通道的贡献，所以偏差完全落在 e_sem。

## 修复验证（工具，2026-10-06）

运行前的预测（本文"机制"一节）：补上原子写掩码只修第 1 处，第 2 处（读取者被融合到原子写之前）不受影响。
补丁 `4_bugs_cases/bugs/repro/B016_mask_patch.py`（`TritonKernel.store` 对无掩码的 `tl.atomic_add` 补 `xmask`），经
`KA_PRELOAD` 加载后用工具重测，种子与未打补丁时相同（`pre-reorg-20261009:results/tool_spec/final/scatter1_patched/`）：

| 用例 | 机制 | 未打补丁 e_sem | 打补丁后 |
|---|---|---|---|
| `sc1_mean_pool_one_graph`（PyG 单图 mean pool） | 1 | 相对 RMS 0.067，100% 坐标证实 | 6e-17，未检出 |
| `sc1_scaled_count` | 1 | 0.067，100% 证实 | 6e-17，未检出 |
| `sc1_mean_pool_two_graphs`（对照） | — | 未检出 | 未检出 |
| `oib_scatter_reduce_mean_19` 反向 | 2 | 0.5，证实 | 0.5，证实 |
| `oib_scatter_reduce_amax_19`、`oib_index_reduce_amax_0` 反向 | 2 | K 有 9 个 inf/NaN | 不变 |

预测成立：工具把同一个补丁的作用精确地分到两处机制上，第 2 处需要另外修（调度器不应把读取原子写目标的节点融合进
同一个 kernel）。附带的教训：代码生成器的补丁不改变 Inductor 缓存的键，第一次重测读到了缓存里未打补丁的 kernel，
结果"全部不变"；`KA_PRELOAD` 现在会强制关闭缓存并在结束时报告补丁改了哪些缓冲区。

**确认级规格（2026-10-06，计划 WP2）**：用严格包围的 f 重跑（对 fp32 输入做精确有理数运算，向外舍入；`sc1_mean_pool_one_graph`、`sc1_scaled_count`，`pre-reorg-20261009:scripts/strict_specs.py`，
`pre-reorg-20261009:results/tool_spec/final/strict/`）：判定分档、e_sem 大小与区间证实比例与筛查级 f（fp64 eager + 2⁻⁴⁰ 预算）完全相同。
本问题的结论标签为"严格包围"，不只是交叉核验。

## 证据

`4_bugs_cases/bugs/repro/B016_repro_inductor_scatter_size1_fusion.py`、`4_bugs_cases/bugs/repro/B016_pyg_scatter_mean.py`（输出
`pre-reorg-20261009:results/tool_spec/final/scatter1/B016_repro_output_torch2.10.txt`）：

| 写法（CUDA） | eager | compiled |
|---|---|---|
| `zeros(1).scatter_add(0, idx, ones(5)).clamp(min=1)` | 5 | **8** |
| `zeros(1).index_add(0, idx, ones(5)) + 0.5` | 5.5 | **8.5** |
| 对照：两个桶 | [5, 1] | [5, 1] |
| PyG `scatter(reduce='mean')`，1 个图，7 / 30 / 300 个节点 | — | 结果 × **0.875 / 0.9375 / 0.781**（= N / ⌈N⌉） |
| 同上，2 个图 | — | 正确 |
| 单元素 `scatter_reduce(mean, include_self=True)` 反向 | [0.5, 0.5] | **[1.0, 1.0]** |

`torch_geometric.utils.scatter(reduce='mean')`（PyG master，`global_mean_pool`、mean 聚合都用它）的 mean 分支是
`count = src.new_zeros(dim_size); count.scatter_add_(0, index, src.new_ones(...)); count.clamp(min=1)`，复现脚本
原样摘录。PyG 明确支持 `torch.compile`（代码里有 `is_compiling()` 分支），所以单图推理（batch 里只有一个图）时
编译后的 `global_mean_pool` 返回均值乘 N / ⌈N⌉，无报错。

## 影响

- 任何"把计算出来的值累加进一个元素"的编译代码：单桶计数、单段 segment 归约、单图的 PyG mean pool / mean
  聚合、累加进标量的 `index_put_(accumulate=True)`，以及单元素 `scatter_reduce` 的反向。
- 误差是系统性的：结果按 N / ⌈N⌉ 缩小（或累加值偏大 ⌈N⌉ − N 份），无报错。

## 状态

**更正（2026-10-06 19:30）：上游已知且已在 main 修复，不作为新问题提交。** 此前的检索（scatter_add size 1 compile、
index_add single element inductor、atomic_add mask 等关键词）漏掉了 pytorch/pytorch#178871（2026-03-31 报告：标量广播与
scatter_add 融合时 `tl.atomic_add` 的 mask 为 None，误差恰为 (next_pow2(N) − N)·标量）；修复 PR #179833（提交
7e9892106e，2026-04-16，在 2.10 发布之后）给 `indexing()` 加了强制掩码，`store()` 对 `atomic_add` 使用。此前写的
「main 上 atomic_add 分支仍直接用 `indexing.mask_str`」来自较早的源码，不成立。

nightly 实测（2.15.0.dev20260907+cu126，CUDA，`4_bugs_cases/bugs/repro/B016_repro_inductor_scatter_size1_fusion.py`）：计数 5（2.10
为 8）、index_add 5.5、单图 mean pool 与 eager 差 7.5·10⁻⁸、单元素 `scatter_reduce(mean)` 反向 [0.5, 0.5]（2.10 为
[1.0, 1.0]）。生成代码：原子写带 `xmask`，读取者（clamp）在单独的 kernel 里——第 2 处机制在 nightly 上同样不再出现，
是否由同一 PR 修复未查。受影响的是 2.10 及更早的发布版。工具在这一条上的检出与修复验证仍然成立（作为对已知问题的
召回），但「上游未见报告」的说法撤回。

修法：`store` 的 `atomic_add` 分支在下标不含迭代变量时用迭代掩码（或更一般地，原子写永远带迭代掩码）；调度器
不应把读取 scatter（原子写）目标的节点融合进同一个 kernel，或者不应因为下标化简成常数就判定"同一下标"。
