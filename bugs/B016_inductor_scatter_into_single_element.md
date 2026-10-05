# B016：Inductor 把 scatter / index_add / index_put(accumulate) 写进只有一个元素的目标时结果错误

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

只要目标在 scatter 维上只有一个元素（下标恒为 0），且 src 是计算出来的值（常数 1、`x + 1`、`exp(x)`、`cos(x)` 等，
而不是直接读入的张量），`scatter_add`、`index_add`、`index_put(..., accumulate=True)`、`scatter_reduce(sum)`、
`index_reduce(mean)` 的编译结果都把 N 个贡献算成 ⌈N / XBLOCK⌉·XBLOCK 个（N=5→8，30→32，1000→1024，
5000→5120）。

## 机制（两处）

1. **原子写丢了掩码。** 目标在该维的大小为 1 时，scatter 的间接下标被区间推理化简成常数 0。`TritonKernel.store`
   的掩码由下标里的迭代变量推出，常数下标没有迭代变量，`mask_str` 为 `None`。普通 store 无所谓（各通道写同一个值），
   `mode == "atomic_add"` 时却让填充通道（xindex ≥ xnumel）也执行原子加，加的是 src 在填充通道上的值：读入的张量
   在那里是 0（无害），计算出来的值则通常不是 0。生成的代码：

   ```
   tl.atomic_add(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp2, None, sem='relaxed')
   ```

   运行时给原子写补上迭代掩码（`xmask`）后，计数、`index_add`、PyG 式 mean pool 全部恢复正确
   （`bugs/repro/B016_*` 与下文"证据"）。

2. **消费者被融合到原子写之前。** 同样因为下标化简成常数，调度器认为 scatter 的写和后续读取的是"同一下标"，
   把读取 scatter 结果的点算子融合进 scatter 的 kernel；读放在 kernel 开头，原子加在后面，读到的是 scatter
   之前的值。单元素 `scatter_reduce(mean, include_self=True)` 的反向（计数 `scatter_add(ones, 0, idx, ones)` 之后
   被 `where` / `div` / `gather` 读取）因此得到 1.0 而不是 0.5；amax / amin 的反向得到 inf / NaN。前向最小例子：
   `c = ones(1).scatter_add(0, idx, ones_like(s)); return t / c, t.gather(0, idx) / c.gather(0, idx)`，编译结果
   为 t / 1。补上掩码不能修好这一处。

CPU（C++ 代码生成）上 `count` 与 mean pool 结果正确，`zeros(1).index_add(0, idx, ones)` 编译时
`AssertionError`（`cpp.py` store 断言下标是向量）。

## 发现过程（工具）

OpInfo × Inductor 筛查（`scatter_reduce.*` 在 PyTorch 的 Inductor OpInfo 测试里只跑第一个样例）：
`oib_scatter_reduce_mean_19`（OpInfo 自带的 0 维样例）两个梯度 e_sem 相对 RMS 0.5、区间证实；
`oib_scatter_reduce_amax_19` / `amin_18` 的 K 出现 9 个 inf/NaN 而 f 有限（特殊值类别不一致）。顺着
`aot_eager` 正确、Inductor 错误，读出反向图与生成代码，得到上面两处机制；第 1 处更普遍，随后用专门用例确认。

专门的工具用例（`scripts/tool_spec_cases_scatter1.py`，96 个种子）：`sc1_mean_pool_one_graph`（PyG 式 scatter-mean，
一个图，30 个节点）e_sem 相对 RMS **0.067**，**100%** 的坐标区间不含 0，R2/R3 检出"幅度被拉向 0"，e_num 1e-7。
TTIR 里那条原子写本来就没有掩码，K_R 如实算进了填充通道的贡献，所以偏差完全落在 e_sem。

## 证据

`bugs/repro/B016_repro_inductor_scatter_size1_fusion.py`、`bugs/repro/B016_pyg_scatter_mean.py`（输出
`results/tool_spec/final/scatter1/B016_repro_output_torch2.10.txt`）：

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

检索 pytorch/pytorch issue（scatter_add size 1 compile、index_add single element inductor、atomic_add mask 等）未见
报告；main 上 `TritonKernel.store` 的 atomic_add 分支仍直接用 `indexing.mask_str`。nightly 实测见
`results/tool_spec/final/scatter1/B016_repro_output_nightly.txt`（若已生成）。

修法：`store` 的 `atomic_add` 分支在下标不含迭代变量时用迭代掩码（或更一般地，原子写永远带迭代掩码）；调度器
不应把读取 scatter（原子写）目标的节点融合进同一个 kernel，或者不应因为下标化简成常数就判定"同一下标"。
