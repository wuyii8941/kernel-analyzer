# 批量参照求值（kernel_spec/bulk.py，2026-10-20）

## 问题

路径 A（`sufficiency/results/PATH_A_20261020/`）中 73 个调用键因参照求值超过 600 s 预算而无法诊断。
cProfile（bmm [14,128,64]×[14,64,128]，608 s）显示时间几乎全部花在合同参照上：
- `kernel_spec/aten.py` 的 `mm` 用 Python 双重循环逐元素构造 `R`（Fraction）对象，`_dot` 一项累计 597 s；
- 程序参照 Gπ（`2_tool` 的区间求值）只用了约 2 s。

## 做法

新增 `kernel_spec/bulk.py`：对少数（合同，profile，算子）条目，在 NumPy 数组上求**同一个**实数结果，以 binary64 下、上界数组返回（`BulkTensor`）。

| 条目 | 算子 | 求值方式 | 输出保证 |
|---|---|---|---|
| MATMUL real | `mm`、`bmm`、`addmm`（beta、alpha 为精确有限标量） | 输入按 binary64 精确拆成整数 × 2^e，用 python-flint `fmpz_mat` 做精确整数矩阵乘；addmm 的缩放与加项用精确有理数 | exact：下、上界是精确值向 −∞、+∞ 的 binary64 舍入 |
| ARITH real | 同形状、同 dtype 的 `add`、`sub`（alpha=1）、`mul` | `two_sum` 无误差变换；f16/bf16/f32 乘积在 binary64 中精确 | exact |
| INDEX real | 默认选项的 `embedding` | 直接取行（精确拷贝）；与适配器相同，权重含 −0 时报 invalid_input | exact |
| SOFTMAX real | `_log_softmax`、`_log_softmax_backward_data` | Arb 球算术（python-flint），128 位起，宽度超过 4 个 binary64 ulp 时升到 512、2048 位；log-sum-exp 写成 log(c)+log1p(s/c) 以保留最大项主导时的相对精度 | rigorous_enclosure（球端点转换后各向外移一个 ulp） |
| ACT torch | `silu_backward` | 同上 | rigorous_enclosure |

范围规则：
- 条目只接受不需要决定就能求值的朴素定义域：有限、同 dtype 的浮点张量，默认选项。
- 其余情况返回 None，由合同适配器求值并决定结论。
- 条目自己报拒绝时，与适配器对同一条件的检查一致（同一异常类）。
- 工作量低于 `MIN_WORK`（20 万次标量运算）的调用仍由适配器求值。

## 接入点

- `corpus/call_binding.py`：`_reference()` 先试 bulk，再回退适配器；每个任务行记录 `reference_backend`。
  - `fixed_sample_difference`、结构检查与成员检查各有数组分支，结果与原逐元素实现的含义相同；
  - 坐标和由 binary64 值精确累加。
- `kernel_spec/compare.py`：`validate_reference_values` 接受数组结果。exact 保证必须由 exact 条目给出，端点必须有限且 lo ≤ hi。
- `2_tool/src/kernel_analyzer/contract_binding.py`：`_arrays` 直接读取 `bulk_bounds()`，不导入规格代码。

## 核验

- `tests/test_bulk_reference.py`（7 项）：在生成的输入上与适配器逐坐标比较。
  - exact 条目：下、上界必须等于适配器精确值的定向舍入；
  - 包围条目：两者必须相交，bulk 宽度不超过 8 个 binary64 ulp；
  - 拒绝：bulk 拒绝时适配器结论相同，bulk 让出时不作任何主张。
- 覆盖：f16/bf16/f32/f64，极大和极小数量级，addmm 的 beta=0 与非 1 的 alpha/beta，bmm，log_softmax 正反向，silu' 零点附近，以及负零、无穷、混合 dtype、广播等拒绝情形。
- 规格全套测试 1130 项通过；工具 contract 相关测试 74 项通过。
- 真实调用：路径 A 中超预算的 bmm 键（qwen fp32 key 59）从超过 600 s 降到 32 s，688,128 个坐标参照完整，Gπ−f 没有坐标排除零。

## 限制

- exact 条目依赖 python-flint 的整数矩阵乘法，以及 Python 整数转浮点的正确舍入。
- 包围条目依赖 Arb 的严格性。
- 这些是被调用的库，不是对被测实现的假设。
- bulk 不覆盖未列出的算子与选项。
