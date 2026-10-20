# 路径 A 收口 C 包（LAYOUT / POOL / INTERP）独立复核（2026-10-21）

复核人：Claude（独立复核方，未参与起草）。只新增了本文件和 `tests/test_layout_pool_interp_a2_review.py`，没有修改任何草稿。

## 复核对象（当前工作树文件）

| 文件 | sha256 |
|---|---|
| contracts/draft/LAYOUT_path_a_A2_20261020.yaml | 34590e28da2664b20fef4d883a8fd7a79b7bc8f3e1817cd2867d4797dad0cb93 |
| contracts/draft/LAYOUT_adapter_path_a_A2_20261020.py | b2601854d7e138e4f343a036868676ad6ec0d15e500225ed02dbfa0571b1e503 |
| contracts/draft/POOL_path_a_A2_20261020.yaml | 3f39d27b20911ab9d6f9bada262aa19cb3fb5f04f3a08a1045fb3b8cf573f1af |
| contracts/draft/POOL_adapter_path_a_A2_20261020.py | 3a0a9591039bf7b290d44ba12f1884027d057d1d727e81d09bb90f5f12afdaa6 |
| contracts/draft/INTERP_path_a_A2_20261020.yaml | 91c5fd9119517077fdfecfbc0a96c34b9978d2dda354d217399c02b113ca97e9 |
| contracts/draft/INTERP_adapter_path_a_A2_20261020.py | 4a263585af639f28f37fcf7235cd367d890584a6789555bff24d3ef954650d5e |

三个适配器在工作树里相对提交 d36cdee 还有一处未提交的修改：读取同目录合同，不再读取顶层合同。复核针对的是上表中的工作树版本。

## 结论

| 合同 | 结论 |
|---|---|
| LAYOUT | **ISSUES**（1 条，实现缺陷；记录到的调用不受影响） |
| POOL | **CONFIRMED** |
| INTERP | **ISSUES**（1 条，语义问题；记录到的 D18 调用不受影响） |

## 1 来源核对

逐条打开了快照，均与条目一致，没有发现条目里有原文没有的内容：
- `LPI_NATIVE_FUNCTIONS`：pad、constant_pad_nd、slice_backward、repeat_interleave 三个重载、tril、avg_pool2d、avg_pool2d_backward、upsample_nearest2d、.vec、backward 以及 _upsample_nearest_exact2d。
- `LPI_DERIVATIVES`：slice.Tensor、avg_pool2d、avg_pool2d_backward、upsample_nearest2d。
- `LPI_DECOMP`：slice_backward 为 `slice_scatter(new_zeros)`；`_compute_upsample_nearest_indices` 为 `isize/(isize*s)` 或 `isize/osize`，再与 float32 的 arange 相乘后 `.to(int64)`。
- `LPI_T_REPEAT_INTERLEAVE`、`T_PAD`、`T_AVGPOOL2D`、`T_F_AVG_POOL2D`、`PH1_DECISIONS.POOL_A1`。

只靠参照程序的依据都写了非主张：
- LY.SLICE_BACKWARD 有对应的 nonclaim；
- IN.ATEN 有对应的 nonclaim；
- PA.ATEN 用派生表把 backward 定义为 VJP，同时以文档化的 PA.DIVISOR 为基础，并保留了任务选择。

## 2 语义核对

- **LY.EXACT_DATA**：
  - 整型填充值是范围内的整数时原样存储，否则 unsupported；b8 只接受 False/True/0/1。
  - 截断、四舍五入、非零即 True 这几种备选都列出了，也注明了 eager 的行为没有写进文档。取舍诚实。
  - covers 中的每个 ATen 算子确实只做选取或搬运，不涉及浮点运算。
- **LY.ATEN_PAD、LY.SLICE_BACKWARD、LY.REPEAT_INTERLEAVE**：与来源一致。
  - INT64_MAX 作为 end 会被夹到维度末尾；
  - step ≤ 0 是 invalid_input；
  - repeat_interleave 中输出下标 j 读取 floor(j/r)。
- **PA.ATEN / D8**：
  - 除数保留为任务选择（`task_choice_required: true`），所以 D8 会归入第二步，这是诚实的。
  - 备注（不阻塞）：`T_F_AVG_POOL2D` 快照第 9 行把 kernel_size 描述为 "size of the pooling region"，这正是 R1（除以完整核大小）的文本依据。POOL.yaml:151 说页面 "say only that the default divisor is the size of the pooling region"，没有点出这一点。建议补一句，但它不改变结论，因为任务选择本来就保留着。
- **IN.ATEN**：重载名已经确定了 legacy nearest（exact 版本是另一个算子），所以这里不需要任务选择，这一点正确。
- **IN.ATEN_SCALES（有 scale 时）**：
  - 三个条件合起来，足以保证源码的浮点求值与 floor(o/s) 一致：isize·s 在 binary64 中精确，1/s 在 binary32 中精确，o/s 在 binary32 中精确。
  - 这些条件偏保守：例如 s=3.0、1.5、2.5 都会被拒绝（unsupported），属于覆盖范围的限制，不算错误。
  - D18 的 s=2.0 满足全部条件。

## 3 问题

**LAYOUT-1**（实现，轻度）
- 位置：`LAYOUT_adapter_path_a_A2_20261020.py:163`（`_exact_scalar`）。
- 现象：只捕获了 `(TypeError, ValueError, ZeroDivisionError)`。`Fraction(float('inf'))` 抛出的是 `OverflowError`，于是 aten.pad 或 constant_pad_nd 的填充值为 ±inf 时，结果类别是 implementation_error。domain 写的是 unsupported（"infinite data"），而第 167 行的 `if not item.finite` 永远执行不到。
- 范围：
  - 新增的整型路径受影响，例如 i32、i64 的 ±inf；
  - 新增的 b8 路径受影响；
  - 浮点路径 `:148` 原有同样的缺陷，顶层已验证版本对 f32 inf 也是 implementation_error。
- 证据：`test_infinite_fill_is_unsupported`（4 个参数组合，strict xfail）。
- 修法：在 except 中加上 `OverflowError`，或者先判断 `math.isinf`。

**INTERP-1**（语义，阻塞晋升）
- 位置：
  - `INTERP_path_a_A2_20261020.yaml:32`、`:98`、`:125`（IN.C6）、`:332`；
  - `INTERP_adapter_path_a_A2_20261020.py:192-193`。
- 问题：
  - 没有 scale 时，profile 无条件断言下标为精确的 floor(o·isize/osize)，而且 `task_choice_required: false`。
  - 但它所依据的保留源码（LPI_DECOMP.UPSAMPLE_NEAREST_SCALES）计算的是 trunc(fl32(o)·fl32(isize/osize))。
  - 草稿在有 scale 的分支里自己确立了一条原则：浮点舍入会决定结果时就拒绝（unsupported）。无 scale 分支没有执行这条原则，只在非主张里写了"不建模"，却照样给出值。
  - 此外，同一规则在公开 profile 里是 IN.INDEX（`needs_binding`，任务选择）。重载名只确定了 legacy 与 exact 的约定，并没有确定按精确算术还是按 float32 算术求下标。
- 反例：isize=2、osize=82 时，o=41：
  - 精确规则给出 1；
  - 源码公式给出 0；
  - PyTorch 2.10 CPU eager 的 `aten.upsample_nearest2d(x[1,1,1,2], [1,82])` 实测也是 0。
  - 在 isize、osize ≤ 200 范围内共有 631 对 (isize, osize) 两者不一致。
  - 按现稿，路径 A 遇到这类调用时，会把遵循源码的实现判为错误。
- 证据：
  - `test_no_scale_source_arithmetic_differs_from_exact_rule`（通过）；
  - `test_no_scale_branch_refuses_or_agrees_with_source`（strict xfail）。
- 修法（任选其一）：
  - (a) 无 scale 时按有 scale 分支的同一标准把关，只在 fl32(isize/osize) 精确、并且每个 o·isize/osize 在 binary32 中精确时给出值，否则 unsupported；
  - (b) 把无 scale 情形拆成任务选择 profile。
- 两种修法都不影响 D18（D18 带 scale 2.0）。修改后需要补写决定的备选，以及对应的例子和变异。

## 4 独立实现测试

`tests/test_layout_pool_interp_a2_review.py`：期望值由我自己写的逐坐标循环（Fraction）独立算出，没有调用草稿适配器的辅助函数，也没有调用 kernel_spec.aten。草稿通过 registry.load 加载自 tmp_path 中重新建立的 stage。覆盖内容：

- **记录到的调用结构**：
  - pad：i64，含 ±2^63 边界值，填充 -100.0；
  - slice_backward：4 种记录布局，包括 end=INT64_MAX；
  - repeat_interleave：[1,2,128,64]，repeats=7，dim=-3；
  - tril：b8 128×128；
  - D8：前向与反向；
  - D18：[2,3,10,8] → [20,16]，scale 2.0。
- **边界情形**：空维、负 dim 与负 start、step>1、slice 为空、填充值边界（u8、i16、i64 的 2^53 与 2^63）、bool 填充、各种 pad 模式字符串、output_size、repeats=0、批量 tril 与超出范围的 diagonal。
- **随机网格**：160 组 avg_pool2d 前向与反向（覆盖 ceil、count_include_pad、正负 divisor_override、stride=[]、4 种 dtype），24 组无 scale 的 upsample，以及有 scale 的前向与反向。
- **拒绝边界**：
  - 结果类别：invalid_input、unsupported；
  - profile 唯一性：LAYOUT 记录到的调用恰好只落在一个自动 profile 上；avg 调用只落在任务选择 profile 上；INTERP 只有 ATen profile 是自动 profile。

## 5 命令与结果

```bash
export HOME=/data1/tzh/kernel-analyzer/.cache XDG_CACHE_HOME=/data1/tzh/kernel-analyzer/.cache TMPDIR=/data1/tzh/kernel-analyzer/.cache/tmp TRITON_CACHE_DIR=/data1/tzh/kernel-analyzer/.cache/triton PYTHONPATH=/data1/tzh/kernel-analyzer/2_tool/src
S=/data1/tzh/kernel-analyzer/.cache/spec_a2_review/C
cd 5_specification && /data1/tzh/envs/liger/bin/python tools/check_contracts.py --contracts $S/contracts --adapters $S/adapters --only <ID>
```

| 命令 | 退出码 | 结果 |
|---|---|---|
| check_contracts LAYOUT | 0 | 例子 73/73，变异 25/25，consistent |
| check_contracts POOL | 0 | 例子 40/40，变异 20/20，consistent |
| check_contracts INTERP | 0 | 例子 20/20，变异 10/10，consistent |
| `pytest tests/test_layout_pool_interp_aten.py tests/test_layout_pool_interp_torch.py`（作者测试） | — | 39 passed |
| `pytest tests/test_layout_pool_interp_a2_review.py`（本复核） | — | 317 passed，5 xfailed |

5 个 xfailed 是第 3 节两条问题的见证，都设为 strict：问题修复后它们会转为 XPASS 并报失败，届时需要去掉 xfail 标记。

## 6 修复后复查（2026-10-21）

作者修复了 LAYOUT-1 和 INTERP-1，并在 POOL 中补充了备注所指的那句话。我逐行看了改动，并在新建的 stage `.cache/spec_a2_review/C2` 上重跑了检查和测试。

| 文件 | 新 sha256 |
|---|---|
| LAYOUT_path_a_A2_20261020.yaml | 873c6805a38e31d1c209fb64cde1f2feaf59d94bf3302cd61e93866c54906ff2 |
| LAYOUT_adapter_path_a_A2_20261020.py | 35f492ba9811bea08c09edfe6604f3c336a5fa2b6b60d043aac9827a699f2e97 |
| POOL_path_a_A2_20261020.yaml | 4ec57fc6e02ea252589b560a4b2538b4bdbb6691b1a02087dcbba1d00f32bf10 |
| POOL_adapter_path_a_A2_20261020.py | 3a0a9591039bf7b290d44ba12f1884027d057d1d727e81d09bb90f5f12afdaa6（未变） |
| INTERP_path_a_A2_20261020.yaml | a5aec30e885fad2339c2249137e193486e2e7886e7f367220ea27210b8b4e05b |
| INTERP_adapter_path_a_A2_20261020.py | c61467dea52cf56c707023fed16e73f6f78589230f71de96e2e0922b3464231f |

**LAYOUT-1 已关闭**
- `_scalar` 和 `_exact_scalar` 都捕获了 `OverflowError`，并转为 unsupported。
- 新增例子 E74–E76：i64、b8、f32 上的 ±inf 填充值都得到 unsupported。
- 改动只限于这一处。

**INTERP-1 已关闭**
- 无 scale 时：取 q = isize/osize，只有 q 和每个 o·q 在 binary32 中都精确时才给出值，否则 unsupported。
- 这个条件是充分的：q 在 binary32 中精确时，fl64(isize/osize) = q，转成 fl32 也是 q；fl32(o)·q 精确，所以截断结果等于 floor(o·q)。因此给出的值不会与源码矛盾。
- IN.ATEN_SCALES 已列出三种备选：精确规则、float32 规则、任务选择。取舍理由不依赖任何实现。
- 例子：
  - E12 改为精确的尺寸对，手算结果为 [[1,3],[5,7]]；
  - E21、E22、E23 的比值分别为 2/82、3/7、2/3，都不精确，得到 unsupported，结果正确。
- 变异 IN.M11 被拒绝。
- D18 的 10/20 和 8/16 都是 1/2，精确，D18 仍然可以求值。
- 代价：无 scale 时，3/7、2/3 这类常见比值现在被拒绝，属于覆盖范围的限制。

**POOL 备注已采纳**
- PA.ATEN 补写了 functional 页面把 kernel_size 称为 "size of the pooling region"，这是 R1 的文本依据。
- 新来源条目 T_F_AVG_POOL2D.KERNEL_SIZE 与快照第 9 行一致。
- 任务选择仍然保留。

**作者对本复核测试文件的修改**
- 去掉了已修复问题的 strict xfail。
- 新增谓词 `f32_exact_no_scale`：用 numpy float32 独立判断精确性，没有调用适配器。
- 对精确的尺寸对，测试同时断言精确规则的下标等于源码 float32 公式的下标。
- 我检查过，这些修改是独立实现，逻辑正确。

**我追加的测试**
- 新增两个无 scale 的反向用例：[1,2,3,2]→[6,8] 和 [2,1,12,8]→[3,2]。修复后，原有的两个无 scale 反向用例都变成了 unsupported，这两个新用例补上了"给出值"的覆盖。
- 新增 `test_exact_predicate_implies_source_agreement`：穷举 isize ≤ 64、osize ≤ 128，凡是被准入的尺寸对，精确下标都等于源码下标。

**结果**
- check_contracts（fresh stage）：
  - LAYOUT：退出码 0，例子 76/76，变异 25/25，consistent；
  - POOL：退出码 0，例子 40/40，变异 20/20，consistent；
  - INTERP：退出码 0，例子 23/23，变异 11/11，consistent。
- pytest：
  - `test_layout_pool_interp_a2_review.py`：325 passed；
  - 作者测试（aten + torch）：39 passed。

**最终结论**：上表中的 LAYOUT、POOL、INTERP 三对文件均为 **CONFIRMED**，可以晋升到顶层并复制到 validated/。

## 7 LAYOUT 第二次复查（2026-10-21，晋升前发现 narrow.Tensor 问题）

POOL 和 INTERP 已按第 6 节确认的哈希晋升。LAYOUT 在晋升检查时被暂缓：`tests/test_layout_profile_domains.py` 显示，aten.narrow 的 start 传入张量时（属于 narrow.Tensor 重载），结果类别是 invalid_input，而应为 unsupported。

作者的修复和复查对象：

| 文件 | sha256 |
|---|---|
| LAYOUT_path_a_A2_20261020.yaml | 5b98080b739a396f2242c454da55003236a5e26b135f0e2db12110a611a8dbe8 |
| LAYOUT_adapter_path_a_A2_20261020.py | 96640c39e1ab2c874daa5afe9672f2a1434f41e1ad566eb67c0db7516af9bcad |

**改动内容**
- 解码器只在张量参数（a、src、grad_output、tensors）上接受张量记录。
- 其他参数（尺寸、维度、下标、范围、重复次数、标量，包括嵌套在列表里的）一旦出现张量记录，就判为 unsupported，视为另一个重载。
- 新增例子 LY.E77（narrow.Tensor）和 LY.E78（repeat_interleave.self_Tensor）。

**核对**
- 用 inspect 枚举了 covers 中全部 28 个 ATen 算子的参数名。所有张量参数都在 {a, src, grad_output, tensors} 之内，所以没有合法的张量参数会被拒绝。
- 新增 `test_tensor_parameters_still_accepted`，在 i64、f32、b8 三种 dtype 上覆盖了 cat 和 stack 的张量列表、select_scatter 和 slice_scatter 的 src、slice_backward 的 grad_output，以及 constant_pad_nd。
- 新增 `test_tensor_for_non_tensor_parameter_is_unsupported`，覆盖 7 种情形：narrow、select、slice 的 dim、view 尺寸列表中嵌套的张量、pad 列表中嵌套的张量、张量形式的 repeats、tril 的 diagonal。
- 作者对 `test_layout_profile_domains.py` 的两处修改都正确：
  - 改为 i64 的 clone 在 real profile 下给出值 [1]，这与 LY.EXACT_DATA 一致；
  - 增加了可选的 `LAYOUT_ADAPTER` 环境变量。
- 对照验证：
  - 该测试指向上一版 adapter（C2）时失败 1 项，即 `test_selected_functional_schemas_reject_other_overloads`；
  - 指向本版时全部通过。
  - 这说明测试确实用到了暂存的 adapter。

**结果**（fresh stage `.cache/spec_a2_review/C3`）
- check_contracts LAYOUT：退出码 0，例子 78/78，变异 25/25，consistent。
- pytest：
  - `test_layout_pool_interp_a2_review.py`：335 passed；
  - 作者测试（aten + torch）：39 passed；
  - `LAYOUT_ADAPTER=<C3>/adapters/LAYOUT.py`，运行 `tests/test_layout_profile_domains.py`：5 passed，20 个 subtests 全部通过。

**LAYOUT 最终结论：CONFIRMED**。确认对象为上表中的文件对，可以晋升并复制到 validated/。
