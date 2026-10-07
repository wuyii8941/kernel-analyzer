# 本质错误一轮·2a 中期记录（2026-10-07，不等待批复）

依据：第二阶段任务书（2a A–F、第一阶段补充「第二类」）；执行协议 `docs/protocol_essential_bugs_phase2_20261007.md`（v2；第 9 节
记录偏离）。第一阶段的结果与分类冻结在 `results/essential/phase1/`，未改。规格仍是 `specs/phase1`（v0.4，入库后未改）。检测器从
`eval-stage-a-20261006`（第一阶段成绩所对应）升到 `detector-v2.2`（A1）。本记录之后直接进入 2b（任务书：2a 完成时提交中期记录，
不等待批复）。

## 0. 状态总览（协议第 7 节的状态）

| 项 | 内容 | 状态 | 依据 |
|---|---|---|---|
| S1 | 前后向自一致性（jvp 的 ⟨v, J u⟩ 对 VJP 的 ⟨B(v), u⟩），eager 候选 | 仅部分方法可执行（前向模式只对 eager、且只对有前向 AD 的算子） | `results/essential/phase2a/s1_jvp_vjp.json` |
| S2 | 补齐 W4 的性质 | 已执行且可裁决 | `results/essential/phase1_supplement/s2_properties.json` |
| S3 | 补齐 W6 的边界输入（池化 108、index/scatter 114 个条件） | 已执行且可裁决 | `results/essential/phase1_supplement/s3/` |
| S4 | Inductor 收到的 label_smoothing 标量 | 已执行且可裁决（并入 A3） | `results/essential/phase2a/a3_constants.json` |
| S5 | W8 的规格书写时间 | 未建立（执行方没有该数据，需审阅方提供） | — |
| A1 | 输出来源绑定（检测器 2.2） | 已执行且可裁决 | `tests/test_output_binding.py`；`docs/detector_changelog.md`；`results/essential/phase2a/a1_recapture.json` |
| A2 | 分类层边界保护 | 已执行且可裁决 | `scripts/essential/tests/test_classify_guards.py`；`results/essential/phase2a/reclassified/` |
| A3 | FR 结论三个字段 | 已执行且可裁决 | 同上；`a3_constants.json` |
| B | W5 测试判据审查（v2.10.0）与 B020 上的 gradcheck | 已执行且可裁决 | `results/essential/phase2a/w5_review.json` |
| C | 轻量方法对照 | 已执行且可裁决 | `results/essential/phase2a/c_methods.json` |
| D | 可复用触发检查 D1–D4 | 已执行且可裁决（实现与单元测试；在 2b 的适用家族中使用） | `scripts/essential/triggers.py`；`scripts/essential/tests/test_triggers.py` |
| E | 报告措辞修正 E1–E6 | 已执行且可裁决 | `docs/essential_bugs_phase1_interim_20261007.md` 第 9 节 |
| F | 上游草稿（提交由用户决定） | 已执行（草稿）；提交未运行 | `bugs/upstream_drafts/` |

S1–S3 是**事后补充**，与第一阶段的预注册成绩分开报告；第一阶段的 28/50（P 组）等数字不变。

## 1. 第一阶段补充（事后补充）

**S1（jvp 对 VJP）。** 单位是（条件 × seed × 设备）。交叉熵：748 相容、74 无定义（非有限值）。池化：无并列 1643 相容、390 输入
非有限、6 不可执行（B021 的两个条件：CPU 不支持前向模式、CUDA 在 gather 中触发设备端断言——越界下标，见 B021）；有并列 289 相容。
index/scatter：无并列 548 相容、778 不适用（`index_reduce` 等没有前向 AD）、**123 不相容**；有并列 96 相容、230 不适用、**31 不相容**。
154 个不相容全部归到两个已知原因：`scatter_reduce` prod 在零值处的 JVP（115，已知 #196702）与 amax/amin（39，B020：VJP 一侧错）。
S1 本身不能指出哪一侧错；B020 一侧由 F 与 gradcheck 确定（第 4 节）。

**S2（补齐的性质）。** 全部受裁决候选（含 nightly）零违反：交叉熵改变被忽略行 / 概率为零处的 logits，不改变 loss（τ 下 0/552；
逐位 0/552）、不改变其余行梯度（0/390），被忽略行梯度为零（0/535）；池化通道置换等变（0/2328，nightly Inductor 0/2308——20 个 seed 级
运行因 B022 的设备端断言缺失）与 avg pool 线性（0/1152）；index/scatter 线性归约的线性与可加性（0/672）。

**S3（边界输入）。** 222 个补充条件 × 3 seed × 6 个候选，E / F / P / FR 同第一阶段（`classify.py --out .../s3`）。逐项裁决：

| 现象 | 候选 | 条件 | 裁决 |
|---|---|---|---|
| CUDA eager 的 avg_pool1d 反向在「转置存储」（通道步长 1，即 channels-last 形态）且 padding > 0 时梯度错（约 1750/2054 个元素） | 2.10 eager CUDA（nightly 20260907 正确；CPU 正确） | s3pool004 / 010 / 016（三种数值） | **召回（已知）**：#188344，由 #188345 于 2026-06-27 修复；最小复现为 `avg_pool2d` 的 channels_last 输入 + padding（`.cache/tmp/s3check/` 中的探针，不入库） |
| Inductor 的 max_pool3d 在空间维存储顺序交换时返回的下标不指向最大值（数值正确），反向因此路由错 | 2.10 与 nightly 20260907 的 Inductor | s3pool094 / 100 / 106 | **召回（已知）**：#197434，由 #197596 于 2026-09-22 修复（晚于可运行的 nightly）；预注册性质「下标在 argmax 集合内」9/162 同时发现 |
| Inductor avg_pool 反向按 R1 读法 | 2.10 与 nightly Inductor | 9 个条件 | 召回（已知，#198119 / B015），与第一阶段相同 |
| index/scatter 前向在 1e30 级与跨尺度数值上有 1–3 个元素超出 τ₃₂，含 2 个「共有偏离」 | 全部 float32 候选（eager CPU / CUDA、Inductor、nightly） | index_add、index_reduce mean、scatter_reduce sum | **数值（抵消），不是语义错误**：每个超出 τ₃₂ 的元素都含正负两号的项，且 \|K − f\| 不超过 float32 递归求和的先验误差界 γ_k·Σ\|项\|（最大比值 0.09；`s3/numeric_bound.json`，`scripts/essential/s3_numeric_bound.py`）。τ₃₂ 相对 \|f\| 而不是相对 Σ\|项\|，抵消严重时任何 float32 实现都达不到 |
| 顺序不变性性质违反（8–11 / 342） | 同上 | 同一批大数值条件 | 同上的抵消现象（两个顺序各自在误差界内；未逐元素另核） |
| Inductor `index_reduce` mean 在全 1 输入上有位置得 0.99999994 | 2.10 与 nightly Inductor | 2d、dim=1 的 9 个条件（计数 55 等） | 数值：mean 以乘倒数实现，55·fl(1/55) = 1 − 2⁻²⁴；属接口 / 常量类，不是计数错误 |

S3 没有新的本质错误；两处召回都是 2.10 中已在上游修复的错误，由补充的存储布局维度触发（第一阶段只有连续布局）。FR 在 S3 上的池化
反向给出 6 个「语义差异」，落在上面两类召回的条件上。

**S4。** 交叉熵 ε ≠ 0 的 Inductor 条件 103 个（float32 与 bf16 各 103）全部「已核实」：生成的 kernel 中可指认 float32(1 − ε) 与
float32(ε/C) 的编译期常数（协议 v2 偏离 2 的判定）。

**S5。** 未知（需审阅方提供）。

## 2. A 工具修复

**A1（检测器 2.2，`detector-v2.2`）。** `TritonLaunchRecorder` 在捕获窗口内持有每次启动的张量存储，并记录 `untyped_storage()._cdata`；
`check.run` 只在输出的存储实例属于记录的写入实例时配对，实例不同记为「不由 Triton 写出」，无法确认记为「未建立」。三个回归情形
（大小不同 / 大小相同内容不同 / 大小相同内容相同）在 keep 开关两种取值下都通过（7 个测试）。定向重捕获 24 个受影响条件：池化 6 个
IndexError 全部消失（grad 由 ATen 写出，正确记为「不由 Triton 写出」）；pool394 / pool438 的「最后一次 Triton 写之后被修改」是
地址复用的假象，已消失；index 前向在部分 seed 上 K_R 未建立是求值器既有的限制（与绑定无关，记入局限）。第一阶段的结论不变
（停止条件 2 不触发）。

**A2 与重新分类。** 用 A2 / A3 的分类层重新分类第一阶段全部数据（`results/essential/phase2a/reclassified/`）：E / F / P 的类别**逐条
不变**（1,652 个条件 × 全部候选 × 前后向）；变化只在 FR：`ok_elements = 0` 的输出由原来的「无语义差异」（空集上的一致）改为「未建立
（工具）」——池化 out 2 个、index grad_self 5 个、grad_source 40 个；交叉熵与池化中「无规格值」的一部分改记为「未建立（工具）」
（交叉熵每个候选的 loss 与 grad 各 21 个，池化 grad 92 个）。

**A3。** 第一阶段「CE 的 bf16 Inductor kernel 489 个单位全部没有语义差异」按三个字段陈述：545 个输出单位中，282 个 e_sem 区间含零；
207 个 e_sem 区间不含零、但 |中点| 都低于行动阈值 2⁻²⁰(1 + |f|)，且这 207 个的接口 / 常量来源全部已核实（ε 相关的舍入常数）；
另 56 个不判（42 个未建立、14 个无规格值）。全部单位都不超过行动阈值。全体 FR 条件中，接口 / 常量来源已核实 362 / 404。

## 3. B W5 测试判据审查（v2.10.0）

`scatter_reduce` / `index_reduce`：OpInfo 样例是连续随机值（`make_tensor`），**到不了「include_self=False 且被排除值等于结果」与参与者
并列**；Inductor 的 OpInfo 测试以 eager 为参照（B020 是 eager 自身错误，差分看不到）；`inductor_one_sample['cuda']` 让 amax / amin / mean
只跑第一个样例（未给原因）。max_pool：OpInfo 样例核 3、padding ≤ 1、长度 3 与 6，**任何窗口都至少采到一个输入**；
`test_max_pool1d_corner_cases` 恰好检查了 B021 的几何，但只看 `return_indices=False` 的前向值（−∞）。B020 的 50 个条件上运行 gradcheck
（float64、CPU、默认容差）：50 / 50 个条件至少一个 seed 失败（无并列 32/49、有并列 82/101 个 seed）。结论：两个错误没被发现，是因为
样例的值域与几何到不了触发条件，而 gradcheck 本身能看到 B020。被跳过检查或以 eager 为参照的算子已进入 G8 的发现驱动队列。

## 4. C 轻量方法对照（事后分析）

| 对象 | gradcheck | jvp 对 VJP | 预注册 P | F | FR |
|---|---|---|---|---|---|
| B020（50 个条件） | 50/50 | 17/17（只 scatter_reduce；33 个 index_reduce 无前向 AD） | 28/50 | 50/50 | 48/50 |
| B021（2） | 2/2 | 报错（越界下标；看作错误而不是不相容） | 2/2 | 2/2 | 不适用（CPU eager） |
| Inductor avg_pool 反向（22） | 22/22（对编译图） | 不适用 | 22/22（伴随） | 22/22 | 22/22 |
| B016（3 种写法） | 1/3（前向错而导数对的「count」通过） | 不适用 | 3/3 | 3/3 | 3/3 |
| HF 梯度累加（4.45.2 / 4.57.3） | 看不到（前向层面的归一化契约；反向是错误目标的精确导数） | 看不到 | 4.45.2 是 / 4.57.3 否 | 两版都是 | 不适用 |

人工：gradcheck、jvp 对 VJP、P 不需要手写 f；F 需要规格；FR 需要规格与 TTIR（只对 Triton 候选）。

**决定规则（协议第 4 节）的结果：** 加入 D 类检查后，轻量方法（gradcheck）在全部**反向**错误上与 F 效果相同；F 的独有价值落在
**前向**共有错误（B016「count」、HF 4.57.3 计数），以及归因（e_sem 对 e_num）与陌生组合的参照复用。论文侧重据此调整；2b 内容不变。

## 5. D 触发检查

`scripts/essential/triggers.py`：D1 极值归约中被排除值等于结果时 ∂out/∂excluded = 0、源梯度不受其扰动、梯度总量等于上游；
D2 无贡献者窗口的下标范围与写入位置（隔离子进程、`MALLOC_CHECK_=3`、邻近通道 / 样本梯度为零）；D3 哨兵下标的反向处理；D4 整数 /
小值域输入保证相等与并列出现。单元测试 6 个通过（对照独立写的参考 autograd Function）。D1–D4 写入 2b 预注册清单（协议 3.2），在
归约、gather/layout 与注意力掩码等适用家族中使用。

## 6. E 措辞修正与 F 上游草稿

E1–E6 已写入第一阶段中期报告第 9 节（结果与分类未改）。上游草稿（`bugs/upstream_drafts/`）：B020、B021（优先；隔离复现与越界证据，
说明参数满足文档前提）、B022（合并 −∞ 并列返回 −1；2.10 正确、nightly 回归）、B015 对 #198119 的补充评论、MaxPool1d 文档公式的文档
issue，各附四项检索记录。提交由用户决定。

## 7. 需要用户或审阅方的事项

1. 上游提交：B020、B021、B022、B015 的评论、文档 issue（草稿已备好）。
2. 2b 新家族的独立规格（优化器、注意力与打包、训练程序层、归一化、嵌入、调度、裁剪、基础算子、MoE）：规格入库前 F 与模式 B 的 FR
   关闭，E、P、状态序列、模式 A 的 FR 照常进行。
3. S5 的规格书写时间。
4. 2b 候选中是否保留非 PyTorch 库（协议 v2 第 0 节已登记的冲突；当前按任务书执行，分开报告，可按要求移除）。
