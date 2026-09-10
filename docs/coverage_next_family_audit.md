# 算子族覆盖与历史接入调查

## 当前搜索优先级（2026-09-08）

目标优先扩大不同算子族的有效测量覆盖，而不是继续增加机制分类，或用同族的
模型、shape、checkpoint、前后向阶段和递推分段重复计数。现有 source-side
asymmetry / response-side rectification 解释在证据适用时保留；不要求每个
新算子族必须产生新机制。

安排新任务时，优先尚未有效测量的算子族，其次补齐已接入但缺少完整测量的族，
最后才扩展已充分测量家族的调用位置。已经启动的任务保留结果，不因新优先级
删除阴性或失败记录。同一家族尽可能复用参考、训练接入和统一统计代码。

`scripts/summarize_operator_families.py` 当前归并为 16 个报告类别：矩阵乘法与
线性层、normalization、softmax、cross-entropy/fused loss、SiLU/门控、
softplus、状态递推、RoPE、独立归约、索引累加、GELU、卷积、embedding lookup
与 Top-k/sort selection、optimizer parameter/moment update，以及 fused causal
attention。这不是已验证的 16 类 bias，也不是穷尽分类；每族
必须分别报告已识别、已接入、有效测量与 bias/训练后果证据。卷积和 GELU
已有历史记录，新的统一接入不能算首次发现。

2026-09-08 的批量更新见 `operator_family_report_v7.json`：当时完整清单有
173,736 个 release-qualified 输出位置，其中 421 个完成有效测量。Softmax
归类 120 个位置，28 个已完成 32-state 三阶段测量、92 个仍待测；embedding
lookup 有 1 个已完成位置。Top-k/sort 是常规 ATen 路径的 24-state 固定集合
三阶段证据，因此作为额外证据进入报告，但不伪造编译图位置，也不签发尚未声明
margin 的等价性结论。

`operator_family_report_v8.json` 保持上述模型图位置清单不变，并把 optimizer-update
测量作为额外固定集合证据接入。该实验在真实 Mamba gradient 上比较 TorchAO 8-bit
AdamW 与标准 AdamW，保留实际生成的 Triton 源码；确认集合参数写入相对 RMS 差约
5.45%，固定集合判断为 `NON_EQUIVALENT`。它不伪造新的模型图位置，也没有训练 loss
结论。自动表为 `operator_family_table_v2.*`。

`operator_family_report_v9.json` 再加入 fused causal attention 固定集合证据。
该实验仅改变 Qwen 第一层的 SDPA backend，Flash 与 math 都以相同 causal 语义运行，
后续层共同使用 math。确认集合 local、gradient、参数写入相对 RMS 差分别约 0.172%、
11.47%、36.91%，判断为 `NON_EQUIVALENT`。它是常规 CUDA candidate 的正式分析，
不是 Triton 位置，也不从单步 loss 差推出训练质量。最新自动表为
`operator_family_table_v3.*`。

2026-09-10 的 v11 没有继续扫描新位置，而是修复证据汇合：三份已经完成的
SiLU/normalization 执行清单共 99 个位置此前没有回写主清单。程序现在按 release 与
task ID 唯一匹配后，得到 520 个已核验位置；SiLU 从 0 修正为 64，normalization 从
228 修正为 263。这个变化不是 99 个新实验，也不增加独立算子问题数。后续优先级使用
`operator_problem_group_depth_v1.*`，不再以位置数排序。

下面保留历史接入过程。历史上按未覆盖位置数量安排的优先级，不再代表当前
按不同算子族安排新搜索的优先级。

汇总新家族测量时，使用 `scripts/join_explicit_output_measurements.py` 将卷积
或 GELU 的三阶段记录接入既有总清单。程序重新检查原始数据、冻结源码快照、
状态与实际写入信息，并要求 release 的 task 文件哈希一致；不能仅按模型名
连接不同版本。`scripts/summarize_operator_families.py` 再按同一别名表归并。
这些保存记录检查不升级为独立 GPU 复现、总体保证或 bias 阳性。

## 历史调查：Mamba 多输出递推计算

本记录是源码接入调查，不是数值测量或 bias 结论。

从 `results/property/numerical_coverage_v1/reference_reach_inventory_v4.json`
的 backward、`NO_CHECKED_REFERENCE_FOR_THIS_OUTPUT` 行统计，出现次数最多的符号是
`triton_poi_fused__unsafe_view_add_bmm_exp_mul_neg_select_softplus_transpose_unsqueeze_69`
（1,449 个输出位置记录）。这些不是 1,449 种不同 kernel 或独立机制。

已检查的源码位置：
`results/coverage/runtime_releases/mamba_seq64_r1/trace/model__1_backward_segment0_executed/output_code.py`。
该定义有 63 个 FP32 输出指针。函数体含展开的乘法、加法、指数和 softplus
计算，不能因名字含 softplus，就套用已经接入的 softplus bias-gradient 参考。
输入包含 BF16 和 FP32，不能称为全 FP32 实验。

下一步接入要求：

1. 核对完整递推关系、每个输出的时间位置及边界条件，而非只检查首尾语句。
2. 用独立的递推数学表达构造参考，明确同精度执行顺序与精度变化的不同实验含义。
3. 同一个 kernel 的多个输出分别绑定真实训练路径；不能把一个输出验证通过推广到全部输出。
4. 复用共同状态、三阶段采集和现有统计代码，不按输出修改判断阈值。

当前接入状态：已经实现完整函数体检查和独立递推参考表达，覆盖全部 63 个输出。
参考表达、源码变更拒绝、多输出选择及带存储偏移的输入布局检查共有 22 项测试通过。
这证明已测试的程序条件，不证明参考是绝对真值或真实训练测量已经有效。

`mamba64_decayed_recurrence_bound_v2.json` 已将 1,449 个位置绑定到训练参数，
并冻结当前源码与参考依赖。`mamba64_decayed_recurrence_batches_v2/` 按原始清单
顺序、每批最多四个位置生成 363 批；全部批次通过输出指针、编号、符号、
重复条目检查，串联后与原始清单完全一致。没有读取数值结果来选取位置。
同一参数对应多个输出时，现有采集流程分别替换每个输出并重放 backward，
不将它们合并成一个测量。

尚未完成此家族的 GPU 运行、真实输入布局验证、有效三阶段测量或机制预测验证。
首批四个位置的 32-state 运行已经通过
`mamba64_decayed_recurrence_batch000_launch_v1` 排队，等待已有 gated-convolution
批量任务成功退出后开始。排队不代表已经测量；其余批次尚未启动。
这些位置目前只能计作源码与训练映射已接入，不能计作已完成有效测量。
此项优先级来自未覆盖位置，而不是已见数值差异的大小。

## 长序列接入检查

对长度 128、256 的 backward segment0 源码进行多输出结构调查后发现，
不能把当前长度 64 的单段接口直接推广：

- 长度 128 的 `...unsqueeze_65` 有 59 个输出、123 个输入，首个时间
  加载偏移为 195072 = 127 × 1536。当前模板按输出数生成 59 × 1536，
  因而会拒绝它。需要明确时间序列总长度与本段起点，不能混为输出数量。
- 长度 128 的 `...unsqueeze_126` 有 60 个输出、124 个输入，第一个输入
  直接按完整状态坐标加载 FP32 值；它不是首段的两个 BF16 向量外积接口。
- 长度 256 也观察到 59/60 输出的分段定义。尚未建立每段完整数学接口，
  不能仅凭输出数认定全部段与长度 128 相同。

以上来自对应 `mamba_seq128_r1` 与 `mamba_seq256_r1` 的
`trace/model__1_backward_segment0_executed/output_code.py`，不是 GPU 数值结果。
下一步应分别验证首段时间偏移与后续段状态输入，复用同一递推数学表达和
统计方法；当前拒绝不表示实现有 bias，也不表示这些位置已经支持。

后续首段检查已经完成：`segmented_recurrence_source.check_first_segment`
对这两个实际首段的完整函数体、参数列表与存储类型检查均通过。
`segmented_recurrence_reference.first_segment` 已实现显式时间窗口解码，
复用原递推表达。现已接入统一 GPU runner，首批已排队；尚未完成有效测量。

用各 release 的 `same_dtype_tasks.json.gz` 和
`expanded_mapping/mamba128.json`、`expanded_mapping/mamba256.json` 调用
现有 `merge_carriers` 核对：每个长度各有 1,416 个首段输出位置，均有
已记录参数映射，每个定义涉及 59 个输出指针。这是映射证据，不是运行时
参数可达性或测量完成证据；两组共 2,832 个位置仍待实际测量验证。

## 后续 FP32 状态分段接入进展

`continued_recurrence_source.check_source` 检查完整函数体、参数顺序与存储
类型。后续段读取 FP32 上一段状态与 log-rate，读取 BF16 时间、bias 与
注入向量，输出 FP32；不能称为全 FP32 实验。

| 序列长度 | 符号后缀 | 下降时间起点 | 输出数 | 已绑定位置 |
|---|---|---:|---:|---:|
| 128 | unsqueeze_126 | 68 | 60 | 1440 |
| 256 | unsqueeze_127 | 196 | 60 | 1440 |
| 256 | unsqueeze_188 | 136 | 60 | 1440 |
| 256 | unsqueeze_249 | 76 | 60 | 1440 |

上述完整定义均通过源码检查，来源是对应 release 的 backward segment0
生成代码与参数映射。这更新了上文初次调查时“尚未建立接口”的状态。
绑定结果是 `mamba128_continued_recurrence_bound_v1.json` 与
`mamba256_continued_recurrence_segment{127,188,249}_bound_v1.json`。
这些是输出位置计数，不是独立机制数或有效测量数。

后续段复用原递推表达、共同状态采集、统计分析和复核入口。
长度 128 按原始清单生成 360 个批次；首批 4 个位置的 32-state 测量
已由 `mamba128_continued_recurrence_batch000_launch_v1` 启动，并登记
`mamba128_continued_recurrence_verification_launch_v1` 自动复核。
启动日志不是实际实现身份或测量成功的证明，仍需等待验证终态。
长度 256 的三个后续段仅完成绑定，尚未启动训练测量。

## 自动发现取代手工符号清单

`scan_continued_recurrence_sources.py` 不要求输入符号或分段起点：先从时间
加载地址提出候选参数，再调用完整源码检查。扫描长度 128 的 159 个定义、
长度 256 的 293 个定义，共确认 6 个后续段，另外发现之前手工清单漏掉的
`unsqueeze_135`（8 个输出，时间起点 8）和 `unsqueeze_266`
（16 个输出，时间起点 16）。扫描没有读取数值结果。

`bind_discovered_recurrences.py` 对每个源码中所有已确认段调用同一个绑定器，
保留未匹配定义与绑定不足项。长度 128 得到 1,624 个位置，长度 256 得到
4,688 个位置；末尾段分别占 184、368 个位置。这更新了上面的初期分段表，
不能把两个版本的重叠位置相加。

`join_recurrence_bindings.py` 将绑定并回全部位置清单，只修改静态接入状态，
不修改任何有效测量标签；已保留的有效测量另行核对原始文件校验值。

首批后续段运行曾被源码身份校验拒绝；另已确认其声明源码设备编号 0、
请求设备 3 不一致。该差异是否解释全部运行时不一致，仍需重试验证。
旧失败与停止队列均保留；显式声明设备 3 后，以相同位置和完整运行时
校验重试。`preflight_recurrence_launch.py` 已接入启动器，后续启动可在
模型加载前拒绝这种声明不一致。该预检不证明实际执行身份或数值结果。
