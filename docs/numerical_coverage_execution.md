# 新主线的覆盖与执行入口

这一入口扩展已有仓库，不修改统计公式、历史阈值或原始结果。全部保存任务的去重清单、
计算分类和跨家族执行顺序见[全部已观测 kernel 清单](observed_kernel_catalog.md)。
旧 T1–T4 及历史重算入口继续保留。

本入口承担[主线四目标](current_mainline.md)中的框架自动化和真实 Triton 接入。
它不是独立的“多找正例”任务：不同家族复用同一分析，未知/阴性保留，覆盖进度
与偏差机制、统计保证、训练后果分别验收。大量同族小差异不能代替强机制或强训练
结果；当前 Q-only 固定集合判断也不等于已经完成统计理论。2026-09-07 的口径校准
不恢复实验队列，以下命令是执行说明而非当前正在运行的声明。

## Triton 优先的全量前沿

观测清单包含大量重复的模型位置和发布目录。为了避免把同一个问题按模型或层号
重复计数，下面的命令按发布包、任务、输出指针和阶段去重，再按算子族和实际
implementation kind 汇总。它只使用执行身份和 support status，不读取任何数值结果
来挑选下一项，因此产物是覆盖计划而不是 bias 结论：

```bash
PYTHONPATH=src:. python scripts/build_triton_coverage_frontier.py \
  --catalog results/property/numerical_coverage_v1/observed_kernel_catalog_v1.json.gz \
  --output results/property/numerical_coverage_v1/triton_coverage_frontier_v1.json
```

该前沿同时列出每个 Triton 家族的已完成、可直接测量、只有参考但缺少训练接入、
以及仍需补参考或参数映射的位置，并自动给出一个下一代表。出现
`READY_FOR_MEASUREMENT` 并不等于已经完成；只有真实三阶段采集和统一复算成功后才
会进入完成计数。运行失败保留在原家族记录中，不能静默换成另一个位置。

2026-09-11 的首轮七族冻结执行及失败原因由
`results/property/numerical_coverage_v1/family_first_campaigns_v2/summary_v3.json`
和同目录的 Markdown 汇总自动生成。该轮没有有效三阶段测量；这表示参考/执行接入
仍需修复，不表示这些家族是数值阴性。

自动化终点是生成可复算的诊断与候选修改证据，而不是让程序自行决定科研修改和长训练
配置。研究者可以据此审核修改并安排训练验证；系统负责冻结该决定、执行已审核配置、
保留失败并复算报告。本文后续所说的“诊断指导修改尚未完成”，指缺少相应实验结果，
不指必须消除这一人工决策步骤。

```bash
PYTHONPATH=src:. python scripts/run_training_numerical_analysis.py coverage freeze \
  --output results/property/numerical_coverage_v1/example \
  --release results/coverage/runtime_releases/deepseek8b_seq128_r1 \
  --architecture deepseek8 \
  --model /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --input-bank results/coverage/deepseek8b_seq128_input_bank.json \
  --carrier-registry results/property/bias_formation/hotspot_search/multishape_backward_carriers.json \
  --registry-model deepseek8b --sequence-length 128

PYTHONPATH=src:. python scripts/run_training_numerical_analysis.py coverage run \
  --output results/property/numerical_coverage_v1/example --device cuda:0 --limit 3

PYTHONPATH=src:. python scripts/run_training_numerical_analysis.py coverage report \
  --output results/property/numerical_coverage_v1/example
```

统一入口会为委托脚本自动补上仓库的 `src/` 与根目录导入路径；命令示例保留
`PYTHONPATH` 只是为了兼容直接调用旧脚本的环境，不再是运行入口的隐含前提。

## 已实现

- 从实际 release 的完整端点清单建立分母，不按数值结果筛选。
- 复用已有图上的参数映射，同时检查映射中的计算符号和 AOT 输出身份。
- 有明确参考与参数映射的计算交给现有三阶段采集器，统一调用
  `training_numerical_analysis.analyze_artifact`，保留实际参数写入和原坐标能量。
- 没有按 Triton、ATen 或外部库名称决定测量结果。
- 未支持项、缺参考项、执行失败和未开始项分别保留。
- 协议、输入银行、任务清单和关键代码保存摘要；执行前检查变化。
- 结果与日志不覆盖；中断任务不能静默当作成功，也不能自动跳到替代案例。

## 仍未完成

清单中的端点不等于独立算子家族，也不等于全部已验证。当前同时保留 AOT 参考图
端点替换、外部矩阵乘法的相同输入参考，以及源码核对的平方和归约、RMSNorm
backward、softmax backward 参考；这几种比较不能混称为单 kernel 来源证据。其他家族仍需明确适配，
不能借名字猜测语义。参数映射已扩展到下文列出的完整 backward 清单，但静态映射
不等于全部位置可运行，更不等于已测量。训练参数可达性和实际执行身份仍须动态确认。

本轮数据是旧结构信息上的新采集，不称为全新、未见实现验证。
固定集合 1% update RMS 判断保留；方向分析不充当全空间上界，随机总体
等价性保证没有因此启用。此入口的程序完成也不等于整份科研计划完成。

后续仍需扩大参数映射和家族参考、统一同数据基线与测量成本、冻结新的家族
验证，以及完成诊断指导修改的训练价值比较。已有 Liger 实验不能代替这些项目。

## 2026-09-06 覆盖规划记录

[四模型三长度清单](../results/property/numerical_coverage_v1/four_model_grid/grid.json)
包含 94,872 条 release 端点记录：723 条结构条件具备、59,496 条缺少当前
参数映射、34,653 条执行边界尚未确认。长度和循环展开会重复计算，因此这些
数字绝不是 94,872 种 kernel 或 723 个已完成实验。

723 条候选按现有执行记录分为 402 条 Triton、321 条外部计算；按已有语义
分类覆盖 CE backward、loss head backward、normalization backward、attention
状态传递、attention projection 和 state-space recurrent 六类。这是结构分类，
不是六类均已获得有效新结果；外部计算也不能仅凭这一标签归为某个具体 CUDA 库。

首次 DeepSeek 尝试因沙箱不能初始化 CUDA 而失败；失败记录保留在
`deepseek128_registry`，可用 GPU 环境的新尝试位于 `deepseek128_execution_retry`。
Phi 首次整图编译因位置编码的数据相关分支失败；新协议明确允许分段编译，
位于 `phi64_segmented_retry`，仍检查原实现身份。以各目录的 `status.json`
和原始结果为完成依据；只有启动日志时只能称为正在执行或未完成。

这些是执行接入试跑，不是方法冻结后的全新盲测。后续修改执行器会触发已有协议
的代码摘要检查；必须另建记录，不能覆盖原协议后宣称原结果从未改变。

## 扩大全部 backward 输出的参数映射

`scripts/build_release_parameter_mappings.py` 复用已有 AOT 数据流追踪，
不再只给已有代表位置绑定参数。DeepSeek seq128、Phi seq64、Qwen seq64、
Mamba seq64 分别得到 1,231、642、930、3,607 条静态映射。
逐项比较旧代表映射的结果保存在 `expanded_mapping/*_verification.json`。
这没有补造动态参数可达性，也没有解决所有 forward 输出的映射。

完整原始量对照见
[21 行三阶段记录](../results/property/numerical_coverage_v1/baselines_existing_svg/baselines.csv)与
[对应图](../results/property/numerical_coverage_v1/baselines_existing_svg/profiles.svg)。
无完整 Gram 时不将摘要当成全空间 mean；Liger 缺失的 local 原坐标量保留 N/A。

新协议可显式选择 `--parallel-measurement`。它仅并行三个独立摘要，保持每个
摘要内部的坐标映射、分块顺序、float64 累加与最终 float32 存储不变。
[测试与 CPU 时间记录](../results/property/numerical_coverage_v1/parallel_measurement_benchmark.json)
显示本次样本逐位一致；该局部加速不是模型训练吞吐收益。旧任务不热更新。

## 参考计算范围与自动接入

`AOT_REPLAY` 是参考图端点替换，不保证被比较的单个计算使用相同输入。
它可能同时包含上游差异。现在清单保存 `reference_scope`，新原始结果保存
`reference_comparison_scope`；不能仅由端点映射将差异归因为该 kernel。

外部 `mm/bmm/addmm` 可使用 `EXTERNAL_FP32_RECOMPUTE`：读取 candidate
实际调用前的输入副本，在 FP32 中执行同一数学计算并转换到 candidate 输出类型。
这不是宣称 FP32 是绝对真值，也不要求已有 AOT 参考切点。
观察器在调用前复制输入，避免 `out=` 与输入共享存储时误读调用后的值。
其他运算不会偷偷使用这一参考，仍显示需要参考接入。

家族选择器支持 `--reference-method EXTERNAL_FP32_RECOMPUTE`，依据已有
执行顺序和运算支持范围选取，不读取数值结果。新比较使用独立 case ID 与新协议。

## 同名源码与整批验收

跨模型源码扫描允许记录多个同名定义，但绑定时如果同名定义的完整函数摘要不同，
必须拒绝，而不是让后一个覆盖前一个。应针对明确的 release 建立绑定；这个检查
与数值结果无关，不改变统计阈值。

源码核对家族的整批结果使用 `scripts/finalize_numerical_family.py` 复算：
冻结计划中的所有位置保留在分母，核对输入银行和顺序、实际参数与边界、参考范围、
三阶段数据及分析代码版本。结果缺失时返回未完成，不把源代码识别、启动日志或
已经完成的部分状态算作完整实验。该验收本身不证明独立训练效果。

`scripts/summarize_source_reference_coverage.py` 将不同参考家族的扫描按同一源码
和函数定义合并；源码版本或定义集合不一致时拒绝汇总。四模型三长度的当前结果为
45份源码、957条 Triton 定义记录；原三模板扫描中9条具有单一已核对模板参考，948条需要其他
参考；不存在“957种 kernel 全部已测”的结论。这张表不包含外部调用，不覆盖仓库
所有模型；同名或重复出现的定义也不是独立机制。
记录见 `results/property/numerical_coverage_v1/common_input_source_coverage.json`。

加入严格的 SiLU backward 模板后，同一45份源码重新汇总为15条具有参考模板，
942条仍需其他参考，见 `common_input_source_coverage_with_silu.json`。
新增匹配6条定义，不等于6种机制，也不等于已经完成对应位置的 GPU 测量。
新旧扫描并列保留；旧 RMS 全源码扫描与提取定义后的加速扫描结果逐字节一致。

Ministral fused RoPE / position scaling 随后也使用登记的相同输入参考接入同一流程。
高/低位置固定集合写入 RMS 约为 10.88%/8.35%；匹配的 cold/warm/reset-moments
比较为 8.67%/0.45%/10.02%。`build_rotary_family_evidence.py` 同时核对两种位置的
语义源码摘要、目标参数、固定集合分析和 optimizer 条件摘要，再把有范围的证据加入
算子族报告。它不会把该记录增加为新的清单位置，也不会补造总体或 loss 结论。
[详细核验](fused_rotary_position_scaling_audit.md)

## 大参数家族的固定顺序队列

`partition_family_plan.py` 按原计划顺序和每批位置数量拆分，不读取数值结果。
`run_family_plan_queue.py` 复用一份已声明执行设置，每次只改变任务清单与输出路径，
冻结全部任务、原始计划摘要和代码摘要。每项自动采集、保存源码快照、复算并生成
三阶段对照；失败时停止该队列，保留日志，不自动换题或覆盖已有目录。
队列完成只表示其中列出的测量完成，不表示完整研究计划或所有 kernel 支持完成。

DeepSeek seq128 的 SiLU 共36个位置，首项单独执行，其余35项分配到两个
不重叠队列 `deepseek128_silu_queue_gpu2`（18项）与
`deepseek128_silu_queue_gpu3`（17项）。每个位置的声明参数约5033万维，
分批是内存限制下的执行安排，不减少源计划分母。
比较边界和数学公式见 [SiLU 相同输入比较](silu_backward_common_input.md)。

`audit_family_plan_execution.py` 从原始全计划核对各队列和单独任务，检查案例、
参数、参考方法和计划摘要是否一致，拒绝重复计数。未排入、未完成、尚无可核验
结果的记录均留在分母中；同时保留原来没有参考的1195个位置。
审计不从日志文件存在推断进程仍活着，也不把36项入队称为36项完成。
首份中间审计为 `deepseek128_silu_execution_audit_initial.json`，后续另写新文件。

会话中断后的实际进程核查、未完成任务恢复和独立运行记录见
[实验恢复说明](numerical_execution_recovery.md)。原目录仍是历史记录，
当前执行目录以恢复审计为准，不将旧队列和新队列相加计算覆盖数量。

## 采集成本

后续家族采集自动保存 `raw/capture_cost.json`：包含加载、编译、逐状态重放、统计
和结果写入的总耗时、进程 CPU 时间、进程内存峰值及 PyTorch CUDA 分配器的显存
峰值。它不是 kernel 耗时或训练吞吐；进程内存峰值包含该进程此前的初始化，
PyTorch 显存数字也不是整台 GPU 上所有程序的用量。
失败调用保留失败状态，不根据调用返回就判断数值测量有效。

`family-report` 引用该记录及摘要；旧结果没有记录时明确给出 `NOT_RECORDED`，
不根据文件存在时间补造历史性能。新增指标不参与数值等价性或 bias 判断。
家族入口同时拒绝覆盖已有冻结协议，重试必须使用新输出目录。
首项为 `deepseek64_attention_common_input`，覆盖 attention projection 的外部 mm。

完整原始结果已经写完但控制进程未留下状态时，可用
`recover_completed_numerical_capture.py --root <目录>` 核对身份、32 个状态及
三阶段原坐标量后恢复分析。它不启动 GPU，也不补造进程退出码；旧协议未冻结的
分析依赖明确标为当前重新分析。之后用 `verify_numerical_coverage.py` 复算。

## 源码核对后的 Triton 归约家族

统一入口增加 `row-reference` 和 `row-capture`，前者调用
`build_row_reduction_contracts.py`，后者调用 `run_row_reduction_capture.py`。
复用原三阶段采集、参数实际写入和统计分析，不再由 Gemma 案例名选择参考公式。

当前识别范围是连续行平方和的一个明确 Inductor 模板：自动读取行数、列数，
核对地址、掩码、乘法、归约与输出写入，并检查输出不被后续代码复用。
这证明模板在声明条件下对应平方和，不证明舍入 bias 非零。
模板之外的实现标为 `NOT_THIS_REFERENCE_FAMILY`，不被视为阴性或自动改成别的公式。

对原 Gemma 执行源码的自动检查覆盖 69 个 Triton 定义，其中 1 个符合模板。
结果在 `row_reduction_contracts_gemma_bound.json`；依据实际任务的 symbol 与
输出指针生成 `family_plans/gemma_row_family.json`，没有读取数值结果来挑选。
运行时再次核对实际加载函数的 AST，参考使用调用前输入副本。

预先列出的数学等价变体为 FP32 常规顺序、FP32 反转 feature 顺序、FP64 计算。
它们不是预先确认的修复；FP64 也不是绝对真值。当前先以原 FP32 参考验证
新家族入口和原案例范围一致，输出在 `gemma_row_family/`。不同变体需要独立
运行记录，不能覆盖旧结果。参数是否全部参与 backward 也作为明确选项记录，
避免改变训练参数范围后仍声称测量原实现。

当前 `gemma_row_family` 已完成 32 状态，统一原坐标分析给出 update RMS
7.678779%，复现原参考范围下的结果。统一入口的
`analyze RAW OUTPUT --protocol PROTOCOL` 使用现有 v2 分析，无须手填结论。

参考范围对照也已完成：同一 DeepSeek seq64 attention projection 位置，
AOT 端点替换 RMS 为 34.7162%，相同输入 FP32 重算为 0.0004777564%。
`compare_numerical_reference_scopes.py` 检查声明配置并生成
`deepseek64_reference_scope_comparison.json`。该结果要求限制 AOT 端点替换的
单 kernel 归因，但不单独证明所有差别都来自上游，也不推翻其他具有真实
相同输入参考的案例。

## 批量采集与参考来源索引

冻结时可声明 `--batch-size N`，运行时的 `--limit` 仍表示本次最多采集多少个位置。
多个位置共享模型加载、参考图计算和只观察不修改的重复计算；真正替换仍逐个进行，
不会同时修改多个位置，也不会把多个位置的数值当作一个统计总体。
共用同一 AOT 参考切点的任务自动拆开，避免参考值绑定歧义。
每个位置保留独立原始结果、状态和分析记录，批次目录保留共同日志与任务清单。
每个位置的 raw 链接指向共同采集目录，索引按实际文件去重，不增加样本数。

`build_mainline_roles_v2.py` 现在为当前原始记录追加 `reference_scope_audit`。
优先读取显式记录；否则只连接同一 case、参数与运行位置的计划，或成功执行命令
指向的计划。不通过案例名称猜参考方法，失败执行记录不能证明后来原始数据的来源。
当前索引 `case_roles_reference_scope_execution_linked_20260907.json` 保留48条记录、
34个case ID；这是历史与当前记录的索引，不是34个统一验证通过的案例。

批量路径的首个运行 `deepseek64_common_input_batch2` 已完成两个位置各32状态。
复测位置的所有原坐标统计与单独运行一致，见
`deepseek64_batch_reuse_verification.json`；另一位置的 RMS 为0.3335231%。
这验证了此次共享采集没有改变已有位置的数值，不宣称任意模型的批量执行都已验证。
