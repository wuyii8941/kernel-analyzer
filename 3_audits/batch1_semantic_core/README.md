# 第一批：共同语义内核的来源、精度、安装与查询（基线 dsl-v2@3532066）

任务：在严谨、可组合的 DSL 语义上，让陌生算子也能自动完成 bias 分析（模式 A 不要求逐算子手写 reference）。本批扩展现有 Program/Op
与共同语义内核，不另造用户语言。对照的审计基线是 `dsl-v2@3532066`（外部审计 1aee15e 修复版的最后提交）；本批开始时 HEAD 与之相同、
工作树干净，`inputs/` 中没有新的审计包，本批的三类来源问题、精度控制器与安装包问题按任务书的描述在基线上复现（第 7 节）。
依据：根目录 `CURRENT.json`、`CLAUDE.md`、rc3 设计 `1_experiments/dsl_v2/design_rc3/design/02_language_and_guarantees.md` 与
`04_implementation_plan.md`（W2 区域与数值、W4 内存与并发、W6 统计与查询、W7 验收）。

本页状态：实现提交 `d26d5f6`（全量测试 1602 通过）；验收协议与保留集在运行之前冻结于 `da4b64e`；验收 24 项全部成立（第 8 节），记有一处执行偏差（竞争程序误上 GPU，结果不入记录）。

## 1. 来源 / 内存效果：三类问题的修复

实现：`2_tool/src/kernel_analyzer/reference_eval/storage_effects.py`（规则 M1–M5 的唯一实现），由 `check.torch_intermediates`
（声明输入判定、跨启动沿用）、`ttir_eval.evaluate_sequence`（参照内存的跨启动沿用）与 `provenance.producer_records`（逐字节生产者）共用。

| 问题 | 基线上的表现（第 7 节的测试） | 修复依据 |
| --- | --- | --- |
| 跨输入摘要误配 | 读 x 时，只要 x 的存储是某个输入的存储、内容摘要在任一输入的摘要集合里，就算声明输入：`x.data.copy_(w · (1 + 2^-30))`（舍入回 w 的字节）被当成声明输入 x，范围 call 级 | 读到的字节必须等于**该存储自己**在调用前的字节，且落在该存储上声明输入张量的覆盖内（M1、M3）；重复启动的输入核对改为按输入路径的有序摘要 |
| 原地算术后字节未变 | `x.add_(2^-25)`（x ∈ [1, 2)，字节不变）、经 `.data` 的同类写入、两次启动之间 `t.add_(2^-23)`：字节相同就算声明输入 / 沿用参照，范围 call 级 | 写入证据：版本计数器（视图共享计数器；AOTAutograd 对编译代码原地写入的 `increment_version` 记为已宣告的裸指针写）与对齐的生产者追踪（看得见 `.data` 写入）；有证据就不是声明输入、不沿用（M3、M4） |
| 重叠 view 部分写入当成整块常数 | `b.as_strided((n,), (0,)).fill_(0.5)`：元素数 × 元素大小等于存储大小就当整块写，标成精确常数（「producer record: const: aten.fill_.Scalar」），范围 call 级 | 写入只改变它的元素真正覆盖的字节（偏移、跨步、重叠、stride 0），不能枚举时取外包并与原标签合并（M2、M5）；读取只算参照真正从初值读到的元素（`KernelReference.loaded_elements`） |

保留的能力（同一批测试中）：正常复制（`clone` / `contiguous` / `cat` / `where`）、精确常数（`zeros` / `ones` / `full`）、合法别名（偏移
view、转置、重叠读取）、读后原地、编译代码的原地写入后再读、半段初始化而只读初始化部分（基线上是 kernel 级，现为 call 级）。

本批同时发现并修复：

- **D12 生产者追踪干扰被测运行**：在 dispatch mode 下调用一次 `torch.compile` 的函数后，该函数此后每次调用都按 eager 执行（不再有
  Triton 启动）。基线中追踪只在出现混合来源时运行，已有这一问题（编译调用的后续 seed 失去被测的编译 kernel）；本批把追踪改在
  `torch.compiler.set_stance("force_eager")` 下运行（编译代码与缓存不变），没有该 API 时不追踪（T1）。基线上的实例：审计修复的端到端
  调用 `compiled_after_copy` 在第一个 seed 之后失去编译读者，报告把 `y` 同时列为已评估和「不是 Triton 写出」；原测试只看范围，没有
  暴露，现已加断言（基线上失败）。可能受影响的已有记录：基线 6f0ef38 以后经 `check.run`、调用含编译函数且某个 seed 出现混合来源的
  结果；v1.1 回归程序与其绑定不含编译代码（已检索），不受影响。
- **快照不得改变分配器状态**：输入快照最初经设备临时缓冲拷贝，改变了被测运行的缓存分配器状态（一个依赖地址复用的测试因此失去前提）；
  改为设备到主机直接拷贝。该测试的场景同时改用私有内存池，使地址复用不依赖测试次序（断言不变，基线上同样通过）。
- **空泛的「达标」**：没有完整有限元素也没有集合目标的输出原会被精度控制器记为达标；新增类别「无数值包围」。

已知限制（写在模块文档里）：不经过 ATen 与 Triton、既不改变任何字节也不经过版本计数器与 dispatcher 的写入者（外部代码的裸指针写）
看不见；编译代码的追踪不能对齐，此时只有版本计数器证据（`.data` 写入在编译代码里看不见）；`index_put_` 等未列为复制的操作按计算值处理
（保守）；版本计数器覆盖整个存储，另一半被 ATen 改写时整块不沿用（保守）。

## 2. 精度控制器

- 基线：第 2 级的达标比例不高于第 1 级就停止，并写「没有改进」——三级求和反例（n = 128、±2^120 抵消对、和约 2^-100）第 1、2 级都
  0% 达标，于是停在第 2 级。
- 现在：只有参照路径上没有精度相关规则（`intervals.PRECISION_DEPENDENT_CALLS`：用级别 K 的 SumK / DotK、前缀和；为 0 时更高一级的包围
  完全相同）才提前停止；否则一直细化到达标、最高级或预算用完。停止类别（`measure.REFINEMENT_CATEGORIES`）：达标 / 预算耗尽 / 后端限制
  （float64 端点、最高级、无精度相关规则）/ 固有集合宽度 / 无数值包围 / 本级失败。每级报告宽度分位数、达标比例、未达标元素数与精度
  相关规则调用数。分辨率只按完整有限元素计算，集合目标单独计数，宽度取集合上下界（原来输出含集合目标时，未达标的数值元素被一并记为
  「集合目标宽度」）。
- 三级求和反例经统一入口：第 1、2 级 0%，第 3 级 100%，结局「met at level 3」；第 2 级宽度比第 1 级小 35 个数量级。独立答案：精确有理数
  行和在每一级包围内，只有第 3 级在 1/8 ulp 内。

## 3. 安装包

- `measure.class_statistics` 原来按 `Path(__file__).parents[2] / "scripts" / "essential"` 导入 `contract_v3`：安装后的包找不到。
  `contract_v3` 移入包内（`kernel_analyzer/contract_v3.py`），原文件只做转出；硬件预言机缓存默认 `KA_CACHE_DIR` /
  `$XDG_CACHE_HOME/kernel_analyzer`（原来是包所在目录向上四级）。
- 新入口 `kernel-analyzer-measure`、`kernel-analyzer-analyze`（`kernel_analyzer/cli.py`；仓库脚本调用同一函数）；`pyproject.toml` 声明运行
  依赖（numpy、scipy、gmpy2、torch、triton；python-flint、z3-solver 为可选）。
- 验收：`2_tool/scripts/acceptance/clean_wheel_install.sh`——从 `git archive` 构建 wheel，装进源码树外的新虚拟环境，在源码树外运行两个入口与
  一次真实测量，并在审计钩子下确认没有打开或导入 `2_tool/` 下的任何路径（第 8 节 H-K1）。单元测试另在仓库外的包副本中以 `python -I`
  加审计钩子运行。

## 4. bias 查询

- 查询在数据之前固定（Q1）：展开声明的 `query` 写出比较目标（模式 A：e_num = K − G_π；模式 B 另有 e_sem）、输入分布（声明的来源、因素与
  采样摘要）、观察量（输出、投影 R1/R2/R3/R5 的定义、坐标集规则）、抽样单位（一个种子一次输入抽取；单位内的重复启动不是单位）、判定轴
  与规则族；这些都进 `declaration_sha256`。
- 严格有界路线与声明族保证（Q2）：给出声明的逐元素界 M 时，每条规则有 Hoeffding 区间与 p 值 p = min(1, 2 exp(−n d² / (2 M²)))（与区间
  互相一致），每个规则类内对有界路线单独做 Holm：类内 FWER ≤ alpha（需 |a| ≤ M 对每个单位成立；无分布假设、有限样本）。近似路线保持
  原来的类内 Holm。
- 两个判定轴（Q3）：`statistics.<类>.axes` 分报非零轴（近似 / 有界，各自的族）与等价轴；只有请求等价轴（`query.axes` 含 equivalence，或给出
  `equivalence.rel`）时要求 δ；请求了却没有 δ 时声明不完整（列出 `equivalence.rel`）；给了 δ 却在 `query.axes` 中排除等价轴也算声明矛盾。

## 5. 可复用的组合规则

登记在 `2_tool/src/kernel_analyzer/composition_rules.py`；`2_tool/tests/test_composition_rules.py` 核对每个实现入口可解析、每个引用的测试存在、
每个报告字段由统一入口写出；展开声明写出规则集与登记摘要（`composition_rules`）。规则不依赖算子或 kernel 名字：满足义务的程序都得到
相应保证；义务不满足时报告指出是哪一条，不会悄悄退到更弱的读法。

| 规则 | 语义义务 | 实现入口 | 可检查证据 / 独立答案 | 组合测试 | 统一入口报告字段 |
| --- | --- | --- | --- | --- | --- |
| M1 storage identity and lifetime | an address names one storage instance only while it lives: every provenance decision is made on storages held alive by the analysis and matched by address and StorageImpl identity | `InputSnapshot`<br>`StorageMap.bind`<br>`StorageWatch.hold` | StorageImpl identity (_cdata) of the held storage versus the captured argument | `test_declared_read_identity_and_coverage`<br>`test_clean_storages_are_held_alive_and_released_when_not_clean` | `provenance_seed0.reads` |
| M2 exact coverage | a write through a view covers exactly the bytes of its elements (offset, strides, overlap, stride 0); a read covers the elements the reference loaded from initial values; a hull write joins | `byte_runs`<br>`read_runs_of`<br>`KernelReferenceEvaluator._note_initial_read` | the bytes a real torch write of 0xFF through the same view changes | `test_byte_runs_equal_the_bytes_a_real_write_changes`<br>`test_overlapping_full_size_view_is_not_full_coverage`<br>`test_partial_write_through_an_overlapping_full_size_view`<br>`test_copies_and_exact_constants_keep_the_call_level_reference` | `outputs.*.reference.reference_scope` |
| M3 content is not provenance (declared-input read) | a declared-input read needs: bytes read inside the declared input tensors, equal to that storage's own bytes before the call, and no write evidence (unannounced version increments, ATen writes in the aligned producer trace) | `InputSnapshot.declared_read`<br>`StorageWatch.count`<br>`torch_intermediates` | the bytes of the storage itself before the call; real version counters | `test_declared_read_uses_the_storages_own_bytes`<br>`test_storage_watch_counts_aten_writes_through_views_and_detach`<br>`test_cross_input_digest_match_is_not_a_declared_input`<br>`test_inplace_arithmetic_with_unchanged_bytes_on_an_input`<br>`test_inplace_arithmetic_through_data_is_seen_by_the_producer_trace`<br>`test_declared_inputs_and_legal_aliases_stay_declared` | `provenance_seed0.reads`<br>`outputs.*.reference.reference_scope` |
| M4 cross-launch carry-over | the reference memory of a recorded launch carries over only with unchanged bytes and no write evidence in between; announced raw-pointer writes (AOTAutograd) are not ATen writes | `carry_over`<br>`evaluate_sequence`<br>`StorageWatch.install` | real version counters around torch.autograd.graph.increment_version | `test_carry_over_needs_bytes_and_agreeing_counts`<br>`test_storage_watch_excludes_announced_raw_pointer_writes`<br>`test_inplace_arithmetic_between_recorded_launches`<br>`test_announced_compiled_inplace_write_keeps_the_composition` | `provenance_seed0.carry_overs`<br>`provenance_seed0.external_reentries` |
| M5 producer lattice | input < const < copy are clean, computed / triton / unlabeled are not; a read joins the labels of the bytes it reads; an exact write replaces, a hull write joins | `join`<br>`StorageMap.write`<br>`producer_records` | a per-byte model of the same lattice | `test_join_is_commutative_associative_idempotent_on_kinds`<br>`test_storage_map_matches_a_per_byte_model`<br>`test_partial_constant_write_over_uninitialised_bytes_stays_computed` | `outputs.*.upstream_with_producer_record`<br>`outputs.*.mixed_non_triton_sources` |
| T1 the traced run does not change the measured runs | the unmeasured producer trace leaves compiled code and its caches untouched (force_eager stance); without that API no trace is made | `producer_records` | the recorded launches of every later seed (the compiled kernel is still launched) | `test_announced_compiled_inplace_write_keeps_the_composition` | `producer_trace_per_seed`<br>`notes.ir_kinds` |
| P1 precision dependence is observed, not inferred | a higher working-precision level can change an enclosure only through a precision-dependent rule (SumK / DotK with the level's K, prefix sums); the pass counts their calls | `PRECISION_DEPENDENT_CALLS`<br>`sum_k`<br>`icumsum` | exact rational row sums inside every level's enclosure | `test_three_level_summation_enclosures_against_exact_row_sums` | `precision_dependent_calls`<br>`refinement.levels` |
| P2 refinement stop categories | the controller refines while an output can still improve and stops only as met, budget exhausted, backend limit (float64 endpoints, highest level, no precision-dependent rule), intrinsic set width, no numerical enclosure or pass failed; an unchanged pass fraction is not a stop reason | `run_level`<br>`REFINEMENT_CATEGORIES` | level-by-level widths of the three-level summation (levels 1 and 2 resolve nothing) | `test_three_level_summation_reaches_level_3_from_the_unified_entry`<br>`test_controller_does_not_stop_on_an_unchanged_pass_fraction`<br>`test_controller_stop_categories`<br>`test_controller_budget_between_levels`<br>`test_refinement_reports_a_target_it_cannot_reach` | `refinement.category`<br>`refinement.outcome`<br>`refinement.per_output` |
| P3 resolution counts numerical enclosures only | the requested resolution is measured on complete finite elements; set targets (L_E) are counted apart with their own width and never reported as an enclosure that is too wide | `reference_quality` | per-element set-target masks of the reference (Buffer.iset) | `test_controller_stop_categories` | `outputs.*.reference.set_target_elements`<br>`outputs.*.reference.unresolved_elements` |
| Q1 the query is fixed before the data | comparison target, input distribution, observable and sampling unit are written into the expanded declaration (and its digest) before any unit is drawn | `query_of`<br>`expand` | the declaration digest changes with each of the four items | `test_query_fixes_target_distribution_observable_and_unit` | `expanded_declaration.query` |
| Q2 strict bounded route with a family guarantee | with a declared per-element bound M, every rule gets Hoeffding's interval and p-value on endpoints truncated to [-M, M]; Holm within each declared rule class gives FWER <= alpha per class for the bounded route, apart from the approximate route | `bounded_route`<br>`class_statistics` | Hoeffding's tail bound evaluated directly; simulated family-wise error under the null | `test_bounded_p_value_matches_the_hoeffding_interval`<br>`test_bounded_route_family_wise_error_under_the_null`<br>`test_bounded_route_holm_within_class` | `outputs.*.statistics.*.axes.nonzero.bounded` |
| Q3 two decision axes | the nonzero axis and the equivalence axis are reported apart; an equivalence tolerance delta is required only when the equivalence axis is requested, and is then part of the declaration | `missing_items`<br>`class_statistics` | declarations with and without the equivalence axis | `test_equivalence_axis_requires_delta_only_when_requested`<br>`test_axes_are_reported_apart` | `outputs.*.statistics.*.axes.nonzero`<br>`outputs.*.statistics.*.axes.equivalence` |
| K1 no repository path at run time | the installed package imports and opens no path of the source repository; console entry points run the unified entry and the captured-package analysis | `measure_main`<br>`analysis_main`<br>`statistical_judgment`<br>`default_cache_dir` | an audit hook on open / import in a process that runs a copy of the package | `test_package_runs_from_a_copy_outside_the_repository` | — |

registry_sha256: c8367a2dcee05e5aae3f0357d6067cb9143d594800792813e0e92544b4c81fd0

## 6. 代码与测试

| 文件 | 内容 |
| --- | --- |
| `reference_eval/storage_effects.py`（新） | 覆盖、标签格、`StorageMap`、`StorageWatch`、`InputSnapshot`、`carry_over` |
| `reference_eval/capture.py` | 记录器持有张量并在每次启动记录 `CapturedArg.mutations`；记录期间拦截 `increment_version` |
| `reference_eval/ttir_eval.py` | `KernelReference.loaded_elements`；`evaluate_sequence` 的沿用按 M4，记录 `carry_overs` |
| `provenance.py` | 逐字节生产者（M2、M5），按参照读到的元素取标签；`force_eager` 下追踪（T1） |
| `check.py` | 调用前快照、观察输入、每个读到浮点初值的 seed 都追踪；报告 `provenance_seed0`、`producer_trace_per_seed`、`precision_dependent_calls` |
| `reference_eval/intervals.py` | `PRECISION_DEPENDENT_CALLS` |
| `measure.py` | 精度控制器与类别、`reference_quality` 分开集合目标、`query`、两个判定轴、有界路线 Holm、`unit_split` |
| `reference_eval/sensitivity.py` | 有界路线的 p 值与方向 |
| `contract_v3.py`、`cli.py`、`composition_rules.py`（新） | 见第 3、5 节 |
| `2_tool/scripts/dsl_v2/regression_v11.py`（冻结后） | 默认不运行竞争程序（第 8 节的执行偏差） |
| 测试（新） | `test_storage_effects.py`（15，CPU，独立答案）、`test_batch1_guarantees.py`（26，GPU 端到端与控制器）、`test_bias_query.py`（6）、`test_composition_rules.py`（4） |
| 测试（调整） | `test_audit_findings.py` 三处：生产者表改用 `StorageMap` 接口（期望标签不变）；精度结局断言改为新类别（「最高级」）；`compiled_after_copy` 加 D12 断言（基线上失败）；`test_output_binding.py` 场景改用私有内存池（断言不变，基线上通过） |

全量测试：1602 通过（`evidence/full_suite.log`，含前两轮发现的失败与原因）。

## 7. 修复前失败的测试（基线 3532066）

把本批新测试放到基线的工作树中运行（工具代码不变）：`evidence/baseline_3532066_new_tests.log`。端到端与控制器 26 项中 20 项失败，失败点
都在实质断言上——三类来源问题全部得到 call 级（重叠写入那一项正是「producer record: const: aten.fill_.Scalar」），三级求和停在第 2 级
（「no improvement」），包副本在 `import contract_v3` 处失败；6 项在基线上也通过，是保留能力的对照（复制、常数、偏移 view、读后原地、
classic）与精确行和的包围检查。编译原地写入那一项在基线上范围正确，只因新报告字段缺失而失败（它是本批改动的回归防护）。
存储效果与组合规则两个测试文件在基线上无法收集（模块不存在）；查询测试 6 项全部失败。

## 8. 验收

协议与保留集在运行之前冻结于 `da4b64e`（工具代码 `d26d5f6`）：`1_experiments/batch1_acceptance/`（`protocol.md`、`items.json`、
`predictions.json`、`holdout_calls.py`、`predict_levels.py`、`run_acceptance.py`、`compare_regression.py`）。保留集的结构与谱系没有参与开发
（Gluon、二维跨步 / 重叠 view、`where` / `cat` / `index_put_`、Inductor 编译的读者、inference-mode 输入、两次启动链、n = 256 的轴 1 求和、
反向前缀和、exp / float64 / 争用整数输出、bf16 舍入偏差、干净 wheel、公开 v1.1 回归），种子从 7000 / 9000 起。逐项结果：
[1_experiments/batch1_acceptance/README.md](../../1_experiments/batch1_acceptance/README.md)，报告与日志在其 `results/`。

| 组 | 项数 | 成立 | 不成立 | 无法判断 | 要点 |
| --- | --- | --- | --- | --- | --- |
| 来源 | 14 | 14 | 0 | 0 | 真值不干净而给 call 级：0（可靠性违反 0）；真值干净而给 kernel 级：3（H-P1b 恒等 `clamp_`、H-P7 `index_put_` 拷贝、H-P12 改写另一半），与事先登记的保守预期一致 |
| 精度 | 5 | 5 | 0 | 0 | H-R1、H-R2 在第 2 级达标，与 60 位精度下的独立预测一致；exp 为「无精度相关规则」、float64 为「float64 端点」、争用整数返回值为「固有集合宽度」 |
| bias | 3 | 3 | 0 | 0 | 向零舍入：R1 / R2 在近似与有界路线（类内 Holm）都判非零（正向，即坐标均值为负、幅度被拉向零），等价轴「未显示」；就近偶舍入：等价轴「在 δ 内」；请求等价而无 δ：声明不完整 |
| 安装 | 1 | 1 | 0 | 0 | `git archive` 构建、源码树外新虚拟环境、两个入口、一次真实测量；审计钩子下对 `2_tool/` 的访问为空 |
| 回归 | 1 | 1 | 0 | 0 | 公开 v1.1 的 28 个常规程序与 reg20261009T2319 逐程序相同（预计命中 27/28、安全违反 0 也相同）；运行 reg20261010T1241 |

另做的捕获抽查：官方主线 Triton 下重跑教程 01、02、05 的捕获，5 次启动的状态与计数和 `auditfix_tutorials` 记录相同（捕获插件按单次启动
求值，本批改动不改变它们）。

**执行偏差（人工介入记录）**：回归运行脚本按测量通道选全部程序；上一轮靠预先放入沿用的作业文件把 4 个竞争程序（T5：prog_01、prog_13、
prog_22、prog_23）排除在 GPU 之外，这一轮没有放入，它们被在 GPU 上运行（各约 11 分钟，正常退出）。这违反任务书「不绕过权限运行竞争
kernel」的要求；运行没有经过、也没有绕过权限提示，但不应发生。处理：4 份结果移出记录（本地 `.cache/batch1/race_programs_run_in_error/`
保留作偏差证据），记录仍沿用 reg20261009T1051 的作业文件；比较只用 28 个常规程序（与上一轮相同的范围）。冻结之后的改动：只改了运行
脚本 `2_tool/scripts/dsl_v2/regression_v11.py`——默认跳过竞争程序，`--program` 拒绝它们，只有显式的 `--allow-race-programs`（需用户本人
许可）才运行；工具代码未改，验收结论不受影响。

## 9. 待验收与未完成

- 外部审阅方的保留集：尚未提供；本批保留集由执行方在代码冻结前编写，不能替代（待验收）。
- AMD 设备：没有，AMD 目标的设备验证仍为 pending。
- 第二工作线的其余共同阻塞（区域求值的更多形状、异步 / 描述符的内存效果、跨启动的保存值与梯度）留给后续批次；本批只交付来源 / 内存
  效果传播与精度传播两组。
