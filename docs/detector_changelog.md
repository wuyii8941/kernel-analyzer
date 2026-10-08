# 检测器 / 工具版本变更记录

## 3.0（2026-10-08，标签 `general-v3.0`）——通用能力轮：参照语义与默认统计

- **Welford 多值归约**（`ttir_mapping.match_welford`、`ttir_eval._welford_reduce`）：按数据流结构匹配（不按名字），实数语义取封闭形式，
  前提权重 ≥ 0 且 W > 0；W = 0 的均值、可能为负的权重 → 未建立。测试 `tests/test_welford_reduction.py` 6 项（含与任意合并树的精确等价、
  反例与负对照）。以前被拒绝的 bf16 方差现在建立完整参照。
- **精确累加**（`intervals.sum_k`、`dot_k`）：求和与点积用 SumK / DotK（K = 3）及 Ogita–Rump–Oishi 严格界，与旧 γ 界取交集；
  `KA_ACCUMULATION=gamma` 只用于前后对照。家族表中宽度约缩小一个数量级。
- **同一变量**：x − x = 0、x / x = 1（x ≠ 0）。
- **统一入口** `kernel_analyzer.measure`（不改冻结统计层，类内 Holm 在入口中进行）；**规则注册表** `reference_eval.rule_registry`。
- **回归**：全部测试 1844 项通过；回归集 15 个用例中判定、参照类别、特殊值计数全部不变，3 个用例的 e_sem 相对 RMS 在 1e-16 量级变化
  （K_R 中点移动不到一个 float64 ulp），1 个用例只有已容许的 float atomics 差异（`results/regression/v3_0`）。

## 2.3（2026-10-08）——上游（非 Triton）来源检测（收束轮 F / FR 运行中发现）

- **问题。** `check.torch_intermediates` 决定一个 Triton 输出是否读了 torch / ATen 在捕获窗口内产生的值（这些值以捕获值作为精确输入
  进入 K_R，e_sem 因而是「混合」而不是纯语义）。2.2 有两处漏判：(a) 某次启动是否写了一个存储，按「启动前后字节是否变化」判断——
  当缓存分配器把装着同一结果的缓冲区交回（预热与第一个单位用同一 seed）时，写入被判为没有发生，上游依赖丢失（组合 C1：y 读 cuBLAS
  写的 h，被报成纯 Triton）；(b) 判断「是否为案例输入」的摘要在启动之后取——启动内的 ATen 原地修改了输入（`F.embedding(max_norm=)`
  对权重的重归一化）时，被修改的缓冲区被当成原始输入。两处都只影响「混合」标记，不影响 K_R、e_num、e_sem 的数值与绑定。
- **修改。** 求值器在每次启动的结果里记录本次参照写过的存储（`KernelReference.stored`：store、atomic、经未建立地址写入的目标）；
  `torch_intermediates` 以它为准判定写入，字节变化只作补充；`check.run` 在启动之前取输入摘要（`input_digests`）并传入。
- **回归。** `tests/test_upstream_sources.py` 4 项：同字节陈旧缓冲区仍报上游、只读输入的输出不报上游、`stored` 记录写过的存储、
  启动内原地修改的输入算作上游；修改前其中 3 项失败。相关既有测试（绑定、TTIR、反例、包含性）共 88 项通过；`test_output_binding.py`
  的版本断言改为「≥ 2.2」。检测阈值未变。
- **影响的结论**：模式 B 中「纯 / 混合」的划分（收束轮的 FR 与组合在 2.3 上重跑）；第一、二阶段与 2b 的冻结记录不重算：第一阶段
  `classify.fr_assess` 的语义判定按 SEM_REL 幅度阈值，只携带混合标记、不读它；2b 的语义结论来自 F 与精度不变性。

## 2.2（2026-10-07，标签 `detector-v2.2`）——输出来源绑定（执行协议第二版 A1）

- **问题。** `kernel_analyzer.check.run` 按 `data_ptr()` 把返回的输出配给捕获窗口内 Triton 写过的缓冲区，再比内容。地址相同不等于
  同一存储实例：输出由 ATen 回退生成、而缓存分配器复用了已释放的 Triton 中间量的地址时，输出被配给该中间量——大小不同时
  IndexError，大小相同时把旧缓冲区当作输出（第一阶段池化 6 个 IndexError、pool394 的虚假语义标记）。
- **修改。** `TritonLaunchRecorder` 在窗口内持有每个被记录启动的张量参数的存储引用（`keep_storages=True`，默认），并记录存储实例
  标识 `storage_id`（`untyped_storage()._cdata`）；`check.run` 只在输出的存储实例属于该地址上记录的实例时才配对，否则记为
  「不由 Triton 写出」（`outputs_at_address_of_another_recorded_storage`）；没有持有引用或没有记录标识时记为「绑定未建立」
  （`outputs_binding_not_established`），不配对。报告新增 `tool_version`。
- **为什么要持有引用。** 实例标识本身（StorageImpl 的地址）在实例释放后同样会被复用；只有在实例存活期间，地址与标识才唯一。
- **回归。** `tests/test_output_binding.py`：复用地址后大小不同 / 大小相同内容不同 / 大小相同内容也相同，各在持有与不持有引用下运行
  （不持有时确实发生了地址复用，结论必须是「未建立」），外加 Triton 写出的输出仍被绑定；7 项通过。相关的既有测试 98 项通过。
- **对第一阶段记录的定向重捕获**（`results/essential/phase2a/a1_recapture.json`，24 个条件）：池化 8 个条件的梯度正确记为「不由
  Triton 写出」（ATen 回退），6 个原 IndexError 的条件 FR 建立（前向无语义差异），pool394 的虚假语义标记消失；index 16 个条件的梯度
  结论不变（idx504、idx542 的 `grad_source` 仍为语义差异，即 B020）；index 前向输出在部分 seed 上 K_R 没有标记写入（Triton 先初始化、
  ATen 回退原地完成归约），在 2.1 与 2.2 中相同，与绑定无关，记为 FR 覆盖的已知局限。第一阶段 F / E / P 的结论无变化。
- 第一阶段成绩仍对应 `eval-stage-a-20261006`（2.1），冻结不改。
