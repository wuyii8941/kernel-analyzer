# 检测器 / 工具版本变更记录

## 4.0（开发中，dsl-v2 分支，未发布、未冻结）——参照 DSL v2 rc3 正式版的第一个增量

依据 `reference_dsl_v2_rc3_official_20261009.zip`（SHA-256 8050903f…6faf）。general-v3.1 与它下面的结构验收 v1.1 结果不动；
v1.1 的计分只绑定运行 r20261008T2030。

- **通用 combine 区域（rc3 02 §6.2，04 W2）**：没有快速路径的 `tt.reduce` 不再以「unrecognized reduction combiner」拒绝。
  它沿锁定 lowering 的合并次序逐步解释区域：次序取自捕获的 TTGIR 布局，线程内按寄存器次序、lane 间 butterfly、warp 间 butterfly，
  一个 warp 内同步的结果取各 lane 包围的并。目标是「顺序特定」的。前提交审阅方：次序模型（逐位模拟器在浮点求和上与设备逐位一致；
  combine 内的操作数次序按 lowering 源码）。没有 TTGIR 布局时记未建立。Welford 闭式（Φ 证书）保留为快速路径；闭式因零权重而依赖
  合并树的行，改按实际的树求值。规则改变：v3.1 的两条「拒绝」测试改为核对顺序特定参照，并在测试里独立实现精确有理数的归约树。
- **执行有效性（rc3 02 §6.3、§8.7，04 W4）**：
  - 同一次启动中，一个 program 读的地址被另一个 program 写，无论求值先后都记执行竞争；v3.1 只抓到「读到已写值」的一半。
  - 同一 program 先 store、后 load 同一元素，中间没有屏障：两次访问由同一个线程持有（TTGIR 布局）才算有序；被别的线程读到是执行竞争；
    线程映射不可得时执行有效性未建立，不宣称竞争。
  - 被复制的元素由最小线程号写入，这一点在 PTX 上可见：标量 store 的谓词只放行线程 0，load 没有谓词，中间没有 bar.sync。
  - `gpu.barrier` 开始新的阶段（原来是空操作）。
  - 新测试只在 CPU 上求值编译产物（`triton.compile`，不启动），有竞争的 kernel 从不运行。
- 注册表写到 `2_tool/data/rule_registry.json`（`2_tool/scripts/dsl_v2/build_rule_registry.py`）；`pre-reorg-20261009:results/general/` 下的 v3.1 注册表冻结不改。

### 外部审计（基线 1aee15e）修复版，2026-10-09

逐项处理、修复前后测试与证据见 [3_audits/fix_1aee15e](../../3_audits/fix_1aee15e/README.md)。

- **F01 整数精确域**：无符号除法 / 余数 / 向上取整除 / 逻辑右移 / 最值与无符号比较在 uint64 上计算（64 位的最高位不再当符号）；
  有符号除法族按 uint64 绝对值计算（INT64_MIN 的绝对值和取负不再在宿主上溢出）。值、下标、地址、条件都走 `_int_op` / `_cmpi`。
- **F02 scaled dot**：非 e2m1 操作数传完整上下界（原来只取下端）；scale 的条件标记进入输出；`amdg.scaled_upcast_fp8` 同样保留上端；
  只有一侧 scale 的 `tt.dot_scaled`（如 bf16 × e4m3）按自定义语法区分操作数（原来中止）。
- **F03 证明状态轴**：只在证据上核对过的前提（CAS 的串行化一致、scan 的两种括号一致）下成立的元素记「complete_under_premise」，
  不进无条件完整率，也不进统计；前提随读写传到后续启动。CAS 在不超过 4 个 program 时求值全部 program 顺序（审计的三 program 非交换锁
  反例因此记未建立），否则仍是正序与逆序。其余 `assumed:` 记为 lowering 事实 / 可信顺序目标（清单交审阅方）。
- **F04 调用级来源**：删除「每个元素的位模式都在输入中」与「全零」作为复制证明；只有声明输入本身的存储（未被就地改写）算输入，
  其余非 Triton 上游一律保持 kernel 级；每个 seed 都检查；原子更新 / CAS / poll 读取目标时记为读取（原来 ATen 填充的累加器不被记为上游）。
- **F05 Gluon 统一入口**：`check.run` 的覆盖检查、原子识别与求值用同一 IR 选择（`launch_ir`：TTIR，Gluon 为 TTGIR），报告 `ir_kind`。
- **F06 声明摘要**：`declaration_sha256` 覆盖整个展开声明（采样划分与 seed、alpha、因素、分辨率、M、δ、重复次数、统计族、策略、版本与
  call / 输入生成器 / 规格源码的内容哈希）；输入采样改用 `sampling_seed_sha256`（与旧摘要同式，样本不变）。alpha、δ（新字段
  `equivalence.rel`）、重复次数（新字段 `repeats`）与 M 传入 `check.run` 并出现在结果中；没有 δ 时写明不作等价陈述；分辨率报告
  `resolution_met`，自适应提高精度未实现（W2 / W6）。
- **F07 统计准入**：执行有效性未建立（线程映射不可得、进展未证明）时停止统计；重复启动的差异只有全部落在浮点原子写入的元素上才按输入内
  平均；竞争原因码按生产者列表结构化（补上原来漏掉的「cross-program write race」「cross-program race on load」「conflicting lanes」）。
- **修复中另发现的入口缺陷**：（1）用例超时（SIGALRM）在用例准备之前就已武装，准备阶段第一次导入 `torch._dynamo` 约需 1 秒，1 秒的预算会
  打断这次导入，留下半初始化的模块，使同一进程中后续所有水平出错——现在先导入再武装；（2）超时异常是 `Exception`，在声明的调用里触发时
  被调用包装当作「调用拒绝了声明的输入」（归为绑定失败）——现在是 `BaseException`，预算耗尽总是报「over budget」。两者都有修复前失败的测试。

## 3.1（2026-10-08，标签 `general-v3.1`）——审计修正（替代 general-v3.0 作为第 5 项的冻结版本）

冻结：提交 d6e177e，src 树 `cd21ef56cf46eb3cff2c00c50b8918b30a604d26`。

- **SumK 的界在 3.0 中不严格**（独立审计给出反例）：最后一遍对全部 n 项用 numpy 成对求和，主导项 p_n 被多次舍入，破坏 Ogita–Rump–Oishi
  界的 u|s| 项；病态行（±2⁵³…2⁸⁰ 相消，n ≥ 8）上 |res − s| 超出 err 可达 1.47 ulp，`isum` 的包围在 12000 行中有 3 行没有包住精确和。
  3.1 改为前 n − 1 项成对求和、最后单独加 p_n（证明所用的结构）；审计探针的全部病态类别 0 违反，`2_tool/tests/test_sumk_rigorous.py`（含反例）。
  3.0 的「严格界」说法撤回；3.0 下的结果用 3.1 重跑（`1_experiments/general_round_v3_1/results/fr_modeB_v3_1`、探针、演示、回归 v3_1）。
- **Welford**：不带保护的比值 r = w_b / W 在两个可能为零的权重相遇时会出现 0 / 0 → 未建立（3.0 误记为完整）；全零权重行改按输入判定
  （3.0 中该分支走不到）；舍入核对模式与逐位模拟器拒绝 Welford（浮点合并依赖树）。测试 +2。
- **上游来源**：纯数据搬运的上游（每个元素的位模式都在声明输入中）标「[copy of inputs]」；统一入口据此报告调用级完整率，其余混合输出
  只算 kernel 级参照（任务书「不得用实际中间值回填充当完整参照」）。测试 +1。
- **统一入口**：调用本身拒绝输入才记「绑定」；其余未分类原因记「未分类（工具错误或新原因）」，不再推给用户；失败映射补齐若干原因；
  声明的误差预算只记录，入口不应用有界路线（3.0 的文字说法过强，已更正）。
- 验收框架：冻结检查覆盖 src、`contract_v3.py`、`common.py` 与框架本身，并检查未跟踪 / 忽略文件；计分核对答案哈希与效应方向。

## 3.0（2026-10-08，标签 `general-v3.0`）——通用能力轮：参照语义与默认统计

- **Welford 多值归约**（`ttir_mapping.match_welford`、`ttir_eval._welford_reduce`）：按数据流结构匹配（不按名字），实数语义取封闭形式，
  前提权重 ≥ 0 且 W > 0；W = 0 的均值、可能为负的权重 → 未建立。测试 `2_tool/tests/test_welford_reduction.py` 6 项（含与任意合并树的精确等价、
  反例与负对照）。以前被拒绝的 bf16 方差现在建立完整参照。
- **精确累加**（`intervals.sum_k`、`dot_k`）：求和与点积用 SumK / DotK（K = 3）及 Ogita–Rump–Oishi 严格界，与旧 γ 界取交集；
  `KA_ACCUMULATION=gamma` 只用于前后对照。家族表中宽度约缩小一个数量级。
- **同一变量**：x − x = 0、x / x = 1（x ≠ 0）。
- **统一入口** `kernel_analyzer.measure`（不改冻结统计层，类内 Holm 在入口中进行）；**规则注册表** `reference_eval.rule_registry`。
- **回归**：全部测试 1844 项通过；回归集 15 个用例中判定、参照类别、特殊值计数全部不变，3 个用例的 e_sem 相对 RMS 在 1e-16 量级变化
  （K_R 中点移动不到一个 float64 ulp），1 个用例只有已容许的 float atomics 差异（`pre-reorg-20261009:results/regression/v3_0`）。

## 2.3（2026-10-08）——上游（非 Triton）来源检测（收束轮 F / FR 运行中发现）

- **问题。** `check.torch_intermediates` 决定一个 Triton 输出是否读了 torch / ATen 在捕获窗口内产生的值（这些值以捕获值作为精确输入
  进入 K_R，e_sem 因而是「混合」而不是纯语义）。2.2 有两处漏判：(a) 某次启动是否写了一个存储，按「启动前后字节是否变化」判断——
  当缓存分配器把装着同一结果的缓冲区交回（预热与第一个单位用同一 seed）时，写入被判为没有发生，上游依赖丢失（组合 C1：y 读 cuBLAS
  写的 h，被报成纯 Triton）；(b) 判断「是否为案例输入」的摘要在启动之后取——启动内的 ATen 原地修改了输入（`F.embedding(max_norm=)`
  对权重的重归一化）时，被修改的缓冲区被当成原始输入。两处都只影响「混合」标记，不影响 K_R、e_num、e_sem 的数值与绑定。
- **修改。** 求值器在每次启动的结果里记录本次参照写过的存储（`KernelReference.stored`：store、atomic、经未建立地址写入的目标）；
  `torch_intermediates` 以它为准判定写入，字节变化只作补充；`check.run` 在启动之前取输入摘要（`input_digests`）并传入。
- **回归。** `2_tool/tests/test_upstream_sources.py` 4 项：同字节陈旧缓冲区仍报上游、只读输入的输出不报上游、`stored` 记录写过的存储、
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
- **回归。** `2_tool/tests/test_output_binding.py`：复用地址后大小不同 / 大小相同内容不同 / 大小相同内容也相同，各在持有与不持有引用下运行
  （不持有时确实发生了地址复用，结论必须是「未建立」），外加 Triton 写出的输出仍被绑定；7 项通过。相关的既有测试 98 项通过。
- **对第一阶段记录的定向重捕获**（`1_experiments/essential_bugs_round/results/phase2a/a1_recapture.json`，24 个条件）：池化 8 个条件的梯度正确记为「不由
  Triton 写出」（ATen 回退），6 个原 IndexError 的条件 FR 建立（前向无语义差异），pool394 的虚假语义标记消失；index 16 个条件的梯度
  结论不变（idx504、idx542 的 `grad_source` 仍为语义差异，即 B020）；index 前向输出在部分 seed 上 K_R 没有标记写入（Triton 先初始化、
  ATen 回退原地完成归约），在 2.1 与 2.2 中相同，与绑定无关，记为 FR 覆盖的已知局限。第一阶段 F / E / P 的结论无变化。
- 第一阶段成绩仍对应 `eval-stage-a-20261006`（2.1），冻结不改。
