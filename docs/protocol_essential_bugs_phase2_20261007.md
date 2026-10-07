# 本质错误一轮·执行协议第二版：第一阶段补充、2a 定向收尾、2b 搜索扩张（运行前提交）

日期：2026-10-07。依据：用户下发的「第二阶段任务书：定向收尾（2a）与搜索扩张（2b），单次执行」，以及用户指示「先补第二类
（第一阶段范围内未做全的部分），然后直接进入第二阶段」。第一阶段的执行协议 `docs/protocol_essential_bugs_20261007.md` 与其结果
（`results/essential/phase1/`，提交 0bb1b68）冻结不改；本阶段结果单独入库（`results/essential/phase1_supplement/`、
`results/essential/phase2a/`、`results/essential/phase2b/`）。本文在对应运行之前提交；之后的改动只追加到第 9 节。

## 0 范围与边界

- 本轮主任务是训练计算的语义与数值搜索扩张；原有的盲测成绩、外部复用试验、成本统计、训练后果案例沿台账维护，不删改。
- 主覆盖范围：LLM / Transformer 训练主干加跨任务通用的基础算子；第二档进扩展队列（任务书 G2）。「覆盖完成」只对冻结清单与条件成立。
- 任何文档不写「完整 DSL 分析是发现这些错误的必要条件」。结论边界按任务书第 1 节。
- **候选来源的说明。** 任务书 G2/G4 把 Liger、Unsloth、FlashAttention、DeepSpeed、apex、bitsandbytes、torchao 列为若干家族的候选来源；
  用户此前指示不要因本机环境而关注 Liger / Unsloth（第一阶段偏离第 5 条）。处理：候选按 G4 逐个登记，PyTorch 自身的实现优先；
  库实现只在任务书点名的家族纳入，并在报告中单列。若用户仍要排除，只删去对应候选行，不影响其余结果。
- 规格语义只由审阅方修改；新家族的独立规格未交付并审阅入库之前，F 与模式 B 的 FR 不运行，E、前提已明确的 P、有 TTIR 时的
  模式 A 照常运行（任务书 G8 与第三部分第 3 条）。

## 1 第一阶段补充（「第二类」，事后补充，单独入库，不改第一阶段成绩）

| 编号 | 内容 | 对象 |
|---|---|---|
| S1 | 实现的前后向自一致性：`torch.func.jvp` 的 ⟨v, J u⟩ 与 autograd VJP 的 ⟨B(v), u⟩，随机整数方向 u、v | 第一阶段三个家族的全部条件，eager 候选（前向模式只对 eager 可用；Inductor 记为「不适用」） |
| S2 | 补齐 W4 的性质：CE 改变被忽略行（及概率为零处不涉及）的 logits 不影响 loss 与其余行的梯度；avg pool 的线性 f(a·x + b·y) = a f(x) + b f(y)；index/scatter 中线性归约（index_add、sum、mean）的线性与可加性；池化的通道置换等变 | 第一阶段三个家族，全部受裁决的候选 |
| S3 | 补齐 W6 的边界输入：池化与 index/scatter 的极大 / 极小 / 跨尺度数值、非连续 stride（转置与步长视图）、非零 storage offset、不是块大小整数倍的较大尺寸 | 新增补充条件（`scripts/essential/conditions_supplement.py`，随本文提交），E / F / P / FR 同第一阶段 |
| S4 | Inductor 收到的 label_smoothing 标量：从捕获的 TTIR 常数与标量参数中核实（与 A3 的「接口 / 常量来源是否核实」合并） | CE 中 ε ≠ 0 的 Inductor 条件 |
| S5 | W8 的规格书写时间 | 规格由审阅方提供，执行方没有该数据，记为「未知（需审阅方提供）」 |

S1–S3 的结果标为「事后补充」，与第一阶段的预注册成绩分开报告。

## 2 2a 定向收尾

### A 工具修复（新检测器版本；第一阶段成绩保留）

- **A1 输出来源绑定。** 现状：`check.run` 按 `data_ptr()` 把输出配给 Triton 写过的缓冲区，再比内容；地址相同不等于同一存储实例，
  内容相同也不证明生产者相同。修法：`TritonLaunchRecorder` 在捕获窗口内对每次启动的张量参数持有存储的强引用（窗口内被释放的
  中间量不会被缓存分配器复用），并记录存储实例标识（`untyped_storage()._cdata`）；`check.run` 只在输出的存储实例与记录的写入
  实例相同时才配对，否则把该输出记为「不由 Triton 写出」；实例无法确认时记为「未建立」。
  回归（`tests/test_output_binding.py`）：(1) 复用地址后大小不同；(2) 大小相同、内容不同；(3) 大小相同、内容也相同——三种情形下，
  由 ATen 生成、落在已释放 Triton 中间量地址上的输出都不得与该中间量配对。另对第一阶段受影响的记录（池化 6 个 IndexError、
  池化 2 个与 index 16 个「Triton 最后一次写之后被修改」的输出）做定向重捕获。
- 版本：修复后打标签 `detector-v2.2`；第一阶段成绩仍对应 `eval-stage-a-20261006`。变更记录写入 `docs/detector_changelog.md`。
- **A2 分类层边界保护**（`scripts/essential/classify.py`）：`ok_elements = 0` 时 FR 返回「未建立」；`combine()` 在 E 没有任何可比
  配对时返回「共有关系未建立」。各加单元测试（`scripts/essential/tests/`）。
- **A3 FR 结论字段**：每个单位分别记录 `e_sem_excludes_zero`、`above_action_threshold`（|中点| > 2⁻²⁰(1 + |f|)，冻结阈值不改）、
  `interface_source_verified`（接口 / 常量来源是否经捕获核实）。第一阶段「489 个 bf16 单位全部没有语义差异」按三个字段分别陈述。

验收：回归与单元测试通过；受影响记录清单、新版本号与变更记录入库。三个回归通过之前不做新的捕获。

### B W5 测试判据审查

对 `scatter_reduce`、`index_reduce`、`max_pool{1,2,3}d`、`cross_entropy`：从锁定版本（v2.10.0）的
`torch/testing/_internal/common_methods_invocations.py`、`test/inductor/test_torchinductor_opinfo.py`、`test/test_scatter_gather_ops.py`、
`test/nn/test_pooling.py`、`test/test_nn.py` 中列出以 eager 自身作参照的地方、被跳过或关闭的检查（附原因原文）、OpInfo 样例的值域，
以及样例是否含 include_self=False 与被排除值等于结果、并列、只采到 padding 的窗口。在 B020 的 50 个条件上运行 `torch.autograd.gradcheck`
（float64，CPU，默认容差）。回答 B020、B021 为什么没被现有测试发现。被跳过检查或以 eager 为参照的算子进入 G8 的发现驱动队列。

### C 轻量方法对照（事后分析，不改第一阶段 P 组的 28/50）

对象：B020 的 50 个条件、B021、B016 三种写法、HF 4.45.2 与 4.57.3、Inductor avg_pool 反向（22 个条件）。方法：gradcheck；
`torch.func.jvp` 对 autograd VJP 的点积核对；预注册的 P；F；FR。每个（对象, 方法）记录：能否发现、人工（是否需手写 f、是否需 TTIR）、
运行时间。

### D 三个发现的可复用触发检查（写入 2b 的预注册清单）

- **D1（B020 型）**：极值归约中，被排除的值等于结果时，反向仍不依赖它（∂out/∂excluded = 0，且被排除值的扰动不改变源梯度）；
  唯一极值处 JVP 与 VJP 相容；极值归约的梯度总量等于上游（每个输出位置的系数和为 1）。
- **D2（B021 型）**：合法的 padding 与 dilation 产生无贡献者窗口时，返回下标是否落在输入范围内；反向写入是否只落在合法位置
  （隔离子进程，`MALLOC_CHECK_=3`，检查邻近通道 / 样本的梯度为零）。
- **D3（B022 型）**：前向产生的哨兵下标（−1 等），反向是否正确处理（不崩溃、不路由）。
- **D4（输入生成）**：整数或小值域输入，使相等与并列确定性出现（每个条件至少一个被排除值等于结果、至少一个并列）。
- 实现：`scripts/essential/triggers.py`，独立于分类程序；单元测试 `scripts/essential/tests/test_triggers.py`。

### E 报告措辞修正（写入修正后的中期报告）

E1「FP64 参照」写成「同一错误实现的 float64 eager 运行」；E2 B020 计数（2.10 Inductor 48 个同样出错、2 个叠加 B016；nightly 覆盖 50
个；一个根因的多条件复现）；E3 H3 是「本轮实施的性质子集」的成绩，W4 为「子集完成，通用自一致性在 C 中补做」；E4 B021 的论证重点
为无效下标、跨通道写入、越界；E5 MaxPool1d 文档版本冲突归第二栏并准备文档 issue 草稿，真实 −∞ 与 padding 并列时 Inductor 返回 −1：
「返回无效下标」记契约违反、「梯度不路由」记约定，并入 B022；E6 不做 3 个 seed 上的平均作用统计。

### F 上游草稿（提交由用户决定）

B020（附 gradcheck、OpInfo 缺口、最小例子、版本）；B021（优先，附隔离复现与越界证据，说明参数满足文档前提）；B022（合并 −∞ 并列
返回 −1，说明 2.10 正确、nightly 回归）；B015 对 #198119 的补充评论；MaxPool1d 文档公式的文档 issue。每份附四项检索记录。

2a 完成时提交中期记录（`docs/essential_bugs_phase2a_record_20261007.md`），不等待批复。

## 3 2b 搜索扩张（定义见任务书第二部分；本节在 2b 的任何运行之前补全）

2b 的候选清单（G4，机器可读：`results/essential/phase2b/candidates.json`）与覆盖矩阵（G3：`results/essential/phase2b/coverage_plan.json`）
在第一次 2b 运行之前追加为第 3.1、3.2 节并提交。搜索单位为「契约 × 实际实现路径 × 输入条件 × 调用 / 状态场景 × 检查方法」。
预算按家族登记 CPU、GPU、接入人工、单例超时四项，首个小规模试跑后按第 6 节规则估算剩余预算。

### 3.1 候选清单（G4，2b 运行之前提交）

机器可读：`results/essential/phase2b/candidates.json`，由 `scripts/essential/p2b_registry.py` 生成。每个候选登记库与版本、API、设备、
dtype、申请的后端、所在环境、是否可取得 TTIR、与其他候选共享的源码 / 分解规则 / 后端；实际运行的后端（SDPA 选中的后端、Inductor
生成的 kernel 或回退）在运行时记录。回退到同一实现不算新的独立覆盖；多个包装入口调用同一内核算多个 API 条件。环境不可用：
DeepSpeed、apex、独立的 flash-attn、megatron-core（未安装，状态记为「环境不可用」；MoE 的 Megatron 风格参考在仓库内书写）。

### 3.2 覆盖矩阵（G3，2b 运行之前提交）

机器可读：`results/essential/phase2b/coverage_plan.json`。每个家族登记契约来源、要查的语义点、因素与水平；计划组合 = 每个因素的
单因素边界（其余因素取第一水平）+ 前两个因素的两两组合 + 登记的高风险多因素组合；每个计划组合 3 个 seed。F 与模式 B 的 FR
在规格交付前关闭（训练程序层的累加窗口已有 `specs/phase1/spec_accumulation.py`）。

| 家族 | 组 | 计划组合 | 候选 | FR |
|---|---|---|---|---|
| matmul_linear | basic | 19 | 9 | mode A (Inductor; matmul itself is cuBLAS/extern unless fuse |
| reductions | basic | 54 | 10 | mode A (Inductor, Liger) |
| activations | basic | 27 | 11 | mode A |
| gather_layout | basic | 18 | 9 | mode A |
| checkpoint | recompute | 12 | 4 | not applicable (property check, no f needed) |
| embedding | embedding | 13 | 7 | mode A |
| attention | attention | 27 | 11 | mode A for Inductor / flex (Triton); SDPA backends are CUDA  |
| packing | attention | 12 | 3 | mode A (flex) |
| rope | attention | 7 | 4 | mode A (Liger, Unsloth) |
| normalization | norm | 25 | 11 | mode A |
| optimizers | optimizer | 31 | 11 | mode A (compiled step) |
| schedulers | optimizer | 12 | 3 | not applicable |
| clip_amp | optimizer | 16 | 5 | not applicable |
| training_program | program | 21 | 6 | not applicable |
| moe | moe | 10 | 3 | mode A (compiled) |

第二档进入扩展队列（KLDiv batchmean、BCEWithLogits pos_weight、CTC、adaptive pooling 的 bin 边界、interpolate align_corners、conv same
padding）；量化训练的缩放取整与 selective scan 的 chunk 边界登记为**延期**；不查：dropout 的随机性（只在固定 mask 下查 1/(1−p)
缩放）、autocast 的 dtype 策略、推理期 KV cache 量化。执行顺序按规格交付的预期收益：优化器、注意力与打包、训练程序层、归一化、
嵌入、调度、裁剪，基础算子与 MoE 随后；每个家族先小规模试跑并按第 6 节估算预算。

## 4 预登记的决定规则（任务书第三部分）

- 若 C 显示加入 D 类检查后，轻量方法在全部反向错误上与 F 效果相同，则论文侧重为前向共有错误、归因成本与陌生组合的复用；2b 的
  内容不变。
- 若 F 在 2b 中仍有轻量方法发现不了的情形，写「独立规格的独有价值」，并逐例说明轻量方法为何漏掉。
- G6 的结果独立成节：自动完成的比例与成本，决定「参照复用」这一主张能写到什么程度。两种结果都如实报告。

## 5 停止条件（按影响范围，任务书 G9）

1. 共同的测量 / 评分链失效（来源绑定、比较入口、分类逻辑）：暂停其影响范围内的全部运行，上报。
2. A1 修复后第一阶段结论变化：保留冻结记录、说明原因、重核相关数据，不阻断无关家族。
3. 规格在审阅中被发现实质错误：只暂停依赖该规格的 F / FR。
4. 单个家族超预算：记录部分完成，继续其他家族。
5. 疑似内存安全或崩溃类问题：立即隔离并上报，不等批次结束。

## 6 预算估算规则

每个家族先做一次小规模试跑（≤ 10 个条件 × 全部候选），记录 CPU 秒、GPU 秒、接入人工（行数与小时）、最慢单例；按「完整条件数 /
试跑条件数 × 试跑耗时 × 1.5」估算，超出单家族上限（GPU 8 小时、CPU 32 核·小时、单例 600 秒）时先缩减组合（保留单因素边界与登记
的高风险组合），并记录缩减。

## 7 计划项的状态

所有计划项取以下状态之一：已执行且可裁决、仅部分方法可执行、不适用、未建立、环境不可用、超预算、未运行。不以找到若干 bug 作为
覆盖完成标准；不以多个包装接口或多个 seed 冒充独立实现或独立问题；最终报告区分预注册成绩与事后能力。

## 8 交付物

见任务书「交付物」。

## 9 偏离记录

1. **（S3 运行之前）** `scripts/essential/conditions_supplement.py` 没有随本文一起提交，而是在 S3 的第一次运行之前提交（池化 108、
   index/scatter 114 个条件：数值 {1e30 级, 1e−30 级, 1e−30 到 1e30 的混合} × 布局 {步长视图, 非零 storage offset, 转置存储} × 两组
   参数 / 较大的非整块尺寸；prod 只用 1e−30 级数值，以免乘积越出浮点范围）。运行入口用环境变量 `ESSENTIAL_SUITE` 切换条件集与缓存
   （`phase1` 默认、`s2`、`s3`），第一阶段的代码路径与结果不变。
2. **（A3 运行之后、使用之前）「接口 / 常量来源已核实」的判定**：对第一阶段 FR 中出现接口项的 Inductor 条件重新捕获一个 seed，
   用工具的 `interface.inventory` 清点生成的 kernel：存在经舍入的编译期常数或运行时标量（如 float32(ε)、float32(ε/C)、float32(1/除数)）
   即记为「已核实」，并列出这些常数；否则记为「未核实」。这是可指认的来源，不是逐常数的反事实重算。结果 404 个条件中 362 个已核实、
   42 个未核实（`results/essential/phase2a/a3_constants.json`）。
3. **分类与汇总的输出目录。** `classify.py --out`、`summarize.py --dir`：2a 中用新分类层（A2、A3）重新分类第一阶段数据时写入
   `results/essential/phase2a/reclassified/`，第一阶段目录不动。
4. **（S3 结果出来之后定的裁决规则，事后）index/scatter 前向在极端数值上超出 τ₃₂ 的元素**：对每个超出 τ₃₂ 的元素，若其项含正负
   两号、且 |K − f| ≤ γ_k·Σ|项|（mean 再除以项数；γ_k = k·u/(1 − k·u)，u = 2⁻²⁴，k = 项数 + 2）——float32 递归求和在任意顺序下的
   先验误差界——则记为「数值（抵消）」，不记为语义错误。该规则在看到 S3 结果之后制定，只用于 S3 的 index/scatter 前向；第一阶段
   与 S3 的分类文件本身不改，裁决写在 `results/essential/phase1_supplement/s3/numeric_bound.json`
   （`scripts/essential/s3_numeric_bound.py`）。
5. **（S2 首次评估之后、报告之前）** `s2_props.py` 的「被忽略行改变时 loss 逐位不变」把两边都是 NaN（全部行被忽略、0/0）的 loss
   判为不等，每个候选都得 71/552 个假违反；改为 NaN 视作相等后为 0/552。
6. **（2a 汇总时）** `summarize.py --dir` 在提交 6244f83 中写错（`global` 声明在使用之后，脚本无法运行）；改为局部变量，第一阶段
   的输出不受影响。
7. **（2b 优化器家族，运行之前定、运行中修正一处）** 条件取覆盖计划（`coverage_plan.json` 的 31 个组合）中去掉因素 `impl` 后的 17 个
   （每条实现路径都是候选、跑全部条件，覆盖计划中的每个（impl, 其余因素）组合因此都被执行），加上每个优化器一条登记的状态序列 1
   （init → 正常 → 零梯度 → 恢复）；状态序列 2、3 即覆盖计划的 save_restore 与 nonfinite_skip。P 的辅助运行：`mirror`（maximize 取反、
   梯度取负）、`uninterrupted`（save/restore 条件不做 save/restore）、`removed`（非有限跳过的条件去掉被跳过的一步）；两组参数（第二组
   lr × 2、无权重衰减）；梯度只取决于（状态, seed），不同优化器看到同一序列。试跑：全部 eager 候选跑完整个网格不到 1 分钟，试跑与
   完整运行重合；编译候选先试跑 3 个条件（16 秒）再完整运行（86 秒）。不支持的组合记为「不支持」：torch 2.10 的 RMSprop、Adafactor
   没有 fused；bnb `AdamW32bit` 与 torchao `_AdamW` 只接 AdamW（无 amsgrad、无 maximize）；HF Adafactor 与 `torch.optim.Adafactor` 是
   不同算法，E 不适用、只做 P。编译的 Adafactor 在 `step.item()` 处断图、实际以 eager 运行：不算独立覆盖，模式 A 的 FR 记为「未建立
   （没有 Triton 启动）」。运行中修正：非有限跳过的条件第一次运行时没有经过 `scaler.scale()`，GradScaler 的缩放未初始化而报错
   （每个候选 9 个运行）；改为用 `scaler.scale(g)` 生成梯度后全部重跑，结果只用重跑。预注册之外的性质（步数计数、wd = 0 时 Adam 与
   AdamW 相同、SGD 第一步动量缓冲等于有效梯度、Adafactor 行 / 列二阶矩均值一致、全部有限）标为「事后」，单独报告。
8. **（2b 注意力与打包，运行之前定、运行中修正一处）** 注意力：条件取覆盖计划去重后的 24 个；掩码一律左上对齐（SDPA 文档的
   `tril(diagonal=0)`），fully_masked_row 为第 2 个样本的第 0 行与第 Lq // 2 行，滑动窗口宽 5；没有任何允许键的行视为规格无定义——
   不进入 E 与 P，按候选记录约定，上游梯度在这些行上为零；候选或参照在这些行上产生 NaN 并进入 dk / dv 时，该条件的 dk / dv 不做 E
   比较（约定已记录）。P 的辅助运行：`repeat`（判定逐位性质前确认可重复）、`perturb`（被掩码键与值换成 10 倍随机值）、`v_ones`、
   `gqa_repeated`；预注册之外加了「对所有行都被掩码的键梯度为零」（标为 P_dead_keys_zero_grad，同属掩码性质，单独列出）。bf16 的 E
   只记录，报告相对 2⁻⁸(1 + |K_eager|) 的倍数。为 E 增加只作参照的 eager：手写注意力 eager、未编译的 flex_attention（打包家族：SDPA math、
   未编译的 flex、HF eager）。运行中修正：xformers 在 GQA 上「No operator found」第一次被记为错误，改记「不支持」后重跑该候选（数值结果
   不变）。打包：条件取覆盖计划 (docs, lengths) 的 9 个组合（`impl` 去掉，每个实现都是候选）；HF 候选用随机权重的 2 层 Llama 而不是
   预训练模型（只查文档隔离与位置重置）。
