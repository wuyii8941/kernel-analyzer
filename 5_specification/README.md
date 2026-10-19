# 规格（specification）

规格只存放在这里：任务规格 f 的方法、来源登记、合同、可执行参照、证据与审阅记录。2026-10-15 从 `1_experiments/specs`
迁来，同时换成 v2.4。规格语义只经审阅方修改，执行方不改。

## 1 规格在工具里管什么

工具比较三个量：K 是设备输出，Gπ 是程序按其声明语义的精确值，f 是任务要求的值。

| 差异 | 回答什么 | 参照从哪来 |
|---|---|---|
| K − Gπ（数值偏差） | kernel 是否按它自己的代码在算：舍入、累加精度、近似指令、归约次序 | DSL v2 rc3 从 TTIR 自动提升，不需要本目录。定义见 [design_rc3](../1_experiments/dsl_v2/design_rc3/README.md)，代码在 `2_tool/src/kernel_analyzer/reference_eval/` |
| Gπ − f（语义偏差） | kernel 写的是不是任务要的计算：公式、归一化、掩码、状态语义 | 本目录 |

- 精度审查只用 Gπ。没有 f 时只报告 K − Gπ，并写明「无语义参照」。
- Gπ 不是把一切换成实数：π 声明的量化、位语义、允许集合都保留。

## 2 f 怎样产生

三个对象，逐级可追溯：

```
来源原文      SOURCES.json                     版本钉死的文档，按条款编号
  → 数学合同  contracts/*.yaml                 条款、合法域、profile、输出与保证、例子、误读
  → 可执行参照 adapters/ + kernel_spec/           合同适配、输入验证与参照求值
                specification_v2/algorithm_specs/  适配器仍调用的 v2.4 参照
```

- **条款依据**：只能是原文条款编号（如 `PCE.REDUCE`），或一条登记过的决定（如 `CE.R_A`）。
- **决定的两种状态**：
  - 已审阅；
  - 需任务绑定：任务负责人的选择，工具不能替用户选。例如 RANKMEAN 的有效 token 等权、CE 加权 label smoothing 的 R_A 读法。
- **原文没规定的行为**：只能记为未规定、允许集合或任务选定的约定，不能为了让某个实现通过而事后补读法。
- **保证标签**：每个输出带一个，取 exact、rigorous_enclosure、allowed_set、declared_approximation 之一。
- **比较结果**：相容、矛盾、未建立分别记录。开发样例的区间相容不证明数学相等；完整输出判定若只有非零宽度区间包含候选值，仍返回未建立。
- **实现准入**：`activation.implementation_status` 必须明确为 `validated`，且 `validated/` 中由独立复核保留的完整合同和适配器必须与当前文件相同。开发检查只在 `contracts/DEVELOPMENT_CHECKS.json` 记录日期和数量，不参与准入；`allow_draft=True` 也不能代替实现复核。这里直接比较可阅读的原文件，不使用内容哈希。参照查询结果仍不等于完成真实 kernel 绑定。
- **任务选择**：`needs_binding` 决定必须由相关 profile 的 `basis` 显式引用；直接或间接依赖它的 profile 必须要求任务选择。需要调用者选择的 profile 使用 `task_bindings={profile: "任务选择的具体声明"}`；布尔值、数字、空白字符串和未知 profile 均被拒绝。声明原文保留在结果的 `task_choice` 中，它不是来源审阅或 kernel 绑定证据。`allow_draft` 只接受布尔值；字符串 `"false"` 不会启用草案。参照查询保留 `semantic_verdict=not_established`，完整绑定后的语义判断另行完成。
- **结果类别**：value、domain_undefined、invalid_input、unsupported、not_established、implementation_error。
- **三种「通过」分开记**：合同内容已审阅、参照实现已验证、本次查询可用。前一项推不出后一项。已审阅合同若仍包含 drafted 的语义决定，结构检查与运行时准入都会拒绝；allow_draft 不能消除这种状态冲突。
- **证据**：关系检查、独立复算、ONNX Runtime 对照、历史误读重建都按条款登记。它们不决定条款状态，但可信的矛盾会暂停相关条款。

完整口径见两份原文：[METHOD.md](docs/METHOD.md)（方法定稿）和 [CONTRACT_PROTOCOL.md](docs/CONTRACT_PROTOCOL.md)（审阅后的实施口径 v1.1）。

## 3 现状（2026-10-19）

最新外部调用批次暴露的 SmoothL1/Huber 缺口已补入 HUBER 合同，覆盖声明范围内的完整前向和
两侧梯度；旧8次调用已用数学参照复核。它们仍是开发证据，未增加正式绑定数。
见[批次结果](corpus/external_batch/README.md)与[来源及实现核验](evidence/huber_implementation.md)。

该批中的 FlagGems MSE 已进一步完成三种 reduction 的完整前向绑定：内部累加器初始化可追溯，
每种24次测量都有完整参照；mean 的系数差有独立精确推导，sum 的统计判断保留未知。
见[实际绑定证据](evidence/flaggems_mse_binding.md)。其他批次对照项不随之升级。

FlagGems 公开默认 RMSNorm 的前向和完整 VJP 也已完成调用绑定。24次测量均有完整 K−f；
y/dx 另有程序分解，原生 sum 产生的 dweight 仍没有 Gπ。实际 epsilon 导致的前向缩小有独立
公式界，范围和限制见[绑定与测量证据](evidence/flaggems_rms_norm_binding.md)。

最新公开调用检查中，nearest 插值因缺少完整参照而新增了有版本边界的 INTERP profile；argmax
则直接复用已有 REDUCE 首索引语义，没有重复建立公式。两批均用另一组实现和未见参数验收；
FlagGems 的非连续 nearest 前向与非连续全张量 argmax 均暴露了忽略 stride 的错误，argmax 另有
一个非二次幂中间维编译失败。范围和原始记录见[外部调用批次](corpus/external_batch/README.md)。

随后真实 Tensor-list `torch.cat` 复用了已有拼接前向，并补齐每个输入的有序 VJP。官方 PyTorch 2.10
定义、独立坐标转置和保留的 autograd 选择互相核对。FlagGems 开发10次与另一实现 TorchInductor 的
陌生验收10次都完整精确，覆盖空输入、零宽轴、五／六输入分组和非连续布局；这仍不增加正式绑定或
统计 bias 结论。见[LAYOUT 核验](evidence/layout_implementation.md)与[外部调用记录](corpus/external_batch/README.md)。

真实标量 `torch.clamp` 又补齐了版本化边界梯度：PyTorch 2.10 明确在 min/max 处选择子梯度1，独立
逐坐标公式与四种 dtype 的公开 autograd 完全一致。冻结的 FlagGems 开发10次全部精确；另一组
TorchInductor 验收保留了2次调用历史相关的前向参数复用错误，最小无记录器程序也复现，规格没有为
候选结果改写。见[ARITH 核验](evidence/arith_implementation.md)与[原始记录](corpus/external_batch/README.md)。

同一轮从真实公开 `torch.maximum/minimum` 调用补齐了两个 Tensor operand 的完整 VJP。官方 PyTorch
2.10 autograd 在相等值处把 upstream 各分一半，独立坐标转置再分别处理两个输入的广播归约。
FlagGems 开发10次和另一实现 TorchInductor 的陌生验收10次全部逐坐标精确，验收包含整张量相等的
强见证；范围不扩展到 reduction、fmax/fmin 或特殊值。见[ARITH 核验](evidence/arith_implementation.md)。

| 层 | 状态 |
|---|---|
| 可执行参照 v2.4 | 31 个角色都有参照代码，另有 SPARSE、STATEIO 两条路径规格；222 项测试通过；8 个角色修过 v1 的错误 |
| 合同 | 当前顶层目录共 62 份、127 个 profile：ACT、ADAFACTOR、ATTN、BN、BSR、CE、ADAM、ADAMW、ARITH、CAST、CLIP、CONV、DIVERG、EMB、F2_MX、OPT4BIT、HUBER、INDEX、INDEXREDUCE、INTERP、L1、LAYOUT、LOGIC、LORA、PACK、PAGEDATTN、MATHFN、MATMUL、MSE、MOE、POOL、RANDOM、RADAM、RMSPROP、ROPE、RPROP、SAMPLING、SSU、REDUCE、RANKMEAN、SGD、SOFTMAX、SORT、NORM 为 `reviewed_for_implementation`，其余 18 份为 `drafted`（含迁入但未启用的 A1_LINEAR，以及只覆盖明确分片合并与基础 collective 数值任务的 K14_COLL）。F2_MX 只准入 torchao 0.16.0 的有限 f32/bf16、末维32项分块、E4M3/E5M2、FLOOR 和普通 scale 布局，不代表所有 MX 实现。开发例子通过不改变审阅状态；规格 v3 计划见 `docs/PLAN_v3.md` |
| 实现复核 | ACT、ADAFACTOR、ATTN、BN、BSR、CE、ADAM、ADAMW、ARITH、CAST、CLIP、CONV、DIVERG、EMB、F2_MX、OPT4BIT、HUBER、INDEX、INDEXREDUCE、INTERP、L1、LAYOUT、LOGIC、LORA、PACK、PAGEDATTN、MATHFN、MATMUL、MSE、MOE、POOL、RANDOM、RADAM、RMSPROP、ROPE、RPROP、SAMPLING、SSU、REDUCE、RANKMEAN、SGD、SOFTMAX、SORT、NORM 已在各自声明范围内记为 `validated`，当前合同和适配器均已完成独立复核留存，共六十五个 profile 可作为规格参照。INDEXREDUCE 只准入物理有限同 dtype 浮点张量、int64 index 和可表示 alpha 的函数式 index_add 前向值；原地更新、index_reduce、scatter_reduce 与反向仍拒绝。RANDOM 只准入无需复现随机流的 dropout 前向边界值，以及显式 Boolean mask 和 scale 的 native dropout backward；随机训练前向继续拒绝。CLIP 只准入一个物理有限梯度缓冲区的 clip_grad_value_ 完整写回，范数裁剪、多缓冲区未完整对应、非正 clipping bound 与特殊值继续拒绝。SGD 只准入一个稠密实数参数的一步公开状态转移，完整返回参数与 momentum buffer；内部 fused 接口、AMP overflow、可微执行与多参数编排继续拒绝。ADAM 只准入一个稠密实数参数的一步公开 coupled Adam 状态转移，完整返回参数、两个矩、可选 AMSGrad 最大矩和 step；AdamW、内部 fused/AMP、capturable、可微和多参数任务继续拒绝。RMSPROP 只准入一个稠密实数参数的一步公开状态转移，完整返回参数、square average、可选 centered average、可选 momentum buffer 和 step；capturable、可微和多参数任务继续拒绝。RADAM 只准入 PyTorch 2.10 的单参数稠密实数一步状态转移，显式区分耦合/解耦 decay 和 rho>5 分支，完整返回参数、两个矩和 step；capturable、可微、稀疏、复数和多参数任务继续拒绝。ADAFACTOR 分别准入 PyTorch 2.10 的单参数向量与末两维分解状态转移，保留 row/col 单位维、延迟状态和 dtype 决定的 eps1，完整返回参数及相应方差状态和 step；其他 Adafactor 实现、稀疏、复数和多参数任务继续拒绝。RPROP 只准入 PyTorch 2.10 的单参数稠密实数一步公开状态转移，完整返回参数、previous gradient、step size 和 step；负数或次序颠倒的 step-size bounds、capturable、可微、稀疏、复数和多参数任务继续拒绝。ATTN 分别准入 PyTorch 2.10 的有限稠密 rank-3/rank-4、dropout=0 且每行至少有一个有限有效 score 的公开 SDPA 完整前向，以及相同任务带完整 upstream 的 out、dq、dk、dv VJP；FlashAttention bottom-right causal、window/softcap、FlexAttention、内部 LSE、全 mask 空行和其他后端接口继续拒绝。PAGEDATTN 只准入安装的 vLLM 0.16.0 causal、未量化、二维输出路径，完整核对分页表、bottom-right 位置、左窗口、GQA、scale、softcap 与输出；双向、量化、分段 softmax、附加 bias 和其他版本继续拒绝。PACK 只准入 FlashAttention v2.8.3 的 f16/bf16 非分页 varlen 前向，按累计长度分隔序列，完整核对 bottom-right causal 或 bidirectional、连续 GQA、scale 与输出；mask/position_ids 构造、dropout、window、softcap、ALiBi、分页、额外输出和反向继续拒绝。BN 只准入显式 running mean/variance、正 eps 和可选仿射参数的物理有限 BatchNorm 推理前向值；训练、状态更新、native 多输出与反向仍拒绝。INDEX 只准入物理有限浮点数据与 int64 索引上的默认只读 gather、index_select 和 embedding 前向；scatter、更新、反向累加、稀疏与非默认 embedding 选项仍拒绝。EMB 只准入 PyTorch 2.10 的有限稠密 functional embedding 前向和完整 dense weight VJP，支持合法 padding_idx 与按完整索引计数的 scale_grad_by_freq；embedding_bag、max_norm、稀疏梯度、模块初始化和布局效果继续拒绝。ROPE 只准入 Transformers 4.57.3 Llama 的 supplied-table rotate-half q/k 前向和完整 q/k VJP；表生成、position lookup、partial/interleaved 布局、其他模型族和 cos/sin 梯度继续拒绝。MOE 只准入 Mixtral 的无并列 top-2 路由与已提供 expert outputs 的加权组合；capacity、丢弃 token、router noise、auxiliary loss、expert MLP 和分布式 dispatch 继续拒绝。BSR 只准入 PyTorch 2.10 二维 canonical BSR 的 softmax value core；稀疏张量分配、batch/hybrid、矩阵乘、sampled_addmm 和梯度继续拒绝。SAMPLING 只准入 vLLM 0.16.0 在有限无并列 float32 logits 上的确定性 top-k 后 top-p 完整 keep mask；并列、masked logits、概率、随机抽样和替代 backend 继续拒绝。LORA 只准入 vLLM 0.16.0 的有限 f16/bf16 LoRA A 投影与完整 float32 输出状态；expand、完整 B*A 更新、fused-MoE LoRA、任意 metadata 和其他输出 dtype 继续拒绝。SSU 只准入 mamba-ssm 2.3.2.post1 在 D 和 dt_bias 均存在时的 float32 单步选择性状态更新，完整返回 y 与 state 后态；论文 ZOH、该版本不能启动的空 D/dt_bias、重复活动状态槽、序列扫描与反向继续拒绝。F2_MX 只准入 torchao 0.16.0 的有限 f32/bf16 输入、末维32项一块、E4M3/E5M2、FLOOR 和非 swizzled scale，完整返回 scale code 与全部元素；RCEIL/CEIL/EVEN、FP4/FP6、任意轴、尾块、特殊值和其他版本继续拒绝。DIVERG 只准入 PyTorch 2.10 的二维物理 f32 probability target、batchmean、log_target=false 及完整 input VJP；其他 reduction、log target、target VJP、非正 target、其他秩和 dtype 继续拒绝。MSE 准入 PyTorch 2.10 无权重、同形状、非空物理 f32 的 none/sum/mean loss 和完整 input/target VJP；原 mean profile 范围不变。广播、加权、其他 dtype 与特殊值继续拒绝。L1 准入同一物理范围的无权重 none/sum/mean absolute error 和完整 input/target VJP，并在相等坐标采用 PyTorch 的零子梯度；原 mean profile 范围不变。广播、加权、复数与特殊值继续拒绝。OPT4BIT 只准入 torchao 0.16.0 的物理 OptimState4bit float32 state-copy，完整核对 packed codes、block scales 和固定 qmap 前态；完整 AdamW4bit 步骤、特殊值、子正规数和替换 qmap 继续拒绝。POOL 分别准入物理有限浮点的二维 max_pool2d 前向值、唯一最大值下显式选择的 PyTorch 2.10 values+int64 indices+完整输入 VJP，以及单独选择的二维 avg_pool2d 完整前向和输入 VJP；并列最大值、自适应池化及未声明参数组合继续拒绝。INTERP 只准入 PyTorch 2.10 rank-4 显式 size 的 legacy nearest 前向与完整输入 VJP，坐标约定需要显式任务选择；nearest-exact、scale-factor 调用、其他秩和插值模式继续拒绝。LOGIC 只准入物理有限的同 dtype 浮点或布尔常用比较、逻辑、布尔位运算和 truth reduction，以及独立的浮点 `isnan` profile；整数位运算、混合 dtype、NaN 比较和未选重载仍拒绝。ARITH 准入有限、同 dtype、值能由声明 dtype 精确表示的 f16/bf16/f32/f64 常见逐元素操作，并单独准入 PyTorch 2.10 标量 `masked_fill` 的完整公开前向和输入 VJP、Tensor/Tensor `where` 的完整前向和两个数据 VJP，标量 bound `clamp` 的完整前向和输入 VJP，以及 Tensor/Tensor `maximum/minimum` 的完整前向和两个数据 VJP；函数式状态由真实调用检查。整数、混合 dtype、特殊值、`pow`、其他重载和未声明梯度仍拒绝。CAST 只准入有限 f16/bf16/f32/f64 之间明确选择 roundTiesToEven 的转换，输入记录必须能由声明的源 dtype 精确表示，整数、布尔、float8、非有限值、目标溢出和内存选项仍拒绝。LAYOUT 准入有限浮点张量的常见函数式逻辑值和形状变换，并单独准入 PyTorch 2.10 constant pad 的完整公开前向和输入 VJP，以及 Tensor-list cat 的完整前向和全部有序输入 VJP；非 constant pad 模式、未声明梯度、存储、stride、别名、创建、dtype 转换和内存效果仍拒绝。MATHFN 只准入绝对值不超过100、满足相应实数域的有限 f16/bf16/f32/f64 公共逐元素函数；整数、无穷和奇点边界不准入。REDUCE 只准入有限浮点的公共 sum、mean、prod、extrema、cumsum、logsumexp，以及 correction 为0或1且自由度为正的 `torch.var`、`torch.std`、有序 `torch.var_mean` 与已声明 VJP，以及实数 f32/f64 的公开 `torch.linalg.vector_norm` 二范数和正范数处 VJP；多元素零方差切片的 std VJP、其他 variance/std/var_mean correction 与重载、已弃用 `torch.norm`、其他 vector-norm order/dtype 和零范数 VJP 仍拒绝；CONV 只覆盖批量非转置 1-D/2-D 有限浮点默认操作；MATMUL 只覆盖有限 f16/bf16/f32/f64 的默认函数重载；RANKMEAN 每次使用仍须明确选择 token 等权任务。证据见 [ACT](evidence/act_implementation.md)、[Common activations](evidence/inductor_common_activations_binding.md)、[BN](evidence/bn_implementation.md)、[CE](evidence/ce_implementation.md)、[Adam](evidence/adam_implementation.md)、[AdamW](evidence/adamw_implementation.md)、[ARITH](evidence/arith_implementation.md)、[CAST](evidence/cast_implementation.md)、[CLIP](evidence/clip_value_implementation.md)、[CONV](evidence/conv_implementation.md)、[KL batchmean](evidence/diverg_kl_implementation.md)、[MSE](evidence/mse_implementation.md)、[MSE Inductor binding](evidence/inductor_mse_loss_binding.md)、[L1](evidence/l1_implementation.md)、[L1 Inductor binding](evidence/inductor_l1_loss_binding.md)、[OptimState4bit](evidence/opt4bit_state_copy_implementation.md)、[Embedding](evidence/embedding_implementation.md)、[RoPE](evidence/rope_implementation.md)、[MOE](evidence/moe_implementation.md)、[BSR softmax](evidence/bsr_softmax_implementation.md)、[SAMPLING](evidence/sampling_filter_implementation.md)、[LORA](evidence/lora_shrink_implementation.md)、[SSU](evidence/ssu_implementation.md)、[INDEX](evidence/index_implementation.md)、[INDEXREDUCE](evidence/index_add_implementation.md)、[INTERP](evidence/interpolate_implementation.md)、[LAYOUT](evidence/layout_implementation.md)、[LOGIC](evidence/logic_implementation.md)、[MATHFN](evidence/mathfn_implementation.md)、[MATMUL](evidence/matmul_implementation.md)、[POOL](evidence/pool_implementation.md)、[RANDOM](evidence/random_deterministic_implementation.md)、[Adafactor](evidence/adafactor_implementation.md)、[Attention](evidence/attention_implementation.md)、[Paged attention](evidence/paged_attention_implementation.md)、[Packed varlen attention](evidence/pack_implementation.md)、[MXFP8 FLOOR](evidence/triton_mxfp8_e5m2_floor_binding.md)、[RAdam](evidence/radam_implementation.md)、[RMSprop](evidence/rmsprop_implementation.md)、[Rprop](evidence/rprop_implementation.md)、[REDUCE](evidence/reduce_implementation.md)、[RANKMEAN](evidence/rankmean_implementation.md)、[SGD](evidence/sgd_implementation.md)、[SOFTMAX](evidence/softmax_implementation.md)、[SORT](evidence/sort_implementation.md)、[NORM](evidence/norm_implementation.md)。其余 18 份仍待实现验证。这不升级真实 kernel 绑定 |
| 唯一运行入口 | `kernel_spec.registry.query` 检查合同准入、来源依据、profile、完整输出字段及返回值的表示保证。`allow_draft=True` 只允许带草案标签的开发查询，不授予认证 |
| 完整输出比较 | `registry.check_output` 先走上述准入，再调用合同专用比较器。SORT 在有限且无并列值的范围内检查全部排序值和原始索引，ADAMW 检查全部后态的精确关系，RANKMEAN 检查选定的全局均值，CE 检查全部 loss/VJP；ARITH、ADAFACTOR、ATTN、BN、CAST、CLIP、CONV、DIVERG、EMB、F2_MX、OPT4BIT、INDEX、INDEXREDUCE、INTERP、L1、LAYOUT、LOGIC、LORA、PACK、PAGEDATTN、MATHFN、MATMUL、MSE、MOE、POOL、RANDOM、RADAM、RMSPROP、ROPE、RPROP、SAMPLING、SSU、REDUCE、SOFTMAX 和 ACT 检查完整张量及 dtype/shape，SGD 要求参数与 momentum buffer 两份完整后态，ADAM 要求参数、两个矩、可选 AMSGrad 最大矩和 step 五项完整后态，RMSPROP 要求参数、square average、可选 centered average、可选 momentum buffer 和 step 五项完整后态，RADAM 要求参数、两个矩和 step 四项完整后态，RPROP 要求参数、previous gradient、step size 和 step 四项完整后态，ADAFACTOR 按参数秩要求参数、完整 variance 或保留单位维的 row/col 状态和 step，INDEX 还核对完整索引结果，LOGIC 还严格要求布尔 dtype 与布尔叶值，LAYOUT 还检查 split 的全部有序结果，REDUCE 还要求 max/min 的值和索引同时存在，CONV 与 MATMUL 核对全部梯度位置和显式空位置，ACT 要求两个 GLU 梯度同时存在。MOE 还核对同一次路由选择下的 Boolean mask、稠密权重和组合结果；草案中的 A1、K14 已有完整结果比较，但不提升准入状态。非零宽度区间包含候选值时返回未建立。没有比较器的合同不自动逐元素比较；所有比较仍以调用方提供的任务为前提 |
| 陌生算子试跑 | FlagGems sort、Liger RMSNorm、Liger 4-D RoPE 与 Liger 原地 CrossEntropy 的第一次试跑分别暴露了输出类型、公开 autograd 边界、批量表形状和输入后态遗漏；修复后均能完成固定输入比较。另以24个输入分别测试 Inductor CE、softmax、saved-output softmax backward、saved-output log-softmax backward、logsumexp forward+VJP、gather、cumsum forward+VJP、corrected variance forward+VJP、corrected standard deviation forward+VJP、ordered var_mean forward+combined VJP、public vector 2-norm forward+VJP、mean MSE loss+完整 input/target VJP、mean L1 loss+完整 input/target VJP、SiLU、ReLU/tanh/sigmoid forward+VJP、直接 Triton matmul、causal GQA SDPA、embedding forward+dense VJP、supplied-table Llama RoPE、Mixtral top-2 routing+combination、BSR softmax values、deterministic top-k/top-p keep mask、Mamba selective state update、LoRA shrink、causal paged attention、packed varlen attention、PyTorch KL batchmean loss+input VJP、PyTorch Inductor LayerNorm forward+VJP、PyTorch Inductor RMSNorm forward+VJP、PyTorch Inductor GroupNorm forward+VJP、PyTorch Inductor exact-form GELU forward+VJP、grouped Conv2d、行求和、逐元素 exp、显式 bf16 转换、逐元素乘加、矩阵转置、大于零比较、行 index-select、无并列值行排序、max_pool2d、BatchNorm 推理、dropout backward、index_add、梯度值裁剪、warm-buffer SGD、warm-state Adam、warm-state RMSprop、warm-state rectified RAdam、warm-state factored Adafactor 和 warm-state Rprop：其中三十五项的公式注入均被规格差检出，并与普通数值差分开；连同 MXFP8 E5M2 FLOOR 和 torchao OptimState4bit state-copy、FlagGems MSE 前向、公开默认 RMSNorm 完整 VJP 和 Inductor 二维平均池化完整 VJP，五十八个正常调用现已完成逐项绑定审核及正式测量。布尔比较的正常版本在24,576个坐标上零误判；阈值改成0.25后检出2,474个假阴性。行选择的49,152个基线坐标逐点相等，加入0.25后全部 Gπ−f 坐标排除零。行排序的49,152个值与索引基线坐标逐点相等，排序值加入0.25后完整 K−f 规则确认非零平均作用；其 bitonic 归约尚无程序参照，因此 K−Gπ 保留未建立。max_pool2d 的9,216个基线坐标在 K−Gπ、Gπ−f 和 K−f 上都为零，加入0.25后全部 Gπ−f 坐标排除零。BatchNorm 的正常程序公式与任务公式相容，设备执行保留约3.46e-8的平均绝对数值差；加入0.25后24,576个 Gπ−f 坐标全部排除零。dropout backward 的24,576个程序公式坐标与任务公式精确一致，设备执行平均绝对差约1.61e-8；加入0.25后全部 Gπ−f 坐标排除零。index_add 的98,304个程序公式坐标与任务公式精确一致，设备执行平均绝对差约9.04e-9；加入0.25后全部 Gπ−f 坐标排除零。梯度值裁剪的98,304个原地写回坐标在 K−Gπ、Gπ−f 和 K−f 上都为零；加入0.25后全部 Gπ−f 坐标排除零。SGD 的参数与 momentum buffer 各98,304个 Gπ−f 坐标都为零；只把 buffer 写回加0.25后，全部 buffer Gπ−f 坐标精确偏移0.25，而参数仍为零。Adam 的 theta、m、v、vmax 各98,304个程序语义坐标与 step 的24个坐标都与任务公式相容；只把 vmax 写回加0.25后，全部 vmax Gπ−f 坐标排除零，其他四项仍为零。RMSprop 的 theta、v、gave、b 各98,304个程序语义坐标与 step 的24个坐标都与任务公式相容；只把 b 写回加0.25后，全部 b Gπ−f 坐标排除零，其他四项仍为零。RAdam 的 theta、m、v 各98,304个程序语义坐标与 step 的24个坐标都与任务公式相容；预先把整流系数化成 host float 的第一次绑定被 Gπ−f 拒绝，改为在程序图中计算后相容；只把 m 写回加0.25后，全部 m Gπ−f 坐标排除零，其他三项仍为零。Adafactor 的 theta 有24,576个、row/col 各768个程序语义坐标，另有24个 step 坐标，均与任务公式相容；只把 row 写回加0.25后，全部768个 row 坐标排除零，theta、col 和 step 不误报。Rprop 的 theta、prev、eta 各98,304个程序语义坐标和24个 step 坐标均与任务公式相容；只把 eta 写回加0.25后，全部98,304个 eta 坐标排除零，theta、prev 和 step 不误报。Mamba selective state update 的 y 24,576个和完整 state 196,608个坐标均建立严格参照；基线 state 检出约1e-8平均绝对残差及小的正向执行偏差，只给最终 state 写回加0.25后全部196,608个 state 坐标排除正确参考，而 y 不误报。LoRA shrink 的24,576个完整输出坐标均建立严格参照；设备执行平均绝对差约2.31e-8，程序公式与任务相容，全部输出加0.25后所有坐标都排除正确参考并检出非零作用。Paged attention 的7,680个完整输出坐标均建立严格参照，程序公式与任务相容；设备执行平均绝对差约1.28e-8并在对齐方向上检出小数值作用，全部输出加0.25后所有 Gπ−f 坐标排除零。Packed varlen attention 的7,680个完整输出坐标均建立严格参照，程序公式与任务相容；完整设备到任务平均绝对差约4.02e-5，全部输出加0.25后所有 Gπ−f 坐标排除零且两类语义规则均检出。Causal GQA SDPA 的49,152个程序语义坐标均与任务公式相容；设备到程序的对齐规则检出小的负向数值偏差，只给最终输出加0.25后全部49,152个 Gπ−f 坐标排除零。Embedding 的49,152个 forward 和24,576个 dense VJP 程序坐标均有完整参照且与任务公式相容；只给非 padding 梯度行加0.25后，24,192个 Gπ−f 坐标排除零。Llama RoPE 的 q_out/dq 各49,152个、k_out/dk 各24,576个坐标均有完整参照且程序公式与任务公式逐点一致；只给两个梯度加0.25后，全部73,728个梯度坐标排除零而前向不误报。Mixtral top-2 的24个调用中，24,576个 Boolean 选择位零误判，权重与组合输出分别有24,576和98,304个完整参照坐标；只给组合输出加0.25后，全部98,304个 Gπ−f 坐标排除零。BSR softmax values 的98,304个基线坐标均有完整参照，程序公式与任务相容；只给写回加0.25后全部 Gπ−f 坐标排除零。确定性 top-k/top-p keep mask 的98,304个基线布尔坐标全部匹配；每行翻转 token 0 后恰有1,536个错位，程序公式与完整输出比较均检出非零作用。正常 `tl.exp` 还在声明分布上检出了很小的负向数值偏差，而任务公式差为零。KL batchmean 的24个 loss 和12,288个 input VJP 坐标都有完整参照；基线任务公式差为零或被严格区间包含，给两项公式加入0.25后全部坐标排除零。Mean MSE 的24个 loss 和 input/target VJP 各98,304个坐标均完整，程序公式与任务相容；设备到程序平均绝对差分别约2.74e-7和1.88e-11，当前规则未确认稳定方向。Mean L1 的对应坐标也全部完整；loss 的设备到程序平均绝对差约7.86e-8，两个梯度三层比较逐点为零并被统计层标为退化，而没有升级为总体无 bias。LayerNorm 的 `y`、`dx` 和 `dweight` 在设备到程序比较中检出正向平均作用；程序到任务比较另保留由 float32 eps 与 `rsqrtf` 带来的更小差异，`dbias` 两类比较均未检出。RMSNorm 的 `y`、`dx`、`dweight` 在设备到程序比较中均未检出稳定平均作用，程序到任务比较则检出更小的负向对齐差异；未检出不表示无 bias。GroupNorm 的 `y`、`dx` 在设备到程序比较中检出正向平均作用，`dweight`、`dbias` 未检出；程序到任务的更小差异继续单独报告。Exact-form GELU 首次因负尾部区间抵消拒绝，通用参照改用严格恒等式 `1+erf(x)=erfc(-x)` 后，同一批前向和梯度坐标全部达到分辨率，并分别检出设备执行与程序公式作用。Saved-output softmax backward 的49,152个梯度坐标全部达到分辨率，设备与程序间现有规则未检出稳定作用，程序公式与任务相容；未检出不表示无 bias。Saved-output log-softmax backward 的49,152个梯度坐标也全部达到分辨率，设备到程序平均绝对差约1.91e-8，程序公式与任务相容，现有规则未检出稳定作用。Logsumexp 的384个前向与49,152个 VJP 坐标全部达到分辨率，程序公式与任务相容；设备到程序平均绝对差分别约1.34e-7和9.42e-10，现有规则均未检出稳定作用。Gather 的18,432个输出坐标在 K−Gπ、Gπ−f 和 K−f 上都精确为零；统计层因零方差和零校准方向拒绝总体 bias 判断。Cumsum 的前向与 VJP 各49,152个坐标全部达到分辨率，跨3个 Triton kernel 的程序参照与任务公式逐点相容；设备到程序平均绝对差分别约4.27e-7和4.08e-7，当前规则均未检出稳定作用。 Corrected variance 的384个前向和49,152个 VJP 坐标也全部达到分辨率；前向程序公式与任务相容，反向因 `2/127` 的 float32 系数选择使全部 VJP Gπ−f 坐标排除零，平均绝对差约7.55e-11且对齐规则检出正向平均作用。 Corrected standard deviation 的384个前向和49,152个 VJP 坐标同样完整；前向设备到程序检出正向平均作用，反向程序公式的全部 Gπ−f 坐标排除零，平均绝对差约1.82e-11，R5检出负向且R2/R3检出正向平均作用。Ordered var_mean 的方差、均值和组合 VJP 均完整绑定；均值程序公式与任务精确相容，组合 VJP 的49,152个 Gπ−f 坐标因方差分支的 float32 `2/127` 系数全部排除零，R2/R3 检出正向平均作用。Public vector 2-norm 的384个前向和49,152个 VJP 坐标也全部建立；程序公式与任务相容，设备到程序平均绝对差分别约5.98e-7和2.55e-9，现有规则未确认稳定平均作用。其余四份外部库计划仍是草案。MXFP8 的安全输入在6,144个元素和192个 scale code 上逐项相等；完整有限范围中有363个坐标本应有限却写成正负无穷，最小尺度边界另有32个 0 与参考0.5的不一致。该结论只属于已绑定的 torchao 0.16.0 FLOOR 任务和 RTX A6000 路径。这些结果不构成总体无偏结论 |
| 工具接入 | `audit(spec=...)` 与 `check`/`measure` 的任意 `spec()` 仍只算外部实现参照，不能作语义裁决。`measure` 合同模式只从已审阅并独立保留的绑定计划、调用文件、合同和适配器取得 f；每次抽样重查前态、后态与全部启动，并在完整公开输出上统计 K−f。当前六十五个 profile 通过规格参照准入，其他62个 profile 拒绝。六十二份真实调用计划中，Inductor 末维 softmax、Inductor saved-output softmax backward、Inductor saved-output log-softmax backward、Inductor logsumexp forward+VJP、Inductor gather、Inductor cumsum forward+VJP、Inductor corrected variance forward+VJP、corrected standard deviation forward+VJP、ordered var_mean forward+combined VJP、public vector 2-norm forward+VJP、融合 mean CrossEntropy loss+VJP、mean MSE loss+完整 input/target VJP、mean L1 loss+完整 input/target VJP、Inductor SiLU、Inductor ReLU/tanh/sigmoid forward+VJP、直接 Triton float32 matmul、causal GQA SDPA、embedding forward+dense VJP、supplied-table Llama RoPE、Mixtral top-2 routing+combination、BSR softmax values、deterministic top-k/top-p keep mask、Mamba selective state update、LoRA shrink、causal paged attention、packed varlen attention、PyTorch KL batchmean loss+input VJP、PyTorch Inductor LayerNorm forward+VJP、PyTorch Inductor RMSNorm forward+VJP、PyTorch Inductor GroupNorm forward+VJP、PyTorch Inductor exact-form GELU forward+VJP、MXFP8 E5M2 FLOOR、grouped Conv2d、行求和、逐元素 exp、显式 bf16 转换、逐元素乘加、矩阵转置、大于零比较、行 index-select、无并列值行排序、max_pool2d、BatchNorm 推理、dropout backward、index_add、梯度值裁剪、warm-buffer SGD、warm-state Adam、warm-state RMSprop、warm-state rectified RAdam、warm-state factored Adafactor、warm-state Rprop 与 torchao OptimState4bit state-copy、FlagGems MSE 前向、FlagGems 公开默认 RMSNorm 完整 VJP 五十八份已审阅并通过正式测量，另外四份仍为草案；生产覆盖只限这五十八个明确调用 |

这些绑定只认证声明的具体调用；尚未完成固定真实工作负载集合的全面验收，不能把合同实现复核数量算作真实 kernel 覆盖率。角色清单见 [roles.md](docs/roles.md)；已知问题、待决事项与版本沿革见
[status.md](docs/status.md)。

## 4 目录

| 路径 | 内容 | 来源 |
|---|---|---|
| `docs/METHOD.md` | 方法定稿 | 规格包 2026-10-15，原样 |
| `docs/CONTRACT_PROTOCOL.md` | 实施口径 v1.1 | 审阅交付包，原样 |
| `docs/roles.md`、`docs/status.md` | 角色清单；现状、已知问题、待决事项、版本沿革 | 本次整理 |
| `SOURCES.json` | 来源登记：每个来源的版本、URL、条款编号 | 审阅交付包 |
| `contracts/*.yaml` | 当前合同：四十四份已审阅，其余为草案 | 审阅交付包及后续开发 |
| `kernel_spec/`、`adapters/` | 唯一注册入口、共用数值表示、参照与合同适配器 | 当前开发 |
| `validated/` | 独立复核保留的合同、适配器、公开调用绑定计划与调用文件；当前文件不同时拒绝准入 | 当前开发 |
| `corpus/` | 真实工作负载候选映射、绑定检查与求值记录；各证据级别分开报告 | 当前开发 |
| `drafts/v2_5/` | 隔离的 v2.5 规格包，不参与运行时自动回退 | 历史草案 |
| `contracts/draft/` | 被取代的合同草案、适配器与草案检查器。检查器会给出假绿（审阅 F1–F5）；适配器是下一步实现的起点 | 规格包，原样（只改了一处相对路径） |
| `tools/`、`tests/` | 合同结构校验、例子对照参照、校验器回归 | 审阅交付包 |
| `specification_v2/` | 可执行参照 v2.4（`algorithm_specs/phase1–3`）、舍入 oracle、关系检查与随机验证、目录与计数、`SPECIFICATION_v2.md`、`THEORY.md` | 规格包，原样 |
| `evidence/` | ONNX Runtime 对照、v1 误读重建、THEORY 撤回部分的核对、Liger CE 对照 | 规格包，原样 |
| `reviews/2026-10-15/` | 审阅报告、实施任务书、审阅方的运行记录与复现脚本、被审规格包的顶层文件 | 审阅包，原样 |
| `run_checks.py` | 一条命令跑全部检查 | 本次整理 |

## 5 运行

```bash
export TMPDIR=/data1/tzh/kernel-analyzer/.cache/tmp XDG_CACHE_HOME=/data1/tzh/kernel-analyzer/.cache
/data1/tzh/envs/liger/bin/python 5_specification/run_checks.py
```

- **环境**：需要 Python ≥ 3.10 与 numpy、scipy、sympy、mpmath、pyyaml、python-flint。本机 `liger` 环境可运行；IR 集成检查另需工具环境的依赖。
- **用时**：随合同数量及机器负载变化；以每次输出的分项时间为准。
- **行为**：
  - 每项检查单独运行，任何一项失败即非零退出；
  - 会写文件的脚本在临时目录里重算，再与提交的记录比较，不写仓库。
- **2026-10-15 的结果**：
  - 16 项必需检查全部通过；
  - 参照测试共 222 项；
  - 关系检查、目录计数、随机验证、phase1 独立复算四份记录，重算后与提交的一致；
  - 例子对照的结果与审阅方的一致；
  - ONNX Runtime 对照未运行（没装 onnxruntime）。

## 6 阅读顺序

1. 本页；
2. [METHOD.md](docs/METHOD.md)，一页；
3. `contracts/CE.yaml`：一份完整的合同长什么样；
4. [roles.md](docs/roles.md)：31 个角色各自的来源与状态；
5. [status.md](docs/status.md)：已知问题与待决事项；
6. 需要细节时再看：[CONTRACT_PROTOCOL.md](docs/CONTRACT_PROTOCOL.md)、`reviews/2026-10-15/REVIEW.md`、`specification_v2/SPECIFICATION_v2.md`。
