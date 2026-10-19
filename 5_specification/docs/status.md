# 规格：现状、已知问题与待决事项（2026-10-19）

## 2026-10-19 三项条件与实际准入复核

Inductor 二维平均池化前向与完整 VJP 已接入正式合同模式。24次调用全部绑定通过；
前向96个、梯度384个坐标均有完整参照，梯度288个坐标的 Gπ−f 严格排除零。
公式差异与设备舍入分开报告，对齐规则确认梯度非零作用，固定方向均值未确认。
完整必需检查及71项集成检查通过。见[绑定与测量](../evidence/inductor_avg_pool2d_binding.md)。

平均池化现有明确任务选择下的完整前向和输入 VJP 数学参照，复用已有窗口计算并独立矩阵复算。
原20次调用的参照缺口已补齐；新8次参数验收中6次相容、2次单元素参数反向编译失败，全部保留。
完整必需检查通过（927项 unittest 及外部调用检查）。见[实现复核](../evidence/pool_implementation.md)
和[真实调用记录](../corpus/external_batch/README.md)。没有增加正式绑定或统计 bias 结论。

FlagGems 公开默认 RMSNorm 已增加一份完整前向与 VJP 绑定。24次调用覆盖全部 y/dx/dweight 的
K−f，y/dx 的 Gπ−f 可解释；dweight 最终经过原生 sum，程序分解仍不可用。前向参数错误导致的
向零缩小有独立公式界，仅适用于声明的输入范围。见[复核及测量](../evidence/flaggems_rms_norm_binding.md)。

最新实际绑定：FlagGems 5.0.2 原 MSE 前向对 `[7,41]`、三种 reduction 各完成24次正式测量，
完整调用级参照均可用，sum/mean 的内部零累加器有来源记录。mean 的 `Gπ−f=(45/2^32)*f`
已独立精确复算；sum 因统计分布条件不满足保留未知。这新增一份绑定，不扩展到其他调用或反向。
见[绑定核验](../evidence/flaggems_mse_binding.md)。

最新扩展：真实 MSE/L1 调用要求的 none/sum/mean 现均有完整前向与双侧 VJP 参照，原 mean
profile 的限制保留。24次 attorch 调用获得完整参照；另以新形状与种子完成12次 FlagGems MSE
前向验收，没有扩大为反向或统计 bias 结论。完整必需检查通过，含882项单元测试与78项相关
pytest 检查；原两个 mean 调用的工具绑定检查通过。详见下述外部批次及两份已有实现核验说明。

外部公开调用驱动的补充：HUBER 已按 PyTorch 2.10 官方来源完成 SmoothL1/Huber 的公式、参数、
三种 reduction 和完整双侧 VJP 核验；原8次开发调用已用新数学参照复核。完整必需检查通过，
另显式运行 pytest 的45项 softmax、损失和批次报告测试，避免 unittest 未执行函数式测试。
后续参数对应检查和真实调用复现 SmoothL1(beta=0) 的相等点梯度错误；不改规格在另一份
FlagGems 实现上完成48次前向验收，当前相关专项为51项通过。它不包含反向验收。
trident 四项验收仍因缺少 nvtx 和指定代理拒绝连接而未完成，不增加正式覆盖数。
参见[外部批次](../corpus/external_batch/README.md)和[HUBER 核验](../evidence/huber_implementation.md)。

真实公开 `Tensor.masked_fill(mask, Python scalar)` 调用暴露了 ARITH 只有前向值、缺少训练接口的问题。
新增 PyTorch 2.10 profile 复用原选择公式，并补全普通双向广播、广播输入的梯度求和、完整输入 VJP
和公开重载边界。20次 CPU 调用与独立坐标公式和标准 autograd 精确一致。FlagGems 5.0.2 开发10次
中4次完整通过，4次合法输入扩展／共同广播被拒绝，2次非连续输入值和梯度正确但输出布局不符合
函数式连续输出承诺；另一实现 TorchInductor 的冻结验收10/10精确通过并捕获前后向 Triton。
不增加正式调用绑定或统计 bias 结论。见[ARITH 核验](../evidence/arith_implementation.md)和
[真实调用记录](../corpus/external_batch/README.md)。完整必需检查通过，其中注册与单元测试946项。

真实公开 `torch.nn.functional.pad(..., mode='constant')` 随后暴露了 LAYOUT 只有前向逻辑值、不能裁决
训练调用的问题。新增 PyTorch 2.10 profile 固定完整公开参数、负 padding 先裁剪的验证顺序、前向及
输入 VJP；20组四 dtype 的独立坐标/autograd 核对和20次冻结 CPU 调用全部精确。FlagGems 5.0.2
开发10次的776个前向坐标均精确且捕获 Triton，但其公开 pad 替换没有可用 autograd 路径，10次都因
缺少 `dx` 保持为不完整任务；原始异常和补充的部分结果均保留。另一实现 TorchInductor 的陌生验收
10/10完成，1,444个前向和1,052个梯度坐标精确，每次均捕获前后向 Triton。不增加正式调用绑定或
统计 bias 结论。见[LAYOUT 核验](../evidence/layout_implementation.md)和
[真实调用记录](../corpus/external_batch/README.md)。该轮完整必需检查通过，其中注册与单元测试950项；
该轮结束时共62份合同、123个 profile，61个具备声明范围内的参照准入条件。

真实公开 Tensor/Tensor `torch.where` 接着暴露了 ARITH 只有选择前向、缺少两个数据 VJP 的问题。
新增 PyTorch 2.10 profile 直接复用已有前向定义，并按逐坐标转置分别在 condition 真／假位置路由
upstream，再把两个梯度各自归约回原数据形状。四种 dtype 的16组独立坐标/autograd 核对和20次
冻结 CPU 调用全部精确。FlagGems 5.0.2 开发10/10完整通过；另一实现 TorchInductor 的陌生验收
10/10完整通过，两组每次均捕获前后向 Triton，合计核对2,046个前向、1,262个 input 梯度和450个
other 梯度坐标。不增加正式调用绑定或统计 bias 结论。见[ARITH 核验](../evidence/arith_implementation.md)
和[真实调用记录](../corpus/external_batch/README.md)。完整必需检查通过，其中注册与单元测试954项；
该轮结束时共62份合同、124个 profile，62个具备声明范围内的参照准入条件。

真实公开 Tensor-list `torch.cat` 又表明 LAYOUT 的逻辑前向不足以裁决完整训练任务。新增 PyTorch 2.10
profile 复用已有前向 copy，并按坐标转置返回每个输入的有序 VJP；官方定义和保留的
`cat_tensors_backward` 选择共同核对公开边界。四种 dtype 的24组独立坐标/autograd 对照和20次冻结
CPU 调用全部精确。FlagGems 5.0.2 开发10/10完整通过，覆盖五输入跨两个 Triton launch、`(0,)` 空输入
和非连续输入；TorchInductor 陌生验收10/10完整通过，覆盖未见 rank、维度、六输入、零宽轴和不同
空输入位置。两组共核对1,492个前向及1,492个输入梯度坐标；每次捕获前向 Triton，VJP 是 upstream
切片且无需额外算术 kernel。不增加正式调用绑定或统计 bias 结论。见[LAYOUT 核验](../evidence/layout_implementation.md)
和[真实调用记录](../corpus/external_batch/README.md)。完整必需检查通过，其中注册与单元测试958项；
该轮结束时共62份合同、125个 profile，63个具备声明范围内的参照准入条件。

真实标量 bound `torch.clamp` 接着暴露了前向规格虽已有、版本化边界梯度仍缺失的问题。新增的
PyTorch 2.10 profile 根据保留的官方 autograd 来源明确在 min/max 相等坐标选择子梯度1，并完整返回
前向与输入 VJP；24组四 dtype 独立逐坐标/公开 autograd 对照、51个合同样例和24种错误公式全部通过。
FlagGems 5.0.2 开发10/10完整精确。另一实现 TorchInductor 的冻结陌生验收8/10完整精确；其余两次
在同一进程的 equal-bound 调用后把旧的0.25前向 bound 复用到 `[-0.75,1.25]`，但 VJP 仍使用本次
bound 并全部正确。无分析工具、无记录器的最小程序复现了这一顺序依赖；全新进程单独编译目标调用
则正确。原失败、独立对照和生成代码全部保留，规格没有按候选结果修改，也未建立统计 bias 或正式
kernel 绑定结论。见[ARITH 核验](../evidence/arith_implementation.md)和
[真实调用记录](../corpus/external_batch/README.md)。完整必需检查通过，其中注册与单元测试962项；
该轮结束时共62份合同、126个 profile，64个具备声明范围内的参照准入条件。

真实公开 Tensor/Tensor `torch.maximum/minimum` 随后表明 ARITH 仍只有前向，不能裁决相等坐标的梯度
选择和两个 operand 各自的广播归约。新增 PyTorch 2.10 profile 直接复用已有前向，并按保留的官方
autograd 来源在严格胜出坐标路由完整 upstream、在相等坐标各分一半，再分别归约两个 VJP。四种
dtype、两种操作和三组 shape 的24个独立对照全部精确；FlagGems 5.0.2 开发10/10与 TorchInductor
陌生验收10/10也逐坐标精确，验收包含整张量 equality 强见证。开发批只把前向归于 FlagGems Triton，
公共 autograd VJP 作为调用上下文；验收每次均捕获前后向 Triton。不增加正式绑定或统计 bias 结论。
完整必需检查通过，其中注册与单元测试966项；当前共62份合同、127个 profile，65个具备声明范围内的
参照准入条件。见[ARITH 核验](../evidence/arith_implementation.md)和
[真实调用记录](../corpus/external_batch/README.md)。

完整回归：本轮以当前规格环境执行 `run_checks.py`，全部必需检查通过；规格单元测试882项通过，此前工具侧完整回归1597项通过、160项按环境跳过。OPT4BIT 的10个合同例子和6个错误变体通过，6项独立参照测试通过；本轮末尾 storage/binding 专项为78项通过、1项按环境跳过，加入全部 `reference_eval_*` 测试后的扩展专项为235项通过、89项按环境跳过。未运行可选 ORT 对照。此前 `.cache/spec_full_checks_after_call_arguments.log` 保留的是早期运行记录，本轮没有改写。本轮新增 OPT4BIT 的物理状态参照、官方实现对照和真实三启动绑定，并新增 PyTorch Inductor LayerNorm、RMSNorm、GroupNorm、exact-form GELU、saved-output softmax backward、saved-output log-softmax backward、logsumexp forward+VJP、gather、cumsum forward+VJP 与 corrected variance forward+VJP、corrected standard deviation forward+VJP、ordered var_mean forward+combined VJP、public vector 2-norm forward+VJP 绑定；十四项都使用新的冻结记录，没有覆盖已有结果。

来源版本复核：VLLM_UA 与 VLLM_HELPERS 的保留字节均与原本地缓存相同，但本轮没有可用的对应checkout或上游响应记录来独立确认51eeb0c；指定27891代理在沙箱内外均拒绝连接。SOURCES保留原引用，并明确标记claimed version unverified，不声称版本已被证伪。来源归档入口现默认27891，不生成新哈希；已有快照不可覆盖，替换、重新抓取覆盖及原ID改标版本均拒绝，新版本须新来源ID。元数据说明更新不能改来源身份；检查器分别显示文件保留与已知版本未核实状态。这些修复不改变合同审核或运行时准入。 SAMPLING 的活动 profile 不再依赖该开发提交：它使用本机已安装 vLLM 0.16.0 发行包中的官方 sampler 文件，并以新来源 ID 单独保留；随机抽样和替代 backend 不在本次结论内。LORA 同样使用该安装发行包中的 shrink、metadata、kernel utility 和 PyTorch reference 文件，以独立的新来源 ID 保留。

当前直接读取62份合同得到127个 profile；44份合同标为 reviewed_for_implementation，18份为 drafted；44份标为 implementation=validated，18份为 pending；62份合同级 runtime_binding_status 均为 pending。ACT、ADAFACTOR、ATTN、BN、BSR、CE、ADAM、ADAMW、ARITH、CAST、CLIP、CONV、DIVERG、EMB、F2_MX、OPT4BIT、HUBER、INDEX、INDEXREDUCE、INTERP、L1、LAYOUT、LOGIC、LORA、PACK、PAGEDATTN、MATHFN、MATMUL、MSE、MOE、POOL、RANDOM、RADAM、RMSPROP、ROPE、RPROP、SAMPLING、SSU、REDUCE、RANKMEAN、SGD、SOFTMAX、SORT 与 NORM 的当前合同和适配器经过独立复核后已保留，共六十五个 profile 可在满足任务选择条件时作为规格参照；其余62个 profile 因实现尚待核验而默认拒绝。F2_MX 只覆盖 torchao 0.16.0 的有限 f32/bf16、末维32项分块、E4M3/E5M2、FLOOR 和普通 scale 布局，完整返回 scale code 与元素；其他 scale mode、FP4/FP6、任意轴、尾块、swizzle、特殊值和其他版本不准入。OPT4BIT 只准入 torchao 0.16.0 的物理 OptimState4bit float32 state-copy，完整核对 packed codes、block scales 和固定 qmap 前态；完整 AdamW4bit 步骤、特殊值、子正规数和替换 qmap 不准入。INDEXREDUCE 只覆盖物理有限同 dtype 浮点张量、int64 index 和可表示 alpha 的函数式 index_add 前向值；原地更新、index_reduce、scatter_reduce 与反向不准入。RANDOM 只覆盖无需复现随机流的 dropout 前向边界值，以及显式 Boolean mask 和 scale 的 native dropout backward；随机训练前向不准入。CLIP 只覆盖一个物理有限梯度缓冲区的 clip_grad_value_ 完整写回；范数裁剪、多缓冲区未完整对应、非正 clipping bound 和特殊值不准入。SGD 只覆盖一个稠密实数参数的一步公开状态转移，并完整返回参数与 momentum buffer；内部 fused、AMP overflow、可微和多参数任务不准入。ADAM 只覆盖一个稠密实数参数的一步公开 coupled Adam 状态转移，并完整返回参数、两个矩、可选 AMSGrad 最大矩和 step；AdamW、内部 fused/AMP、capturable、可微和多参数任务不准入。RMSPROP 只覆盖一个稠密实数参数的一步公开状态转移，并完整返回参数、square average、可选 centered average、可选 momentum buffer 和 step；capturable、可微和多参数任务不准入。RADAM 只覆盖 PyTorch 2.10 的单参数稠密实数一步公开状态转移，显式区分耦合/解耦 decay 和 rho>5 分支，完整返回参数、两个矩和 step；capturable、可微、稀疏、复数和多参数任务不准入。ADAFACTOR 分别覆盖 PyTorch 2.10 的单参数向量与末两维分解状态转移，保留 row/col 单位维、延迟状态和 dtype 决定的 eps1，完整返回参数及相应方差状态和 step；其他 Adafactor 实现、稀疏、复数和多参数任务不准入。RPROP 只覆盖 PyTorch 2.10 的单参数稠密实数一步公开状态转移，完整返回参数、previous gradient、step size 和 step；负数或次序颠倒的 step-size bounds、capturable、可微、稀疏、复数和多参数任务不准入。ATTN 覆盖 PyTorch 2.10 的有限稠密 rank-3/rank-4、dropout=0 且每行至少有一个有限有效 score 的公开 SDPA 前向，以及同一任务的完整 out、dq、dk、dv VJP；FlashAttention bottom-right causal、window/softcap、FlexAttention、内部 LSE、全 mask 空行和其他后端接口不准入。BN 只覆盖带显式 running statistics、正 eps 和可选仿射参数的物理有限 BatchNorm 推理前向值；训练、状态更新、native 多输出和反向不准入。INDEX 只覆盖物理有限浮点数据与 int64 索引上的默认只读 gather、index_select 和 embedding 前向；scatter、更新、反向累加、稀疏与非默认 embedding 选项不准入。EMB 只准入 PyTorch 2.10 的有限稠密 functional embedding 前向和完整 dense weight VJP，支持合法 padding_idx 与按完整索引计数的 scale_grad_by_freq；embedding_bag、max_norm、稀疏梯度、模块初始化和布局效果继续拒绝。ROPE 只覆盖 Transformers 4.57.3 Llama 的 supplied-table rotate-half q/k 前向和完整 q/k VJP；表生成、position lookup、partial/interleaved 布局、其他模型族和 cos/sin 梯度不准入。MOE 只覆盖 Mixtral 的无并列 top-2 路由与已提供 expert outputs 的加权组合；capacity、丢弃 token、router noise、auxiliary loss、expert MLP 和分布式 dispatch 不准入。BSR 只覆盖 PyTorch 2.10 二维 canonical BSR 的 softmax value core；稀疏张量分配、batch/hybrid、矩阵乘、sampled_addmm 和梯度不准入。SAMPLING 只覆盖 vLLM 0.16.0 在有限无并列 float32 logits 上的确定性 top-k 后 top-p 完整 keep mask；并列、masked logits、概率、随机抽样和替代 backend 不准入。LORA 只覆盖 vLLM 0.16.0 的有限 f16/bf16 LoRA A 投影与完整 float32 输出状态；expand、完整 B*A 更新、fused-MoE LoRA、任意 metadata 和其他输出 dtype 不准入。POOL 覆盖物理有限浮点上的二维 max_pool2d 前向值、唯一最大值下显式选择的 PyTorch 2.10 values+int64 indices+完整输入 VJP，以及单独选择的二维 avg_pool2d 完整前向和输入 VJP；并列最大值、自适应池化及未声明参数组合不准入。INTERP 只覆盖 PyTorch 2.10 rank-4 显式 size 的 legacy nearest 前向和完整输入 VJP，坐标约定要求显式任务选择；nearest-exact、scale-factor 调用、其他秩和插值模式不准入。LOGIC 只覆盖物理有限的同 dtype 浮点或布尔常用比较、逻辑、布尔位运算和 truth reduction，以及单独的浮点 `isnan` profile；整数位运算、混合 dtype、NaN 比较和未选重载不准入。ARITH 覆盖有限、同 dtype、值能由声明 dtype 精确表示的 f16/bf16/f32/f64 常见逐元素操作，并单独覆盖 PyTorch 2.10 标量 `masked_fill` 的完整公开前向和输入 VJP、Tensor/Tensor `where` 的完整前向和两个数据 VJP，标量 bound `clamp` 的完整前向和输入 VJP，以及 Tensor/Tensor `maximum/minimum` 的完整前向和两个数据 VJP；函数式状态由真实调用检查。整数、混合 dtype、特殊值、`pow`、其他重载和未声明梯度不准入。CAST 只覆盖有限 f16/bf16/f32/f64 之间明确选择 roundTiesToEven 的转换，输入必须能由声明的源 dtype 精确表示且结果有限；整数、布尔、float8、非有限值、目标溢出和设备或内存选项不准入。LAYOUT 覆盖有限浮点张量的常见函数式逻辑值和形状变换，并单独覆盖 PyTorch 2.10 constant pad 的完整公开前向和输入 VJP，以及 Tensor-list cat 的完整前向和全部有序输入 VJP；非 constant pad 模式、未声明梯度、存储、stride、别名、创建、dtype 转换和内存效果不准入。MATHFN 只覆盖绝对值不超过100、满足相应实数域的有限 f16/bf16/f32/f64 公共逐元素函数；整数、无穷和奇点边界不准入。REDUCE 只覆盖有限浮点的公共 sum、mean、prod、extrema、cumsum、logsumexp，以及 correction 为0或1且自由度为正的 `torch.var`、`torch.std`、有序 `torch.var_mean` 与已声明 VJP；多元素零方差切片的 std VJP、其他 variance/std/var_mean correction 与重载、已弃用 `torch.norm`、其他 vector-norm order/dtype 和零范数 VJP 仍不准入。CONV 只覆盖批量非转置 1-D/2-D 有限浮点默认操作；MATMUL 只覆盖有限 f16/bf16/f32/f64 的默认函数重载；RANKMEAN 另要求每次查询明确选择 token 等权任务。Inductor 末维 softmax、Inductor saved-output softmax backward、Inductor saved-output log-softmax backward、Inductor logsumexp forward+VJP、Inductor gather、Inductor cumsum forward+VJP、Inductor corrected variance forward+VJP、corrected standard deviation forward+VJP、ordered var_mean forward+combined VJP、public vector 2-norm forward+VJP、融合 mean CrossEntropy loss+VJP、Inductor mean MSE loss+完整 input/target VJP、Inductor mean L1 loss+完整 input/target VJP、Inductor LayerNorm forward+VJP、Inductor RMSNorm forward+VJP、Inductor GroupNorm forward+VJP、Inductor exact-form GELU forward+VJP、Inductor SiLU、Inductor ReLU/tanh/sigmoid forward+VJP、直接 Triton float32 matmul、causal GQA SDPA、embedding forward+dense VJP、supplied-table Llama RoPE、Mixtral top-2 routing+combination、BSR softmax values、deterministic top-k/top-p keep mask、Mamba selective state update、LoRA shrink、causal paged attention、packed varlen attention、PyTorch KL batchmean loss+input VJP、MXFP8 E5M2 FLOOR、grouped Conv2d、行求和、逐元素 exp、显式 bf16 转换、逐元素乘加、矩阵转置、大于零比较、行 index-select、无并列值行排序、max_pool2d、BatchNorm 推理、dropout backward、index_add、梯度值裁剪、warm-buffer SGD、warm-state Adam、warm-state RMSprop、warm-state rectified RAdam、warm-state factored Adafactor、warm-state Rprop、torchao OptimState4bit state-copy、FlagGems MSE 前向、FlagGems 公开默认 RMSNorm 完整 VJP、Inductor 二维平均池化完整 VJP 共五十八个具体调用已完成独立绑定审核；合同级状态仍为 pending，因为这不认证相应角色的其他操作或调用。此前 `.cache/spec_admission_after_profile_outputs.json` 是加入六个 collective profile 前的123项检查记录，没有据此改写或开启准入。

DIVERG 已从混合 PyTorch/Liger KL、JSD 和 TVD 的草案收窄为一个可核查任务：PyTorch 2.10 二维物理 f32 probability target、batchmean、log_target=false 和完整 input VJP。保留的官方源码与安装文件一致；独立768位公式、精确梯度、公开 PyTorch/autograd、完整输出及六种错误公式均通过复核。直接 Triton 8×64 调用的24个 loss 和12,288个梯度坐标都建立完整参照；基线任务公式差为零或被严格区间包含，给 loss 与梯度公式加入0.25后全部坐标排除零。证据见 `evidence/diverg_kl_implementation.md` 和 `evidence/triton_kl_batchmean_binding.md`。其他 reduction、log target、target VJP、非正 target、其他秩和 dtype 仍拒绝。

MSE 新增一个保守的 PyTorch 2.10 公开任务：无权重、同形状、非空物理 f32 的 `reduction="mean"` loss，并完整返回 input 与 target 两个 VJP。安装发行包中的官方文档、functional wrapper 和 native schema 固定调用边界与公式；4组独立精确有理数复算、12组公开 PyTorch/autograd 对照、13个合同例子及6种错误公式均通过。Inductor 实际生成一个融合 reduction kernel；24个 loss 和 input/target VJP 各98,304个坐标的完整率与分辨率均为1，程序公式与任务相容。设备到程序的平均绝对差约为 loss `2.74e-7`、两个梯度 `1.88e-11`，当前规则未确认稳定方向。第一次绑定因额外 `weight=None` 输入未被规格消费而拒绝，修正为参数缺席后通过，旧失败报告保留。证据见 `evidence/mse_implementation.md` 和 `evidence/inductor_mse_loss_binding.md`。广播、加权、其他 reduction、dtype、特殊值与空张量仍拒绝。

L1 随后补上 PyTorch 2.10 无权重 mean absolute error 与完整 input/target VJP。官方 L1Loss 文档、public wrapper、refs 实现、autograd 不可导点规则和 sign 定义共同固定公式及相等坐标的零子梯度；4组独立精确复算、12组强制包含相等坐标的公开 autograd 对照、14个合同例子和6种错误公式均通过。一个融合 Triton reduction kernel 在24个输入上建立24个 loss 与两个梯度各98,304个完整坐标；程序公式与任务精确相容。loss 的设备到程序平均绝对差约 `7.86e-8`，当前规则未确认稳定方向；两个梯度三层比较逐点为零，统计层明确标为退化、信息不足。证据见 `evidence/l1_implementation.md` 和 `evidence/inductor_l1_loss_binding.md`。范围不扩展到广播、加权、其他 reduction、复数、特殊值和空张量。

torchao 0.16.0 OptimState4bit 的旧草案只观察反量化动量，无法发现 packed codes、scale 或 nibble 顺序写错，已归档。活动合同改为版本固定的物理 state-copy，独立参照按官方源码的 float32 scale、canonical qmap 近邻选码和高半字节优先打包逐点计算。CPU/GPU 官方实现对照14次通过；真实 `torch.compile` 三次 Triton 启动的24个正式输入中，49,152个 packed-code 参照坐标和768个 scale 参照坐标完整且残差为零。由于残差全零，统计规则保留“无法判断总体无偏”，没有把样本相等升级为普遍结论。范围只是 signed float32[4096]、block size 128 的 state-copy，完整 AdamW4bit 仍未绑定。证据见 `evidence/opt4bit_state_copy_implementation.md`。

PyTorch 2.10 Inductor LayerNorm 随后完成公开 float32 `[16,128]` affine 前向和完整 x/weight/bias VJP 绑定。两次 Triton 启动的24个输入中，`y`、`dx` 各49,152个严格参照坐标，`dweight`、`dbias` 各3,072个，完整率和分辨率均为1。`y`、`dx`、`dweight` 的 K−Gπ 对齐规则均检出正向平均作用，Holm 校正后 p 值不大于 `7.24e-3`；`dbias` 未检出。程序还将 public double eps 改为 float32 常数并调用 `__nv_rsqrtf`，Gπ−f 在前三项上检出约 `1e-13`至 `4e-13` 的更小对齐差异。两类差异分开报告，范围不扩展到其他 shape、dtype、RMSNorm 或 GroupNorm。证据见 `evidence/inductor_layer_norm_binding.md`。

PyTorch 2.10 Inductor RMSNorm 随后完成同形 float32 affine 前向和完整 x/weight VJP 绑定。两次 Triton 启动的24个输入中，`y`、`dx` 各49,152个严格参照坐标，`dweight` 有3,072个，完整率和分辨率均为1。K−Gπ 的固定与对齐规则均未检出稳定平均作用，最接近阈值的 Holm 校正 p 值约为 `0.0748`；这不是无 bias 结论。生成程序在 float32 张量上使用 eps 并调用 `libdevice.rsqrt`，Gπ−f 在三项输出上检出约 `1e-13` 至 `4e-13` 的更小负向对齐差异。结论只属于该调用、shape、dtype 和输入分布。证据见 `evidence/inductor_rms_norm_binding.md`。

PyTorch 2.10 Inductor GroupNorm 随后完成 float32 `[4,16,8,8]`、4组、带 affine 参数的公开前向和完整 x/weight/bias VJP 绑定。四次 Triton 启动的24个输入中，`y`、`dx` 各98,304个严格参照坐标，`dweight`、`dbias` 各384个，完整率和分辨率均为1。K−Gπ 的对齐规则在 `y` 和 `dx` 上检出正向平均作用，Holm 校正后 p 值不大于 `5.15e-4`；两项参数梯度未检出。Gπ−f 在前三项输出上检出更小的负向差异，`dbias` 精确相容。设备差异与程序精度选择分开报告，范围不扩展到其他 shape、dtype、组数或布局。证据见 `evidence/inductor_group_norm_binding.md`。

PyTorch 2.10 Inductor exact-form GELU 随后完成 float32 `[32,128]` 公开前向和完整输入 VJP 绑定。第一次正式报告虽有完整任务参照，但程序参照在负尾部直接求 `1+erf(x)` 时发生区间抵消，205个前向和75个梯度坐标未达到既定分辨率，因此没有准入。程序参照加入严格恒等式 `1+erf(x)=erfc(-x)` 后，以相同24个输入重新测量，`y`、`dx` 各98,304个坐标的完整率和分辨率均为1，全部失败分类归零。K−Gπ 在前向固定方向检出负向平均作用，在梯度对齐方向检出负向平均作用；Gπ−f 另保留较小的程序公式差。旧失败报告与新报告同时保留。证据见 `evidence/inductor_gelu_erf_binding.md`。

PyTorch 2.10 Inductor saved-output softmax backward 随后完成 float32 `[16,128]`、末维归约的完整梯度绑定。输入是显式 upstream 和保存的 softmax 输出，任务公式直接来自 `_softmax_backward_data` 分解；构造保存输出的函数不参与 ground truth。一次 Triton reduction 启动的24个输入建立49,152个严格参照坐标，完整率和分辨率均为1。K−Gπ 的平均绝对差约 `2.01e-10`，固定与对齐规则均未检出稳定作用，最接近阈值的 Holm 校正 p 值约 `0.0529`；Gπ−f 与合同相容。未检出不表示无 bias，范围也不扩展到前向与反向的端到端组合。证据见 `evidence/inductor_softmax_backward_binding.md`。

PyTorch 2.10 Inductor saved-output log-softmax backward 也完成 float32 `[16,128]`、末维归约的完整梯度绑定。输入是显式 upstream 和保存的 log-softmax 输出，任务公式直接来自 `_log_softmax_backward_data` 分解；构造保存输出的函数不参与 ground truth。一次 Triton reduction 启动的24个输入建立49,152个严格参照坐标，完整率和分辨率均为1。K−Gπ 的平均绝对差约 `1.91e-8`、最大约 `4.87e-7`，现有固定与对齐规则均未检出稳定作用；Gπ−f 没有坐标严格排除零。范围不扩展到前向与反向的端到端组合。证据见 `evidence/inductor_log_softmax_backward_binding.md`。

PyTorch 2.10 Inductor logsumexp 随后完成 float32 `[16,128]` 末维公开前向与完整输入 VJP 绑定。一个融合 Triton kernel 同时写出16个前向值和2,048个梯度；24个输入共建立384个前向和49,152个 VJP 严格参照坐标，完整率和分辨率均为1。Gπ−f 两项均无坐标严格排除零；K−Gπ 的平均绝对差分别约为 `1.34e-7` 和 `9.42e-10`，现有规则均未检出稳定作用。证据见 `evidence/inductor_logsumexp_binding.md`。

PyTorch 2.10 Inductor gather 也完成 float32 `[32,64]`、int64 `[32,24]`、`dim=1` 的公开前向绑定。一个 Triton kernel 在24个输入的18,432个完整坐标上得到 K−Gπ=0、Gπ−f=0 和 K−f=0，两个输入保持不变。因为每组残差和校准方向都为零，统计层明确返回 `statistics insufficient`，不把样本内精确一致升级为总体无 bias。证据见 `evidence/inductor_gather_binding.md`。

PyTorch 2.10 Inductor cumsum 随后完成 float32 `[16,128]` 公开前向与完整输入 VJP 绑定。实际生成3个 Triton kernel，包含前向 scan、反向 scan 及 flip 写回；24个输入为前向和 VJP 各建立49,152个严格程序参照坐标，完整率和分辨率均为1，Gπ−f 两项逐点相容。K−Gπ 的平均绝对差分别约为 `4.27e-7` 和 `4.08e-7`，当前规则均未检出稳定作用。证据见 `evidence/inductor_cumsum_binding.md`。

PyTorch 2.10 Inductor corrected variance 随后完成 float32 `[16,128]`、`dim=1`、`correction=1` 的公开前向与完整输入 VJP 绑定。一个融合 Triton kernel 在24个输入上建立384个前向和49,152个 VJP 程序参照坐标，完整率和分辨率均为1。前向 Gπ−f 没有坐标排除零；反向将精确 `2/127` 生成为 float32 系数，全部49,152个 Gπ−f 坐标排除零，平均绝对差约 `7.55e-11`、最大约 `1.03e-9`，对齐方向规则检出正向平均作用。K−Gπ 的平均绝对差分别约为 `1.58e-7` 和 `7.60e-10`，当前规则均未确认稳定平均作用。证据见 `evidence/inductor_var_binding.md`。

PyTorch 2.10 Inductor corrected standard deviation 随后完成同形、同轴和同 correction 的公开前向与完整输入 VJP 绑定。一个融合 Triton kernel 在24个输入上建立384个前向和49,152个 VJP 程序参照坐标，完整率和分辨率均为1。前向 Gπ−f 没有坐标排除零；K−Gπ 的平均绝对差约 `5.32e-8`、最大约 `2.49e-7`，R1、R2、R3 检出正向平均作用。反向的全部 Gπ−f 坐标排除零，平均绝对差约 `1.82e-11`、最大约 `2.44e-10`；R5 检出负向、R2/R3 检出正向平均作用。完整设备梯度的规则未确认这一更小作用。证据见 `evidence/inductor_std_binding.md`。

PyTorch 2.10 Inductor ordered var_mean 随后完成同形、同轴和同 correction 的完整有序前向对与组合输入 VJP 绑定。一个融合 Triton kernel 在24个输入上分别建立384个方差、384个均值和49,152个 VJP 程序参照坐标，完整率和分辨率均为1。均值 Gπ−f 逐点为零；方差没有坐标严格排除零。组合 VJP 的全部 Gπ−f 坐标排除零，平均绝对差约 `6.99e-11`、最大约 `1.04e-9`，R2/R3 检出正向平均作用。这与生成程序中方差分支采用 float32 `2/127`、均值分支采用可精确表示的 `1/128` 相符。K−Gπ 单独未确认稳定作用；完整 K−f 仅 R2 检出正向平均作用，Holm 校正后 `p≈0.0447`。证据见 `evidence/inductor_var_mean_binding.md`。

PyTorch 2.10 Inductor public vector 2-norm 随后完成 float32 `[16,128]`、`ord=2`、`dim=1` 的公开前向和完整输入 VJP 绑定。一个融合 Triton kernel 在24个输入上建立384个前向和49,152个 VJP 程序参照坐标，完整率和分辨率均为1。前向使用 `tl.sqrt_rn`，反向除以范数；零范数分支存在，但当前规格只在正范数处裁决 VJP。前向与 VJP 的 Gπ−f 均无坐标排除零。K−Gπ 的平均绝对差分别约 `5.98e-7` 和 `2.55e-9`，最大约 `2.03e-6` 和 `9.06e-8`，现有规则均未确认稳定平均作用。证据见 `evidence/inductor_vector_norm_binding.md`。

SAMPLING 随后从同时声称 deterministic filter、masked logits、概率和随机抽样的宽泛草案，收窄为安装的 vLLM 0.16.0 `apply_top_k_top_p` 确定性 keep mask。保留的 nucleus-sampling 论文和安装源码共同固定 top-k、top-p 及其先后次序；无并列条件避免替实现选择未规定的 tie order。20个合同样例与6个错误读法通过，20组768位独立公式、公共 PyTorch 公式和安装的 vLLM 函数均一致。真实 Triton 调用的98,304个布尔坐标全部匹配；每行翻转 token 0 后恰有1,536个错位，并由语义与完整输出比较检出。证据见 `evidence/sampling_filter_implementation.md` 和 `evidence/triton_topk_topp_keep_binding.md`。

SSU 随后从混合 mamba 实现公式和论文 ZOH 的草案收窄为安装的 mamba-ssm 2.3.2.post1 单步任务。保留的安装源码固定 dA=exp(dt*A)、dB=dt*B、连续 head 分组、完整 state 写回和 threshold-20 softplus；审核还复现了该版本在 D 或 dt_bias 为空时 wrapper 启动失败，因此活动范围要求两者存在。18个合同例子与11个错误公式通过，7组768位独立逐元素公式、公开 PyTorch 公式和6组官方 Triton/参考调用一致。正式 Triton 基线建立24,576个 y 与196,608个 state 坐标；程序公式与任务相容，state 的 float32 执行检出约1.01e-8平均绝对残差和小的正向平均作用。只给最终 state 写回加0.25后，全部196,608个 state 坐标排除正确参考，y 不误报。证据见 `evidence/ssu_implementation.md` 和 `evidence/triton_mamba_ssu_binding.md`。

LORA 随后从未固定发行版、物理 dtype 和完整输出前态的草案，收窄为安装的 vLLM 0.16.0 LoRA A 投影。保留的论文与安装源码共同固定低秩任务、映射、scaling、切片和状态规则；尤其区分活动调用清零完整输出与全 `-1` 调用提前返回并保留旧输出。16个合同例子与5个错误实现通过，12组独立精确循环和 PyTorch 公式以及7组官方 Triton 调用一致。正式 Triton 基线建立24,576个完整输出坐标，程序公式与任务相容，设备执行平均绝对差约2.31e-8；全部输出加0.25后，每个坐标都排除正确参考且检出非零作用。证据见 `evidence/lora_shrink_implementation.md` 和 `evidence/triton_lora_shrink_binding.md`。

SORT 随后从允许并列索引集合的宽泛草案收窄为唯一答案范围：只准入物理有限 f16/bf16/f32/f64、选定维度每行无重复值的 sort、argsort 和 `sorted=true` top-k。PyTorch 2.10 文档明确说明不稳定排序和 top-k 的并列索引没有稳定保证，因此并列、NaN、无穷值、无序 top-k 和其他重载均继续拒绝。17个合同样例与6个错误读法通过，240组 NumPy 坐标构造和 PyTorch 2.10 四种 dtype 对照一致。真实 Triton 行排序在24个输入上对全部49,152个公开输出坐标得到精确 K−f=0；临时 `+0.25` 值错误被完整任务比较和固定均值规则检出。Triton bitonic sort 的归约组合器尚未被程序参照解释，因此本例没有 K−Gπ，报告保留未建立而没有用 K−f 冒充。证据见 `evidence/sort_implementation.md` 和 `evidence/triton_sort_unique_binding.md`。

POOL 随后从混合 max/average/adaptive、索引和反向的宽泛草案收窄为普通二维 max_pool2d 前向值。保留的 PyTorch 2.10 文档与 native schema 支持窗口、输出形状、参数默认值和 values-only 调用边界；平均池化尾部除数、索引 tie 选择、反向路由、ceil mode 与特殊值没有被自行补成答案。18个合同样例与6个错误读法通过，24组独立坐标循环和 PyTorch 对照覆盖四种浮点 dtype、rank 3/4、矩形窗口、stride 与 dilation。真实 Triton 调用在24个输入的9,216个坐标上得到精确 K−Gπ=0、Gπ−f=0、K−f=0；临时 `+0.25` 错误使全部 Gπ−f 坐标排除零并被完整输出规则检出。证据见 `evidence/pool_implementation.md` 和 `evidence/triton_max_pool2d_binding.md`。

BN 随后从混合训练、running-state 更新、native saved 输出和反向的宽泛草案收窄为普通 BatchNorm 推理前向值。保留的 PyTorch 2.10 文档、decomposition 与 native schema 支持显式 running statistics、逐通道仿射公式和单输出调用边界。18个合同样例与6个错误读法通过，12组独立坐标循环和 PyTorch 对照覆盖四种浮点 dtype 与 rank 2/3/4；非平方根另由100位独立求值核对严格包围。真实 Triton 调用在24个输入的24,576个任务坐标上没有 Gπ−f 坐标排除零，设备到程序平均绝对差约3.46e-8；临时 `+0.25` 错误使全部 Gπ−f 坐标排除零并被完整输出规则检出，同时 K−Gπ 保持普通 float32 量级。证据见 `evidence/bn_implementation.md` 和 `evidence/triton_batch_norm_eval_binding.md`。

RANDOM 随后撤下了依赖合成 `keep` 参数的 conditioned 草案。PyTorch 2.10 的函数 schema 没有这个参数，而当前绑定也不能用返回 mask 反向构造任务输入；这样做还会失去对 mask 本身的独立判断。新合同只保留结果不依赖随机流的前向边界，以及显式 mask 的 native dropout backward。17个合同样例与5个错误读法通过；28组独立坐标公式和 PyTorch 对照覆盖四种浮点 dtype 与 rank 1–4。真实 Triton backward 在24个输入的24,576个程序语义坐标上与任务公式精确一致，设备到程序平均绝对差约1.61e-8；临时 `+0.25` 错误使全部 Gπ−f 坐标排除零，并由语义与完整输出规则检出。随机训练前向仍明确拒绝，等待独立的随机分布检验。证据见 `evidence/random_deterministic_implementation.md` 和 `evidence/triton_dropout_backward_binding.md`。

INDEXREDUCE 随后从混合 index_add、index_reduce、scatter_reduce、原地状态、VJP 和并列极值集合的宽泛草案收窄为函数式 index_add 前向值。保留的 PyTorch 2.10 文档和 native schema 直接支持目标切片、alpha、source 形状和单输出调用；其他 reduction 的 identity、include_self 和并列梯度没有被带入该 profile。20个合同样例与6个错误读法通过，16组独立坐标循环和 PyTorch 对照覆盖四种浮点 dtype、rank 1–4、不同维度、重复 index 与非单位 alpha。固定偶数行 index 的真实 Triton 调用在24个输入的98,304个程序语义坐标上与任务公式精确一致，设备到程序平均绝对差约9.04e-9；临时 `+0.25` 错误使全部 Gπ−f 坐标排除零，并由语义与完整输出规则检出。证据见 `evidence/index_add_implementation.md` 和 `evidence/triton_index_add_binding.md`。

CLIP 随后从混合总范数、范数裁剪、非有限策略和多个公开调用的宽泛草案收窄为一个梯度缓冲区的 `clip_grad_value_` 完整写回。PyTorch 2.10 文档直接规定区间 `[-clip_value, clip_value]`，保留源码明确按下界和上界原地裁剪；范数归约、epsilon、多个未完整对应的梯度和特殊值没有被带入该 profile。19个合同样例与6个错误读法通过，16组独立坐标和 PyTorch 对照覆盖四种浮点 dtype、rank 1–4 与三种 foreach 选择。真实原地 Triton 调用在24个输入的98,304个坐标上得到精确 K−Gπ=0、Gπ−f=0 和 K−f=0；临时 `+0.25` 错误使全部 Gπ−f 坐标排除零并被方向规则与完整输出规则检出。证据见 `evidence/clip_value_implementation.md` 和 `evidence/triton_clip_grad_value_binding.md`。

SGD 随后从同时声称内部 `_fused_sgd_`、忽略额外字段且没有完整输出比较的草案，收窄为一个稠密实数参数的一步公开 `torch.optim.SGD` 状态转移。PyTorch 2.10 文档和源码共同固定 maximize、coupled weight decay、首个与已有 momentum buffer、dampening、Nesterov、空梯度及构造器约束。21个合同样例与11个错误读法通过；80组独立公式和 PyTorch 对照覆盖四种浮点 dtype、五类状态及 loop/foreach/default/fused 路径。真实 Triton 调用中，参数与 buffer 各98,304个 Gπ−f 坐标都精确为零；设备执行的平均绝对 K−f 分别约1.91e-8和2.92e-8。临时版本只给 buffer 写回增加0.25，全部 buffer Gπ−f 坐标精确偏移0.25，而参数仍为零。证据见 `evidence/sgd_implementation.md` 和 `evidence/triton_sgd_binding.md`。

ADAM 随后从混合公开 Adam、AdamW 和内部 fused 调用且总是制造 `vmax` 的草案，收窄为一个稠密实数参数的一步公开 coupled `torch.optim.Adam` 状态转移。PyTorch 2.10 文档与源码共同固定 lazy state、真实 state keys、maximize/decay 次序、bias correction、AMSGrad 与 eps 位置。22个合同样例与12个错误读法通过；80组独立公式和 PyTorch 对照覆盖四种浮点 dtype、五类状态及 loop/foreach/default/fused 路径。真实 Triton 调用中，theta、m、v、vmax 各98,304个 Gπ−f 坐标和24个 step 坐标均相容；只给 vmax 写回增加0.25后，全部 vmax Gπ−f 坐标排除零，其他四项仍为零。证据见 `evidence/adam_implementation.md` 和 `evidence/triton_adam_binding.md`。

RMSPROP 随后从始终制造 centered average 和 momentum buffer 的草案，收窄为一个稠密实数参数的一步公开 `torch.optim.RMSprop` 状态转移。PyTorch 2.10 文档与源码共同固定 lazy state、真实 state keys、maximize/decay 次序、centered 估计、eps 位置与 momentum 更新。22个合同样例与12个错误读法通过；60组独立公式和 PyTorch 对照覆盖四种浮点 dtype、五类状态及 loop/foreach/default 路径。真实 Triton 调用中，theta、v、gave、b 各98,304个 Gπ−f 坐标和24个 step 坐标均相容；只给 b 写回增加0.25后，全部 b Gπ−f 坐标排除零，其他四项仍为零。证据见 `evidence/rmsprop_implementation.md` 和 `evidence/triton_rmsprop_binding.md`。

RADAM 随后从要求预初始化状态、混用广义 RAdam 读法的草案，收窄为 PyTorch 2.10 的一个稠密实数参数一步公开状态转移。保留的官方文档和源码共同固定 lazy state、真实 state keys、maximize/decay 次序、rho>5 分支和 loop/foreach 路径。22个合同样例与11个错误读法通过；72组独立公式和 PyTorch 对照覆盖四种浮点 dtype、六类状态及 loop/foreach/default 路径。第一次 Triton 绑定把整流系数预先化成 host float，Gπ−f 因此检出 theta 的约3.10e-11公式差；改为从精确可表示的 beta2 在程序图中计算后，theta、m、v 各98,304个坐标和24个 step 坐标均相容。只给 m 写回增加0.25后，全部 m Gπ−f 坐标排除零，其他三项仍为零。证据见 `evidence/radam_implementation.md` 和 `evidence/triton_radam_binding.md`。

ADAFACTOR 随后从混合 PyTorch 与 Transformers 两个任务、要求预初始化状态并压掉 row/col 单位维的草案，收窄为 PyTorch 2.10 的单参数向量与末两维分解状态转移。保留的官方文档和源码共同固定 lazy state、真实 state keys、mean 归约、两处 eps1 clamp、decoupled decay 和 loop/foreach 路径。21个合同样例与12个错误读法通过；48组独立 NumPy 公式和 PyTorch 对照覆盖四种浮点 dtype、四类状态及 loop/foreach/default 路径。真实 32×32 Triton 调用中，theta 的24,576个、row/col 各768个程序语义坐标和24个 step 坐标均相容；只给 row 写回增加0.25后，全部768个 row 坐标排除零，theta、col 与 step 不误报。证据见 `evidence/adafactor_implementation.md` 和 `evidence/triton_adafactor_binding.md`。

RPROP 从同时混入版本化执行行为和一般算法描述的草案，收窄为 PyTorch 2.10 的单参数稠密实数一步状态转移。保留的官方文档和源码共同固定 lazy state、真实 state keys、sign 分支、maximize 以及 loop/foreach 路径；源码还表明零乘积分支仍会 clamp step size。负数或次序颠倒的 step-size bounds 虽会被公开构造器接受，但语义异常，当前 profile 明确拒绝而不自行解释。20个合同样例与10个错误读法通过；60组独立 NumPy 公式和 PyTorch 对照覆盖四种浮点 dtype、五类状态及 loop/foreach/default 路径。真实 length-4096 Triton 调用中，theta、prev、eta 各98,304个程序语义坐标和24个 step 坐标均相容；只给 eta 写回增加0.25后，全部98,304个 eta 坐标排除零，其他输出不误报。证据见 `evidence/rprop_implementation.md` 和 `evidence/triton_rprop_binding.md`。

ATTN 原草案把 PyTorch SDPA、FlashAttention、FlexAttention、Triton 教程 LSE 和完整反向放在同一合同。当前活动合同只保留 PyTorch 2.10 有限稠密 rank-3/rank-4、dropout=0、每行至少有一个有限有效 score 的公开前向输出；原扩展版保留在草案目录。18个合同样例与8个错误读法通过；5组768-bit Arb 独立公式和20组 PyTorch math backend 对照覆盖四种浮点 dtype、causal、Boolean/additive mask、GQA 与 head broadcasting。真实 causal GQA Triton 调用的49,152个程序语义坐标均与任务公式相容；设备到程序的对齐规则检出小的负向数值偏差，只给最终输出加0.25后全部49,152个 Gπ−f 坐标排除零。证据见 `evidence/attention_implementation.md` 和 `evidence/triton_sdpa_binding.md`。

EMB 原草案同时包含 embedding、embedding_bag、max_norm 原地重归一和两种冲突的 renorm profile。当前活动合同只保留 PyTorch 2.10 functional embedding 前向和完整 dense weight VJP；旧扩展版保留在草案目录。18个合同样例与7个错误读法通过；独立坐标公式和20组 PyTorch 对照覆盖四种浮点 dtype、重复及多维索引、正负 padding_idx 和 frequency scaling。真实 Triton 调用的 forward 49,152个、dweight 24,576个程序坐标均建立完整参照且与任务公式相容；只给非 padding 梯度行加0.25后，24,192个 Gπ−f 坐标排除零，两类语义规则均确认非零。证据见 `evidence/embedding_implementation.md` 和 `evidence/triton_embedding_binding.md`。

ROPE 原草案把角度生成、half/interleaved、partial rotary、GPT-OSS 表布局和 Liger 批量实现放在五个 profile 中，并把优化实现只读取左半表的条件误当成公开 Llama table-application 的条件。当前活动合同只保留 Transformers 4.57.3 Llama supplied-table rotate-half q/k 前向和完整 q/k VJP，且正确保留不相等的表两半；旧扩展版移入草案。16个合同样例与6个错误读法通过；独立坐标公式、20组 PyTorch/autograd 对照及实际 Transformers 4.57.6 的20组调用一致。直接 Triton 两次启动的基线中，q_out/dq 各49,152个、k_out/dk 各24,576个程序坐标全部建立且 Gπ−f 为零；只给两个梯度写回加0.25后，全部73,728个梯度坐标排除零，两个前向结果不误报。证据见 `evidence/rope_implementation.md` 和 `evidence/triton_llama_rope_binding.md`。

MOE 原草案混合 Switch capacity、Mixtral routing、raw gates 和 auxiliary loss。当前活动合同只保留 Mixtral 的无并列 top-2 路由与 supplied expert outputs 加权组合；保留论文和 Transformers 4.57.3 源码共同固定任务，安装的 Transformers 4.57.6 对应函数体一致。14个合同样例与6个错误读法通过；768-bit Arb 独立求值和20组公开 PyTorch 公式覆盖四种浮点 dtype。直接 Triton 基线中，24,576个 Boolean 选择位零误判，权重与组合结果的24,576和98,304个坐标均有完整严格参照。正常 float32 执行显示相对精确实数任务的微小数值效应；只给组合结果加0.25后，全部98,304个 Gπ−f 坐标排除零。Triton reduction 的具体 lowering 次序仍作为显式前提，因此整体程序语义状态不冒充已完全建立。证据见 `evidence/moe_implementation.md` 和 `evidence/triton_mixtral_top2_binding.md`。

BSR 原草案把 `bsr_dense_mm`、`bsr_softmax` 和 BSR `sampled_addmm` 放在一个 profile 中，其中 sampled-addmm 的 `spy(input)` 对显式零仍有文字歧义。当前活动合同只保留 PyTorch 2.10 二维 canonical BSR softmax value core。官方稀疏文档与保留 Triton 源码共同固定：每个 scalar row 跨该 block row 的全部 specified block 归一化，显式零参与、未指定位置不参与；安装源码的对应函数体与保留版本一致。16个合同样例与6个错误读法通过；768-bit Arb 独立求值和15组公开 PyTorch 公式覆盖三种 dtype。直接 Triton core 与官方 Triton 调用逐位一致。正式基线的98,304个坐标均有完整参照且程序公式与任务相容；只给写回加0.25后，全部 Gπ−f 坐标排除零。证据见 `evidence/bsr_softmax_implementation.md` 和 `evidence/triton_bsr_softmax_values_binding.md`。

运行时准入已经去掉内容哈希。开发检查现在只把日期、例子数、读法数、变异数和结果写入 `contracts/DEVELOPMENT_CHECKS.json`，注册入口不读取该文件。独立实现复核若要准入，需在 `validated/` 保留当时审阅的完整合同和适配器；运行时直接比较原文件，当前文件不同即拒绝，开发检查不会自动复制。这样“测试通过”和“当前实现经过独立核验”不会再由同一个记录混淆。

本轮还用四类真实 Triton 算子检查了合同边界。FlagGems sort 首次试跑发现比较器不接受合同声明的带 dtype/shape 张量记录；Liger RMSNorm 发现公开 autograd 接口并不返回内部 rstd；Liger RoPE 发现合法的 `[B,H,L,d]` 输入及批量 cos/sin 表未被 profile 表达；Liger CrossEntropy 发现前向会覆盖 logits、且 `reduction=none` 与标量归约的后态不同。修复后四项设备重跑均完成全部声明输出和后态的比较。这里记录的是固定输入上的 K 与 f 数值差异，不能单独推出分布意义上的 bias；合同仍按原审阅和绑定状态处理。FlagGems public sort 的后续正式入口因九次启动之间的内部 values、indices 和 exclusive-prefix 状态尚未获得完整 producer 证据而在统计前拒绝，继续保持草案；见 `evidence/flaggems_sort_binding_gap.md`。

| 原始要求 | 当前可核查证据 | 尚未闭合的部分 |
|---|---|---|
| 唯一入口 | registry 只读顶层合同；四十份已审阅合同以当前保留文件准入；A1 与 K14 等仍在草案，历史不回退 | 其余18份合同尚未完成同等复核 |
| 任务定义与合法 profile | A1 区分未归一化/归一化及各递推；F2 两个尺度选择分别定义 | A1/F2 来源尚未独立核实；不能称完整论文/标准算法已认证 |
| 完整输入、状态和输出 | A1 完整状态/VJP；MOE 矩形专家保留三个输出；共同严格结构检查 | 草案的大多数实现仍未完成独立审核；不能以测试数量抵消 |
| 严格数学保证 | MOE 等权1/3包围测试、区间后端回归；分片合并已接入 K14 草案 | 必须按具体合同审查所有允许集合和边界，不能由后端通过推导全库保证 |
| 检查器不假通过 | 错预期、损坏读法、缺输出、子进程失败等注入检查；全库58份相容 | 相容性检查不提供来源审阅或运行时准入 |
| kernel 对应 | 四十八个具体调用已完成正式绑定、完整参照和生产测量；另有候选、结构对应及设备核对记录 | 这四十八个调用不能代表同名 kernel、整个角色或真实工作负载覆盖率 |
| 独立可信性证据 | 手算、不同算法精确复算、边界与错误变体材料已存在 | 作者开发检查不是独立审阅；本地源码上游版本、发布证据与独立验收仍欠缺 |

三试点准入衔接前又发现并修复了依赖入口问题：v24.module 原先按通用模块名导入，可能读取其他搜索路径或缓存中的同名文件。现在只接受六个明确的仓库文件，并核对优化器/训练程序实际引用的本地数值后端；同名冲突拒绝，不覆盖已有模块，也不执行替代文件。五个专项反例包括搜索路径、缓存、后端冲突和运行时拒绝传播。这是来源路径对应，不是源码内容或数学语义认证；尚未恢复旧准入记录。

除本页明确记录的 ACT、ADAFACTOR、ATTN、BN、BSR、CE、ADAM、ADAMW、ARITH、CAST、CLIP、CONV、DIVERG、EMB、F2_MX、OPT4BIT、HUBER、INDEX、INDEXREDUCE、INTERP、L1、LAYOUT、LOGIC、LORA、PACK、PAGEDATTN、MATHFN、MATMUL、MSE、MOE、POOL、RANDOM、RADAM、RMSPROP、ROPE、RPROP、SAMPLING、SSU、REDUCE、RANKMEAN、SGD、SOFTMAX、SORT 与 NORM 规格参照准入外，没有将其他合同、待选任务语义或真实绑定自动放行，也没有重写冻结记录。下一优先项是其余高频通用任务的来源和实现审核，而非增加专用角色或通过计数。

工具接入复核又发现一条同源参照越界：`audit(spec=...)` 已经把 float64 回调降为 implementation reference，但 `check`/`measure` 及接入指南仍把任意 `spec()` 的 K_R 差异命名为 semantic，并默认写成 rigorous float64 enclosure。现在保留旧数值字段以兼容记录，但报告明确给出 `reference_kind=implementation_reference`、`semantic_verdict=not_established` 和 `verdict_scope=reference_difference; not a semantic verdict`；`f64_point_spec` 只称经验筛查带。CPU 黑盒回归和一条真实 Triton GPU mode-B 回归均通过。区域求值同样把捕获编译图标为 `captured_graph_complete`，不再称 K-versus-f。这些修正阻止任意回调被误当成 specification。

在此基础上，`measure` 已增加正式合同模式。它不接收任意参照回调，只接受状态为 reviewed 且独立保留的绑定计划、调用文件、合同和适配器，并核对绑定计划声明的已安装发行包版本；绑定计划固定合同输出与测量输出的对应。每个抽样调用重新核对输入前态、实际后态、全部 Triton 启动和严格参照。所有实际输入必须映射到合同，或由审阅计划逐项说明为何不用；容器中隐藏的张量不允许绕过前后态检查。此前原地调用后才执行 `spec(inputs)` 会让参照误读修改后的输入，现已在同一存储和别名关系下暂时恢复调用前字节，参照完成后恢复真实后态。统计现在分别给出 K−Gπ、条件成立时的 Gπ−f，以及完整公开输出上的 K−f；后者不再遗漏未被 Triton 解释器覆盖的坐标。精确布尔输出另报告完整误判、假阳性和假阴性计数，避免正负差抵消。真实 Triton CE 回归在24个独立输入上检出了人为加入的 `+0.25` loss 偏差，也检出了仅存在于未写输出坐标的梯度偏差。当前六十二份真实调用计划中，五十八份已完成绑定审核和正式测量，逐项清单以本文开头的当前状态为准；另外四份继续拒绝；第三关已有逐元素、转换、基础归约、逻辑与比较、只读索引、indexed add、逻辑布局、多启动前后向、矩阵乘、空间卷积、精确排序、窗口池化、推理归一化和显式 mask 反向的正式实例，不能据此推算普遍覆盖率。

随后新增并审核了真实 PyTorch 2.10 Inductor CE 调用。公开调用在一个 `torch.compile` 区域中返回二维 unsmoothed mean CE 的 loss 和完整 VJP；计划逐项对应七个输入、两个未修改输入状态和全部公开输出。GPU 开发检查捕获三个 Triton kernel。隔离副本以相同24个输入比较基线与编译区内 `+0.25` 注入：基线的 loss/gradient 均未检出稳定 bias，注入版只在 loss 上检出约 `-0.25000004`，gradient 统计保持相同。当前 PyTorch 2.10 的独立 ATen 图核对了 log-softmax、mean NLL、上游1及完整反向关系；保留当前计划和调用文件后，正式入口在24次调用中建立全部 loss 与4416个梯度参考坐标，三个启动完整覆盖且没有失败分类。正常实现的 Gπ−f 没有坐标排除零，K−Gπ 与完整 K−f 均未检出稳定均值。证据见 `evidence/inductor_cross_entropy_binding.md`。

SOFTMAX 随后完成当前文件的独立来源、公式、输入域和完整输出复核。保存的 PyTorch 2.10 文档及源码分别支持普通 softmax/log-softmax、safe-softmax、两个反向公式和在线准备的两个输出；标量、空维度、无穷、输入转换和多个轴均另行核对。其 `real` profile 现通过规格参照准入。Inductor 末维 softmax 先在隔离副本中用相同24个输入比较基线和编译区内 `+0.01` 注入：注入版的 K−Gπ 未检出，而 Gπ−f 与完整 K−f 均检出非零。随后按当前 PyTorch 2.10 的实际 ATen 图逐项审核公开调用、三个输入、完整输出、输入后态、隐藏来源及全部启动，并保留计划和调用文件。正式声明连续运行两次，24个绑定、14208个参考坐标、全部统计和输入来源逐项一致：Gπ−f 没有坐标排除零，K−Gπ 的 R1 均值约为 `-6.58e-9`、Holm 校正后 p 约为 `0.00545`。该数值结论只对应已声明输入分布；审核范围和证据见 `evidence/inductor_softmax_binding.md`。

NORM 的三个 profile 随后完成来源、公式、输入域和实现复核并通过规格参照准入。复核先发现两处会产生错误 ground truth 的边界：合同原先把负 eps 一律判为非法，但 PyTorch 公式和实现会在根号内结果仍为正时接受；公开 `group_norm` 还具有底层 native 运算没有的单值输入拒绝。现以 positive-radicand 定义实数域，并把 public/native 边界分到 `torch_autograd` 与 full/native `torch` profile。独立 Fraction dual-number 自动微分从前向式直接复算 LayerNorm、RMSNorm、offset RMSNorm 和 GroupNorm 的全部参数梯度；22个合同样例相容，14个错误公式被拒绝。真实 Liger RMSNorm 计划仍因实测0.6.2与保存0.7.0来源的对应尚未独立确认而保持草案。

REDUCE 原来的宽泛草案把普通求和、均值和极值与多种统计量和范数放在同一个准入单元中；标准差和范数在零点的导数选择会阻塞已经清楚的归约。当前运行时合同覆盖 PyTorch 2.10 明确定义的有限浮点 sum、mean、prod、amax、amin、argmax、argmin、max、min、cumsum、logsumexp，以及 correction 为0或1、自由度为正的 `torch.var`、`torch.std`、有序 `torch.var_mean` 与已声明 VJP；另准入公开 `torch.linalg.vector_norm` 的实数 f32/f64 二范数及正范数处 VJP。std VJP 只在正标准差或单元素 correction-zero 恒零函数上准入；多元素零方差切片仍拒绝。var_mean 必须完整返回方差和均值，其 VJP 必须同时接收两路完整 upstream。vector norm 的零点没有唯一实数导数，因此拒绝该 VJP，而不沿用框架的零梯度约定。适配器先严格核对 dtype、维度、keepdim、correction、order、上游形状和全部结构化输出，再计算参照；独立有理数循环、标量目标的精确对称差分与100位平方根复算不调用参照内部帮助函数。55个合同样例相容，20个错误实现被拒绝；其他 variance/std/var_mean correction 与重载、已弃用 `torch.norm`、其他 vector-norm order/dtype、非有限值和其他未审范围继续拒绝。证据见 `evidence/reduce_implementation.md`。

MATHFN 的旧草案同样混入了证据不足的范围：任意整数输入被称为精确提升，无穷输入和多个奇点边界被统一选择为扩展实数值。当前合同只准入一个有限浮点张量、绝对值不超过100的 functional exp/log/log1p/expm1/sqrt/rsqrt/sin/cos/tanh/sigmoid/erf；log、log1p、sqrt、rsqrt 分别使用明确的实数域。整数、float8、无穷、范围外输入、out/in-place 调用继续拒绝。独立768位 Arb 求值覆盖十类函数，erf 另由带余项的有理积分级数核对；22个合同样例相容，8个错误公式被拒绝，完整比较核对 dtype、shape 和全部坐标。证据见 `evidence/mathfn_implementation.md`。

ACT 的两个 profile 也完成来源、公式、输入域和实现复核。PyTorch Softplus 的严格 `beta*x > threshold` 分支属于公开任务；原先所谓 `softplus_formula` 现只在调用者明确选择另一数学任务时使用。`beta*x == threshold` 走公式分支，但这里的 backward 是 PyTorch 定义的操作，不能误称为不连续前向式的普通导数。`tanh_backward` 与 `sigmoid_backward` 只有在第二个参数确为保存的前向输出时才是相应 VJP；新加入的 `vjp.aten.tanh` 与 `vjp.aten.sigmoid` 则明确从原始输入求导，防止端到端调用把 `x` 错绑成保存输出。准入范围收紧为有限浮点张量及相同形状的多输入 backward/GLU；比较器核对 dtype、shape 和全部 GLU 梯度。独立 dual-number 与 Arb 复算通过，37个合同样例相容，20个错误实现被拒绝。真实 Inductor SiLU 在24个固定输入上获得完整参照：正常式的 Gπ−f 未检出，加入 `+0.01` 后 Gπ−f 明确非零；两者都检出很小的 K−Gπ 数值差异。随后按当前 `aten.silu` 图审核并保留生产计划和调用。正式入口建立16,728个参考坐标且无失败分类；Gπ−f 没有坐标排除零，K−Gπ 的 R1、R2 检出约 `1e-8` 量级的负向平均效应。证据见 `evidence/inductor_silu_binding.md`。

PyTorch 2.10 Inductor ReLU、tanh 与 sigmoid 又分别完成公开前向和完整输入 VJP 绑定。每个输出在24次输入上建立98,304个严格参照坐标，完整率和分辨率均为1。ReLU 的三层差异逐点为零，统计因零方差保持退化；tanh 前向在 K−Gπ 与完整 K−f 上检出正向作用，VJP 未确认稳定方向；sigmoid 前向检出固定及对齐方向作用，VJP 在R2/R3检出正向作用。三个调用的 Gπ−f 均无坐标排除零，说明这些数值结论没有由任务公式错绑造成。范围只覆盖声明的版本、float32 `[32,128]`、连续布局和抽样分布。证据见 `evidence/inductor_common_activations_binding.md`。

正式合同路径随后补上未声明张量来源的拒绝。绑定计划虽然能核对顶层输入、输出和状态，但调用仍可能从闭包、全局变量或模块状态读取另一个张量；若该来源到达公开输出而未被绑定，参照实际是在比较不同任务。工具现在利用同一条 Triton 捕获依赖链逐输出核对所有张量类型，包括整数索引和布尔 mask；只有已声明输入以及有调用内记录支持的输入拷贝或精确常量可以继续。真实 Inductor softmax 在新增检查下仍完成24个样本且完整输出覆盖为1；故意加入未声明浮点或整数全局张量的两个隔离变体都在第一个样本以绑定失败拒绝，没有生成 bias 结论。该检查限于捕获到的张量依赖，不能代替对 Python 标量、控制流和调用边界的独立复核。

调用映射的完整性又补了三项拒绝规则。收到的张量或结构化参数必须整体映射，不能用其中一个元素或字段冒充整个输入；严格输出必须保留参照结果的全部嵌套字段和位置，例如有权重的 NORM 不能只比较 `grad.x` 而漏掉 `grad.weight`；空维张量以显式 shape 为准，因此 `[0, 2]` 与 `[3, 0]` 不会因嵌套值都是空列表而丢失后续维度。张量 dtype/shape 继续参与完整输出检查，但固定样本的数值差只统计实际数值元素。

K14_COLL 草案现另外定义六个有限实数 collective 数值任务：all-reduce sum、按参与者数平均、显式有理权重平均、显式整数计数平均、等长分片 reduce-scatter sum 与等长输入 all-gather。保留的本机 PyTorch 2.10 `distributed_c10d.py` 只支持接口、参与者输出和等长分片约定；它不证明设备上的精确实数算术。每项任务都要求显式 operation、非空且唯一的参与者顺序、完整矩阵，以及与参与者一一对应的权重或计数；先完成结构检查再运算。整数计数不截断，零总权重/计数为数学无定义，输出包含每个参与者的完整结果。count-mean 已用逐样本展开独立复算；完整输出比较会拒绝缺参与者、缺坐标、多余字段和任意非零精确差异。这些是任务定义与开发核对，尚未绑定真实 process group、通信、异步状态、内存写回或 Triton 区域。

分片合并的后续核查纠正了此前“空分片等既有检查”的笼统表述：PAGEDATTN 只定义完整分页读取后的 attention。现已把 `kernel_spec/attention_partials.py` 接入顶层 K14_COLL 草案：依据保留 VLLM_UA 第591–645、734–764行的未归一化累加值及 T_SDPA 的 softmax 定义，明确区分 shifted numerator/mass 与 normalized output/LSE；空质量独立表示，只有明确标为空的边界转换才接受 infinity sentinel。所有分片先检查完整维度，空分片不能遮蔽错误字段或缺坐标；全空须明确选择 undefined 或 zero。内部状态保存所有坐标和有理数区间端点，恢复允许向外扩宽，不使用显示小数。专项检查包括1/3精确结果、空分片、错误输入拒绝、512位 Arb直接指数和的独立复算及保存往返。K14 当前16个合同样例相容、六个错误变体被拒绝；完整输出比较不把区间包含当作精确相等。审核、实现准入、kernel绑定均未完成；唯一参与者ID不证明键集合互不重叠且完整，原先两个profile要求LSE；新增scaled_numerator_out依照保留源码第770–776行只声明完整out，不要求不存在的LSE写回。公共schema、检查器和运行时共用profile的输出声明；查询不能临时过滤输出。这仍不证明实际区域只包含该写回。K14 的 GEMM collective、EP dispatch/combine、log2接口及设备状态对应尚未迁入；旧代码继续隔离，不能宣称K14或原始合并要求已全部完成。

## 2026-10-18 原始规格审查要求的未完成项

真实producer→merge后续证据见 `evidence/partial_merge/producer_observed.json` 和 `producer_independent.json`。33个query的固定causal任务逐项验证了key分区和全部分片坐标，确认空分片存在M=-inf/L=1和M=0/L=0两种表示，不能仅凭L=0转换。当前PAGEDATTN和K14参照均与独立定义一致；设备每次1056个输出中保留122个K-f差异，最大绝对值1/262144，差异定位到merge阶段。后续 `producer_composed_exact_values.json` 完成两次完整链Gπ求值，每次8+33个程序实例，三份分片参照状态按实际字节及版本记录连续传递，没有重载设备中间值。122个K值位于Gπ包围之外，是限定输入上的数值差异见证；全部f仍只被包围，未证明Gπ=f，不能称语义bias，也不从该样例推导全域producer绑定。

K14 的真实函数调用证据保存在 `evidence/partial_merge/`：原样提取保留的 vLLM reduce_segments 及两个 helper，在 RTX A6000 上核对两组人工分片输入。每次48个输出坐标在两种初值下均与独立有理数计算精确一致，每组14个非输出存储位置保持初值，全部实际输入位模式前后相同。标准库核对器从实际输入复算，五项记录完整性/错误标记测试通过。这只覆盖固定的等 shift、非FP8调用；后续保存实际调用返回的TTIR并完整求值：每次6个程序实例无中止，分片输入读取与24个输出写回均对应任务，无其他缓冲区写入；Gπ包围包含任务值但宽度非零，等式仍未建立。完整producer分区和全域写集合尚无证明，仍不计为完成绑定或可信覆盖。

顶层合同的开发检查通过，不表示 10 月 16 日包中 14 份合同已完成审阅迁入。A1_LINEAR 仍是未准入草案；F2_MX 已收窄到版本固定的 torchao 0.16.0 MXFP8 FLOOR 任务并完成实现复核与一份真实调用绑定。旧 `drafts/v2_5` 保持原样，不能作为运行时回退。

本次重新执行隔离副本的反例，结果见 [复现记录](../../.cache/original_spec_gaps_current.json)，脚本为 `.cache/reproduce_original_spec_gaps.py`（仅审计，不是新的运行入口）：

| 原始要求 | 当前复现 | 仍需完成 |
|---|---|---|
| A1 传入 S0 并保留后态 | q=k=1、v=3、S0=2，旧适配器仍返回 (3,3)，应为 (5,5) | 新 A1 适配器已保留完整状态；独立审核仍待完成，旧适配器不能启用 |
| A1 默认 scale 一致 | dk=4，逐步递推输出 1/2，分块输出 1 | 新 A1 各路径已共用参数解析并核对反向；合同仍待审核 |
| 未归一化子计算与完整论文算法分开 | 当前 A1 草案区分七个核心 profile，尚未通过准入 | 明确 unnormalized_core；完整归一化任务须包含 Z 及分母，不能以 GDR 合同代替 |
| MX 合法尺度选择分开 | 旧适配器忽略输入中的 scale_rule；ceil_pow2 仅作为替代读法存在，两种结果确实不同 | 活动 F2 只定义 torchao 0.16.0 的 FLOOR profile；RCEIL、CEIL、EVEN 和旧 ceil_pow2 不参与本次裁决 |

A1 当前实现位于 `kernel_spec/linear_attention.py`，delta 递推复用 `d2_linear.py`。顶层 `contracts/A1_LINEAR.yaml` 与 `adapters/A1_LINEAR.py` 现固定七个明确选择的核心 profile：未归一化、含 S/Z 的参数化归一化、标量/逐 key 衰减，以及 delta/gated_delta/KDA。它们不是完整模型或 FLA API；所有 profile 都要求具体任务选择。前向返回全部 token 输出和末态，反向返回全部输入梯度并计入末态梯度，变长接口保留空段位置。默认 scale 共用 1/sqrt(K)，epsilon 必须显式给定，对数门必须显式选择 ln/log2。

32 项核心专项检查包含独立因果矩阵/增广仿射矩阵展开、精确梯度核对、严格包围、零衰减和空段。旧 S0 反例在新适配器得到完整正确结果，零衰减的旧分块除零反例见 `.cache/linear_decay_legacy_zero.json`。A1 的 13 个合同例子相容、6 个错误变体被例子拒绝；这些不是认证。适配器另检查全部参数适用性，不能通过未知字段或缺失梯度得到部分答案。

已保留六份本地 FLA 支持源码及许可证，登记为 `A1_LOCAL_*`；没有新增哈希。其上游版本和本地修改尚未独立核实，来源记录明确说明这一点。Gated Delta 的公式另引用已有 GDN_PAPER/FLA_GDR_NAIVE。Katharopoulos 原文和完整 FLA 输入转换、head/layout、存储效果仍未审定。A1 的 review=drafted、implementation=pending、binding=pending；运行时即使 allow_draft 也不能启用它。完整输出比较现已接入，核对全部前向状态与反向梯度；区间包含不代表相等。实际使用仍须通过审核和真实绑定。


F2_MX 的旧宽泛草案已归档为 `contracts/draft/F2_MX_pre_torchao016_20261019.yaml`。活动合同只定义安装的 torchao 0.16.0 `to_mx`：有限物理 f32/bf16 输入、最后一维每32项一块、E4M3/E5M2、FLOOR 和普通 scale 布局。它完整返回 E8M0 code 与每个 FP8 数值元素，并保留该版本在 code 0 时用 2^-126 作转换除数、同时返回表示 2^-127 的 code 0 这一可观察行为。RCEIL、CEIL、EVEN、FP4/FP6、任意轴、尾块、swizzle、特殊值和其他版本均拒绝，不把另一种合法算法判错。

三份已安装 torchao 源文件和许可证已保留；独立测试用 float32 位字段推导尺度，并用整数格式枚举复算元素，没有导入官方实现产生期望值。13个合同例子和4个错误变体通过，14组 f32/bf16、E4M3/E5M2 官方函数对照逐项一致，其中包括两块、全零、固定随机值和 code-0 非零边界。当前合同与适配器的复核副本已保留，profile 已通过运行时准入。该结果建立的是版本固定的框架任务，不是完整 OCP 标准的替代品。

本地来源后续核查：九份 A1_LOCAL_*/MX_LOCAL_TRITON_* 保留文件均与本地检出提交的文本一致（直接比较，未生成摘要）；这排除了未说明的本地修改，未证明远端来源。早期 MX 审核曾按一个自定的 clamped real conversion 目标发现 FP4、ROUND_DOWN、amax=2^-127 反例，记录在 `.cache/mx_source_boundary_audit.json`。该目标已不再是活动 F2 规格；记录只作为发现边界问题的历史材料，不能用来认证 torchao 0.16.0 或完整 OCP 语义。

旧目标下的设备记录仍保存在 `evidence/mx_boundary/observed.json`，且复跑入口拒绝覆盖已有输出。新的 torchao 0.16.0 任务另在 RTX A6000 上绑定 E5M2 FLOOR：安全输入完全一致；有限饱和范围内有坐标写成正负无穷；code 0 的非零输入应得到0.5而 kernel 写成0。E4M3 路径因当前设备/compiler 不支持所请求类型而不能编译。具体裁决及其范围见 `evidence/triton_mxfp8_e5m2_floor_binding.md`。

上述缺口没有被新增 ATen 合同或它们的通过数量抵消。A1 仍须保留并逐条核对线性注意力论文；若将来要声称通用 OCP MX 合规，也仍须取得并逐条核对相应标准原文。当前 F2 仅以已保留的 torchao 0.16.0 源码定义版本固定任务。代理 localhost:27891 仍拒绝连接，因此没有绕过指定代理补取材料。

## 2026-10-18 新捕获的完整输出核对

对 `.cache/spec_region_entry` 中 OPT-350m、bf16、2 层、128 tokens 的已有新捕获，按 manifest 保留全部 26 次已保存启动。环境为 PyTorch 2.10.0、Triton 3.6.0、Transformers 4.57.3。没有把旧捕获缺失的执行标志补为通过，也没有覆盖历史记录。

26 项都有精确源码候选；25 项的图运算在当前目录中，1 项含尚不支持状态组合的 `aten.index_put_`。进一步检查中，20 项通过当前输入、输出和存储结构检查，6 项仍有来源或张量对应缺口。在这 20 项中，17 项超过完整 IR 写入检查的程序预算，3 项完成该检查。这里的“结构检查通过”不等于正式语义绑定。

| 捕获 launch index | 计算 | 完整输出坐标 | 设备与参照核对 |
|---|---|---:|---|
| 0 | 融合 attention mask 构造 | 16,384 | 两次重放字节一致；固定规则模拟全部写入逐位一致；全部输出与图参照精确相等 |
| 1 | add | 128 | 同上 |
| 11 | constant_pad_nd | 129 | 同上 |

三个计算在 eager 与 inductor_default 两种读法下都完成全输出比较。`reference_status` 为 `complete`，`binding_status` 和 `semantic_verdict` 仍为 `not_established`：这些是固定捕获输入上的开发证据，未替代来源审阅、各基础合同的实现认证或设备内存访问轨迹。其余 23 项没有被删除，也没有因这 3 项通过而升级。

逐项记录：[全部 26 项结构检查](../../.cache/spec_region_entry/current_binding_all26.jsonl)、[3 项设备核验](../../.cache/spec_region_entry/small_launches_device_check.json)、[3 项完整参照比较](../../.cache/spec_region_entry/small_launches_full_reference.jsonl)。候选来自 `units_for_binding.jsonl` 与 `candidates_for_binding.json.gz`；设备核验之后另存 `candidates_with_small_launch_checks.json.gz`，原候选未改写。这些 `.cache` 原始材料目前只保存在工作区，尚不是冻结的发布证据。

上述 6 项单启动对应缺口中，5 项属于 launch 12–16 的分块 softmax/NLL 区域，已有该区域的组合证据，不能通过猜测多结果索引来逐个绑定。剩余一项属于 embedding backward 的 launch 44–46。区域检查原先因 `(None, 梯度张量)` 中的 `None` 被当成缺失节点而拒绝；现要求图与 wrapper 在同一返回位置都显式为 `None`，并将该位置单独保留，不计入存储覆盖。修后该区域的捕获连续性、来源、输入、输出及局部临时存储结构通过，记录见 [embedding 区域检查](../../.cache/spec_region_entry/embedding_region_absent_output.json)。其图仍含 `aten.index_put_`，状态组合的数学求值、完整 G 比较和设备核验未完成；没有将原地操作改名，也没有升级语义结论。

embedding 局部更新的后续核对：T_FULL 与 T_INDEX_PUT 的保留定义，加上任务图中 `full_default_1` 仅作为一次 `index_put` 的目标被引用，支持 GR.EFFECTS 中新增的受限情况。`functional_value_graph` 先检查新分配、唯一使用、没有别名、没有返回旧名字或 `out=`，才生成非原地的值计算图；原图保留，外部输入和共享张量继续拒绝。完整与惰性求值共用该检查。真实区域的条件见 [局部更新核对](../../.cache/spec_region_entry/embedding_region_local_update.json)。独立 PyTorch 原地计算和拒绝例子的回归通过；GRAPH 仍为草案。

该区域随后完成全输出数学核对：`check_region.py --evaluate-task` 使用共同的 `region_reference.compare_task`，检查全部声明输出、类型、形状和每个坐标，并分别保留输出与求值预算不足。真实 embedding 区域的 2,099,200 个坐标全部与 eager 任务参照精确相等，共使用 6,691,201 次标量求值，见 [完整参照结果](../../.cache/spec_region_entry/embedding_full_task_reference.json)。另一路不调用图参照的 index_put 或舍入函数，按捕获索引用 Fraction 累加，再以整数商余数实现 bf16 最近偶数舍入；全部坐标也相等，见 [独立复算](../../.cache/spec_region_entry/embedding_independent_reference.json)。这支持该输入上的 K 与 f 相容，不认证所有 INDEX/CAST 查询。

设备检查执行全部 4,356 个程序实例：三个启动各两次 GPU 重放均与捕获字节一致；初始化和最终转换各完成 2,099,200 个写入坐标的模拟与逐位核对。中间累加启动的 256 个程序全部因模拟器尚不支持 `tt.atomic_rmw` 而退出，见 [设备检查结果](../../.cache/spec_region_entry/embedding_device_check.json)。因此完整 G 比较与设备对应仍未建立，不能把重放一致或 K 与 f 相等升级为没有语义 bias。新增区域参照测试覆盖错误输出、缺失输出、类型/形状不符及预算耗尽；当前完整回归 412 项通过，日志为 `.cache/region_reference_regression.log`。这些计数是开发检查，不是数学证明数量。

## 2026-10-18 动态参照复核

MATMUL 的 `real` profile 已完成来源、公式、输入域和实现复核。`tests/test_matmul_exact_derivatives.py` 按保留的 PyTorch 2.10 T_MM、T_BMM、T_MATMUL、T_LINEAR、T_ADDMM 前向定义，使用 NumPy object 数组和 Fraction 求值，不调用参照的 VJP、转置梯度或广播归约函数；35 个配置核对全部操作数梯度、形状、dtype 及缺省偏置的 None。复核发现原草案会把定宽整数矩阵乘法当作无限精度整数任务，还把 addmm/baddbmm 的 beta、alpha 位置参数及未声明的 out_dtype 静默接入；现已把准入范围收紧到有限 f16/bf16/f32/f64 的默认函数重载，其他 dtype、独立 dtype/out 重载、稀疏和非有限乘数均拒绝。`beta=0` 只允许跳过 addend 的数值，仍核对其 dtype 和 shape。完整证据见 `evidence/matmul_implementation.md`；22个合同样例相容，9个错误实现被拒绝。

直接 Triton float32 matmul 随后用24个固定输入跑通隔离测量。基线的 Gπ−f 未检出；只在累加结果加入 `0.01` 后，Gπ−f 在6144个输出坐标上全部排除零，固定均值规则给出非零，而 K−Gπ 仍未检出稳定均值。这支持工具把注入项归到任务公式，而不是归到普通浮点执行。随后依据 Triton 3.6 的 `tl.dot` 定义、实际二维索引、全部输入与写回完成审核并保留生产计划和调用。正式入口建立6144个参考坐标且无失败分类；Gπ−f 没有坐标排除零，K−Gπ 与完整 K−f 当前投影均未检出稳定均值。范围限于16×32乘32×16的连续 float32 输入和 ieee dot。证据见 `evidence/triton_matmul_binding.md`。

高级索引读写复核：完整与惰性路径共用索引类型和广播检查，int32/int64 作为位置索引，bool/uint8 掩码明确 unsupported；写回要求源与目标 dtype 相同、accumulate 为布尔值，并核对完整源形状。非累加的重复写入使整个结果无定义，布尔累加使用布尔加法。惰性 index_put 现复用读取坐标映射，支持非前导、非连续索引及空选择；手工坐标检查和 1,075 个本机 PyTorch 2.10 配置对照通过（`.cache/index_put_nonleading_native.json`）。这组对照未包含越界索引：对于未索引轴为空且索引越界的情况，本机 CPU 会跳过范围检查，参照仍按 INDEX 的已声明范围拒绝，不据此扩展合法域。相关测试见 `tests/test_index_put_domains.py`、`tests/test_advanced_index_domains.py`、`tests/test_index_put_nonleading.py`。这些是限定范围的开发证据，未认证所有 INDEX 查询、原地存储效果或真实 kernel 绑定。

卷积参照的完整性复核：反向现在先核对上游梯度与完整前向输出形状一致，不能遗漏额外 batch、通道或只计算部分空间坐标；完整、惰性前向与反向共用空间参数、分组和输出形状检查，前向偏置必须恰好覆盖输出通道。`output_mask` 按本机 ATen schema 要求三个布尔标志，始终保留三个梯度位置，未请求者为 None；全部 8 种 mask 及分组、膨胀卷积的独立 autograd 对照通过。另检查 324 个一维／二维配置，两条路径对 308 个配置的形状、完整值或拒绝结果与原生算子一致；16 个零 dilation 配置保留 not_established，而非计为通过（`.cache/convolution_domains_native_classified.json`）。原生算子在部分空 batch 上接受零 dilation，保留文档尚不足以确定统一域，localhost:27891 仍无法连接；零输入通道也保持未建立。此修复不认证所有 dtype、反向 bias_sizes 或真实 kernel 对应。

同 dtype 卷积的后续核对：前向不再把整数输入自动标成 f32，完整与惰性路径保留整数运算和结果 dtype；六种 dtype、分组、偏置、空 batch 的 48 个配置及一个超过 2^53 的 i64 案例与原生算子一致（`.cache/convolution_dtypes_native.json`）。混合 dtype 的原生行为依赖空 batch、偏置等条件，转换规则未选定时返回 not_established；布尔卷积和整数梯度返回 unsupported。整数溢出、所有浮点舍入行为仍不在这组证据的认证范围内。

反向 bias_sizes 不再被无条件忽略：按已有 CV.VJP 中每个输出通道对应一个偏置的定义，缺省值或 `[C_out]` 可以继续求值；非整数尺寸拒绝，其他尺寸返回 not_established，未请求偏置梯度时也不以 mask 掩盖元数据矛盾。本机 CPU 路径会忽略若干不一致尺寸，记录在 `.cache/convolution_bias_sizes_native.json`；该行为不作为普遍任务约定，参照没有将这些原生调用判为语义错误。正常情况下仍固定保留三个梯度位置及 None。

CONV 随后完成独立来源、schema、域和实现复核。审查发现参照曾为实际 schema 必需的 `output_mask` 私自提供默认值，并且合同在没有保留 Conv3d 定义时宣称覆盖3-D，还把定宽整数实现与无限精度整数任务混在一起。当前准入只覆盖批量非转置 1-D/2-D、同 dtype、有限 f16/bf16/f32/f64 的默认函数操作；output_mask 必须显式给出，完整比较核对全部三项梯度及 None。独立 Fraction 直接枚举和精确差分复算四组普通/分组、1-D/2-D 前向与全部梯度；13个合同例子相容，5个错误实现被拒绝。直接 Triton grouped Conv2d 随后完成正式绑定，24个调用的11,520个坐标全部建立严格参照并完整追踪三个输入来源和写回；Gπ−f 没有坐标排除零，K−Gπ 与K−f当前规则均未确认稳定均值。只在写回增加 `0.01` 的受控版本使全部11,520个任务残差排除零，R1、R5检出非零，K−Gπ仍未确认。正式运行约257秒，证据见 `evidence/conv_implementation.md` 与 `evidence/triton_conv2d_binding.md`。该具体调用不认证其他卷积配置。

直接 Triton 行求和也完成正式绑定。计划固定连续 float32 `[32,256]` 输入、沿第1维求和、完整 `[32]` 输出和未修改输入状态；每行由一个 Triton program 读取并调用 `tl.sum`。24个调用共建立768个严格参照坐标，全部 TTIR 程序和写回均被覆盖，Gπ−f 没有坐标排除零。设备结果与任务值的最大绝对差约为 `4.86e-6`，当前统计规则未确认稳定 bias。只在求和后增加 `0.01` 的受控版本使768个任务残差全部排除零，平均残差约为 `0.01`，而 K−Gπ 仍未检出稳定差异。证据见 `evidence/triton_row_sum_binding.md`；该结果只认证这一具体形状、dtype、轴和实现。

直接 Triton 逐元素 exp 随后完成正式绑定。计划固定连续 float32 `[1024]` 输入、完整同形输出和未修改输入状态；一次启动的四个 program 覆盖全部坐标。24个调用共建立24,576个严格参照坐标，全部达到第一级分辨率且无失败分类。Gπ−f 的均值和最大绝对残差均为零；设备到程序及完整设备到任务比较则检出很小的负向数值偏差，R1、R2 的投影均值约 `-1.21e-7`，完整 K−f 平均绝对残差约 `6.57e-8`。这属于 `tl.exp` 在声明输入分布上的数值结果，不是任务公式错误。只在 exp 后增加 `0.01` 的版本使24,576个任务残差全部排除零，平均绝对任务残差约 `0.01`。证据见 `evidence/triton_exp_binding.md`；不能据此推断其他形状、输入分布、数学函数或融合实现。

CAST 的旧 profile 把浮点、整数、布尔、float8、非有限值和多个未说明的舍入约定放在一起，无法形成单一可审核任务。当前 `finite_rne` profile 只接受有限 f16/bf16/f32/f64 输入，要求输入记录可由声明的源 dtype 精确表示，并明确选择 IEEE roundTiesToEven；结果必须有限。独立整数位长算法覆盖中点两侧、正负 tie、次正规数、正规边界、最大有限值、扩宽和恒等转换，不复用适配器的舍入帮助函数。12个合同样例相容、4个错误实现被拒绝，另与本机 PyTorch 2.10 在全部16种源/目标组合的2048个坐标上逐位相同。旧的宽范围合同保留在草案目录，不参与准入。证据见 `evidence/cast_implementation.md`。

直接 Triton float32 到 bfloat16 的显式 RTNE 转换也完成正式绑定。24个调用的24,576个坐标全部建立并达到第一级分辨率，K−Gπ、Gπ−f 和完整 K−f 的最大绝对差均为零。默认程序参照会把任务要求的舍入错误地拆成两段相反差异，因此该纯转换绑定显式选择已有的 `rounding_check` 程序参照；报告逐项记录该选择，它只改变两段差异的归因，不改变 K−f。临时受控版本在转换前加 `0.25` 后，K−Gπ 仍为零，全部24,576个 Gπ−f 坐标排除零，平均绝对差约 `0.249998`，固定均值规则确认非零效应。证据见 `evidence/triton_cast_bf16_binding.md`；其他转换、融合算术、隐式舍入与特殊值仍未绑定。

ARITH 的旧合同把有限浮点、固定宽度整数、布尔、无穷、`pow` 和多种实现选择放在一个 profile 中，还允许不能由声明 dtype 表示的数学输入，无法给普通逐元素 kernel 提供可靠裁决。当前 `real` profile 收窄为二十个常见 functional 操作，只接受有限、同 dtype、物理可表示的 f16/bf16/f32/f64 输入。适配器先检查准确调用重载、输入表示、广播、布尔 mask 和完整输出，再计算精确实数公式。34个合同例子相容、12个错误实现被拒绝；独立 Fraction/NumPy 复算覆盖160组广播输入，PyTorch 2.10 对照核对四种 dtype 的形状与存储结果。证据见 `evidence/arith_implementation.md`；整数溢出、混合 dtype、特殊值、`pow` 和内存效果仍拒绝。

直接 Triton float32 逐元素乘加也完成正式绑定。计划固定三个连续 `[1024]` 输入、完整同形输出及三个未修改输入状态；一次启动的四个 program 覆盖全部坐标。24个调用共建立24,576个严格参照坐标。正常实现的 Gπ−f 全部为零；K−Gπ 的平均绝对差约 `2.32e-8`、最大约 `4.13e-7`，当前统计规则未确认稳定均值。只在结果加入 `0.25` 后，K−Gπ 仍保持普通浮点误差量级，全部 Gπ−f 坐标排除零且平均绝对差恰为 `0.25`，完整 K−f 的规则确认非零平均效应。证据见 `evidence/triton_multiply_add_binding.md`；这不认证广播、其他 dtype、其他形状或任意同名 kernel。

LAYOUT 原先把逻辑布局、创建、dtype 转换、复制和写状态混在一个 profile 中，还接受不能由声明 dtype 表示的数学值。当前 `real` profile 只保留有限 f16/bf16/f32/f64 张量的常见函数式逻辑值与形状变换；另有 task-bound constant-pad 与 Tensor-list cat 完整前向/VJP profile。当前44个合同例子相容，18个错误实现被拒绝；独立 NumPy 坐标复算和 PyTorch 2.10 四种 dtype 对照通过。证据见 `evidence/layout_implementation.md`；storage、stride、alias、创建、dtype 转换及未声明内存效果仍拒绝。

直接 Triton float32 矩阵转置随后完成正式绑定。计划固定连续 `[32,64]` 输入、完整 `[64,32]` 输出及未修改输入状态；一次启动的八个 program 覆盖全部2,048个坐标。24个调用共建立49,152个严格参照坐标，正常实现的 K−Gπ、Gπ−f 与完整 K−f 逐点为零。统计规则因全零样本不作总体等价声明。只在每个输出加入 `0.25` 后，全部 Gπ−f 坐标排除零，语义方向规则与完整输出规则确认非零平均效应。证据见 `evidence/triton_transpose_binding.md`；这不认证非连续输入、存储别名、其他形状、dtype 或任意同名 kernel。

LOGIC 随后按保存的 PyTorch 2.10 运算文档和原生 schema 完成来源、调用边界、物理输入与完整布尔输出复核。两个 profile 只覆盖已选的常用比较、逻辑、布尔位运算、truth reduction 和浮点 `isnan`；整数位运算、混合 dtype 和 NaN 比较保持拒绝。直接 Triton float32 大于零比较完成正式绑定，24次调用共24,576个坐标全部精确相等。测量层只在严格布尔验证后使用 False=0、True=1，并将误判计数作为首要结果；阈值改为0.25的临时版本检出2,474个假阴性，K−Gπ保持为零而 Gπ−f 与完整 K−f 显示任务公式差。证据见 `evidence/logic_implementation.md` 与 `evidence/triton_greater_zero_binding.md`。

INDEX 随后将原来混合查询、scatter、更新、反向累加和不确定重复写入的草案收窄为默认只读 gather、index-select 与 embedding 前向。保存的 PyTorch 2.10 文档、ATen schema 和 `Embedding.cpp` 支持坐标、形状与调用边界；物理有限四种浮点 dtype 和 int64 索引经独立 NumPy 循环与 PyTorch 对照。直接 Triton float32 行 index-select 的24次调用共49,152个坐标全部精确相等。临时版本在查得值后加入0.25，全部 Gπ−f 坐标排除零，完整 K−f 固定规则确认非零效应；普通间接寻址仍由 K−Gπ 单独核对。证据见 `evidence/index_implementation.md` 与 `evidence/triton_index_select_binding.md`。

未实现参数不再静默丢弃：创建、复制等参照中的 `**kw` 原先会接受 `out=`、未知布局及拼错的参数名，图的部分惰性路径也会忽略这些参数。算子入口、合同适配器和图结构检查现拒绝没有明确实现的关键字，返回 `unsupported`，不把它误称为输入非法。23 个静默接受反例修前失败、修后通过；真实 embedding 区域的 8 个节点按已绑定输入重新做结构检查仍通过。该检查只证明参数没有在这些入口被静默忽略，不证明所有已接受参数的语义都已认证。

调用参数的后续核对：图与合同入口共用完整签名检查，多余位置参数、重复传参、未使用节点缺少必需参数均明确拒绝；完整和惰性求值先按合同签名整理命名参数，再选择是否忽略中间浮点转换。由此修复命名 cast 在选择读法时访问空位置参数列表的问题。copy 按本机 ATen schema 显式保留布尔 `non_blocking`，不再用 `*args` 吞掉其余参数；这只处理逻辑值，不认证调度与异步内存效果。

填充值转换存在待选语义：本机 PyTorch 2.10 的原生 fill/full/constant_pad_nd 对整数目标的 ±1.5 向零截断；保留的 TS_REFS.fill 及本机对应函数则拒绝 float 到整数，连 1.0 也拒绝。完整与惰性 padding 目前对此存在分歧，不能用于这类查询的语义裁决；已向任务负责人询问是否采用原生算子的标量转换，未自行选择，复现见 `.cache/fill_semantic_choice.json`。这与 CAST 已有的 Tensor.to 规则不是同一来源问题，不能仅引用 CAST 便声称所有创建算子自动采用该规则。

索引与切片写回的完整性复核：select 不再把越界索引取模；slice 在空输入上也验证轴、边界类型和正步长。select_scatter 与 slice_scatter 在完整、惰性两条路径共用目标范围和源张量精确形状检查，不再广播不完整源张量或丢弃多余元素。依据为 LY.SLICE、T_SELECT、T_SELECT_SCATTER 与 T_SLICE_SCATTER 的保留定义。合法负索引、空切片及空的非索引维度保留；测试检查全部输出和输入未被改写。本机 PyTorch 2.10 对照分别覆盖 1,015 个读取配置、2,025 个写回配置，两条路径的合法域、形状与全部坐标一致，记录在 `.cache/select_slice_native.json`、`.cache/slice_scatter_native.json`。这些是逻辑值参照的证据，不认证实际 kernel 的内存写回范围。

惰性布局求值复核：`lazy` 原先接受越界或重复排列轴、不能广播的 expand、无法唯一推断的空 reshape，以及三维 t；标量 transpose 则除零。完整与惰性求值现共用 view、permute、expand、squeeze 的尺寸与轴规则，unsqueeze/transpose 共用轴校验；形状乘积使用 Python 整数，验证不读取输入坐标。新增反例修前出现 12 个失败和 1 个错误，修后布局专项 11 项通过。本机 PyTorch 2.10 对照 774 个配置，772 项的合法域、形状和值一致；另两项标量 expand 到 `[0,-1]`、`[-1,0]`，本机返回含负尺寸的空张量，而 T_EXPAND 保留文档禁止新增轴使用 -1，因此参照继续拒绝，未按本机行为放宽合同。结果见 `.cache/lazy_layout_native.json`。本次仅核对逻辑值和尺寸，不认证存储别名或 view 的 stride 条件。

cumsum 复核：按 T_CUMSUM 的前缀和及“先转换输入”定义，修复轴越界取模、标量除零和 dtype 转换遗漏。默认整数/布尔输入仍提升为 i64，全部前缀坐标、空张量形状有回归检查。显式布尔输出若实际需要累加，当前返回 not_established：原参照会产生标为布尔的整数计数，而本机 PyTorch 2.10 CPU 未实现该累加；未据此自行选择布尔累加语义。无需累加的标量、空张量及长度为一的轴返回转换后的值。此未决项需来源或任务选择进一步明确。

归约 dtype 复核：T_SUM、T_MEAN、T_PROD 的保留文档明确指定 dtype 时先转换输入。sum/mean 原先未兑现浮点格式转换，prod 忽略 dtype；现复用 CAST 转换，再进行合同声明的实数归约。mean 拒绝没有浮点结果类型的查询，prod 的整数/布尔输入默认提升为 i64；显式布尔 sum/prod 返回布尔值及相应空集单位元。90 个含空输入的类型配置与本机 PyTorch 2.10 的值、dtype、拒绝/无定义分类一致（`.cache/reduction_dtype_torch_crosscheck.json`）。这不等同于认证全部舍入过程或整数溢出行为。

算术缩放复核：`kernel_spec.aten.add` 原先将整数结果的非整数 alpha 截断，且布尔加法忽略 alpha=0。现按运算提升后的结果类型，在逐元素计算前验证 alpha，空张量也不能跳过验证；布尔零缩放不再引入第二个输入。保留的 T_ADD/T_SUB 文档与 TS_REFS 实现说明缩放及类型约束；216 个整数、浮点、布尔混合与空张量配置和本机 PyTorch 2.10 的数值/拒绝结果一致，记录在 `.cache/arithmetic_alpha_torch_crosscheck.json`。此检查不涉及整数溢出或全部类型提升规则的认证。

布局参照的合法域复核：根据保留的 T_VIEW、T_RESHAPE、T_PERMUTE、T_EXPAND、T_T 文档，修正了 `kernel_spec.aten` 中四类错误接受：空张量 reshape 的 `[0,-1]` 无法唯一推断、expand 新增轴使用 `-1`、permute 越界轴被取模、t 接受三维输入。尺寸和轴要求整数类型，shape 乘积使用 Python 整数；合法空张量、负轴排列及扩展的坐标结果有回归检查。本机 PyTorch 2.10 对上述四类非法输入也明确拒绝。随后修复 squeeze、unsqueeze 的越界轴接受，以及 squeeze/transpose 对合法标量的错误拒绝；重复轴在 squeeze 中明确拒绝。六种形状（含标量与空张量）的 804 个轴配置与本机 PyTorch 2.10 对照，形状、值和拒绝结果均一致，记录在 `.cache/layout_axes_torch_crosscheck.json`。这些检查兑现现有 LAYOUT 合同的相关合法域，不认证别名、存储布局或真实 kernel 绑定。

逐位验证的完整输出修复：`2_tool` 的 `emulate.verify` 原先只统计浮点存储，纯整数输出返回未建立；更严重的是，混合输出中整数错误可被遗漏而返回逐位一致。现统计全部已写输出，整数和布尔值直接按存储整数表示比较，不经 float64；未建立的值、集合、条件或未证前提保留拒绝。失败注入先复现“浮点正确、整数错误仍通过”，并覆盖超过 2^53 的整数差异。GPU 回归又发现整数 scan 的通用路径依赖未证结合律，现对支持的布局使用已有的明确执行顺序模型，不删除前提标记来放行。68 项仿真与审查测试在真实 GPU 上通过。旧混合输出的逐位通过记录不能据此自动获得完整输出认证，需要重新验证。

设备访问已确认：早期默认沙箱无法访问 CUDA，但沙箱外设备运行正常，已生成新的同进程来源与设备捕获（见 `corpus/README.md`）。代理 `localhost:27891` 仍拒绝连接。OPT-350M 的两个整数 kernel 已完成各两次逐位重放、固定规则下的完整网格仿真，分别核对 128、129 个输出；对完整实际输入求值后，全部输出与 eager、inductor_default 两种参照读法精确相等。记录在 `.cache/spec_capture_graph/`，属于具体实例的对应和相容性证据；合同审阅、一般设备访问范围和整体语义认证没有自动升级。

公共区间后端复核：`specification_v2/algorithm_specs/phase2/rigorous.py` 的平方分支原先直接使用外部 Decimal 精度，反例包括 1/3 的平方不在结果内、极小正数平方变成零区间。现复用向外舍入乘法及其下溢包围，跨零平方的下界取零；`tests/test_decimal_interval_powers.py` 用精确有理数极值验证。AdamW 当前求值不经过该平方分支，未据此宣称历史 AdamW 结果错误。另用整数平方根构造独立有理数界核查 AdamW 输出包围，覆盖正负更新、非零前态、初始步及零学习率，并补齐缺失输入的拒绝。相关实现验证仍按各自状态处理，修复后端不自动认证所有调用者。

审查回调的输出对应：`audit` 原先展平 callback 结果后按存储地址索引，偏移、转置和切片输出可能比较错坐标。现要求回调与捕获输出逻辑形状一致，并按 stride 与存储偏移映射，拒绝重叠布局和存储外视图；未指定的存储位置不参与参照比较。独立坐标枚举覆盖转置、切片、标量和空张量，审查专项 7 项 CPU 测试通过、7 项 CUDA 测试跳过。报告不再由“阈值内未发现差异”推导“问题出在数值实现”，也不声称未核验的回调使用了实际输入和前态。此修复仅建立输出位置对应，float64 回调仍不是已认证的 specification。这组早期测试在默认沙箱中无法访问 GPU，代理也未连接成功；后续沙箱外设备验证和新捕获见本节开头，在线来源核验仍未恢复。

已审阅训练合同的完整输出比较：ADAMW 与 RANKMEAN 新增 `check_output`，通过现有注册入口调用，不修改任务选择和准入要求。ADAMW 按 AW.C1–C5 核对 theta/m/v/t 的全部坐标；对于有理数候选，令 d=theta_decay-theta_new，当 d 非零时核对 r=lr*m_hat/d-eps、r>=0、r²=v_hat，d=0 则要求 lr*m_hat=0。分母为零仍按合同无定义处理；非负条件避免平方后接受负平方根。RANKMEAN 按 RM.C1 核对 loss*sum(n)=sum(S)，保留完整参与者及零计数验证。新增 7 项测试覆盖全部状态遗漏、None 与零梯度、正负梯度和零学习率、非零微小扰动、平方根符号、无理数目标的近似候选及全局均值。两份合同的现有检查均为 consistent；不自动更新实现准入。比较仅限选定的实数数学任务，不把浮点输出与数学值不相等直接认定为 kernel 语义错误，仍需要 Gπ 与完整任务绑定。

IR 参照写入范围：新增 `reference_write_scope`，直接使用既有解释器的写入记录核对声明输出，包括未改变字节的写入。只执行完整 grid，默认至多 64 个程序实例；超预算、执行中断、理由或附带前提、别名及间接内存未绑定均保留未建立，不以抽样代替完整范围。CPU 反例“输出之外写入原有零值”通过字节差异检查但被此检查拒绝，另覆盖存储外写入与未声明输出。绑定专项 25 项通过。真实 Qwen3 arange 捕获完整执行 1 个程序、记录 128 个写入元素，范围检查通过（`.cache/spec_ir_write_scope_arange.json`）；旧执行证据仍未知，语义未裁决。该结果是 IR 参照执行的范围证据，仍不能自动推导设备实际访问范围或数学任务正确性。

完整存储的变化核验：动态绑定新增 `check_storage_changes`，按捕获的字节长度核对全部前后态，拒绝未声明为输出的存储变化，以及输出视图间隙或视图外的变化。支持非负且不重叠的步长，包括转置和切片；窗口、不完整快照、不支持的布局明确拒绝。测试使用独立坐标枚举核对逐字节归属，绑定专项 24 项通过。对此前 47 个结构对应记录单独读取全部存储，47 项均未发现声明输出之外的变化，结果另存 `.cache/spec_storage_changes_47.jsonl`，每项仍保留执行证据未知、间接内存记录缺失和语义未裁决。变化检查不证明无相同值写入，也不证明未访问捕获存储外的地址，不能替代 IR 的完整访问范围核验。旧生成脚本默认执行与抽样 `bit_identical` 记录不能独自恢复缺失的捕获信息，本轮未自动放行旧数据。

捕获完整性的新限制：`capture.save_launch/load_launch` 原先遗漏 `implicit` 间接内存及 `executed` 执行标志，导致未执行记录读回后变为已执行。CPU 反例已复现并修复：直接和间接内存分别保存前态、后态和窗口，执行标志原样保存；旧文件缺字段时保留未知，重新保存也不能升级证据。重放不支持间接地址重映射时明确拒绝，重放比较要求已执行证据。固定 100 个旧捕获经逐份检查均缺这两项字段，包括下述 47 个结构对应记录。因此下述数字仅表示此前检查到的结构关系，不再满足当前完整捕获准入；不能追补默认值使其通过。新动态入口拒绝执行未知、间接内存是否完整未知及缺少间接内存任务映射的记录。需要已有独立捕获记录补证或新捕获，旧记录不覆盖。相关工具测试 9 项通过、34 项因 CUDA 条件跳过，绑定测试 21 项通过。

固定样本的当前结构检查：沿用 seed=20261016 的 100 个捕获记录，只读取捕获，不重新运行 GPU；结果另存 `.cache/spec_binding_current_100.jsonl`。47 项通过结构对应检查；其余为来源记录缺失 18、输出类型或形状对应缺证据 10、输入类型或形状对应缺证据 2、调用与来源记录不一致 7、来源绑定不完整 9、片段缺输入 6、前态未对应 1。47 项仍全部是 `semantic_verdict: not_established`、`reference_status: not_evaluated`，不能作为可靠参照覆盖率。标量参数、完整写回范围和任务选择仍需独立核验。报告入口另补回归：零样本、空 `--only`、任何指定记录不存在均非零退出且不创建报告；拒绝记录显式保留未裁决和未求值状态。

调用参数复核：新增 `bind_call_scalars`，核对源码签名与捕获参数完整对应、调用中明确传入的标量类型和值；不执行调用表达式，不经 float 转换整数，并区分正负零。模板额外启动参数仅在源码 `inductor_meta` 明确给出 `FixedGrid`、三个参数的顺序，且与捕获 grid 一致时接受；其余额外参数不再被 `zip` 静默截断。新增结果 `.cache/spec_binding_scalars_grid_100.jsonl` 中仍为 47 项结构通过，所有拒绝项保留，说明此次补充没有靠减少样本维持成功数。省略的 autotuning constexpr 只由捕获记录保存，尚未证明其与图中任务 profile 对应；调用现场一致也不代替数学任务绑定或完整写回核验。全套 133 项测试通过，随后补充整数精度与正负零测试后，绑定专项 20 项通过。

来源依据链检查：`schema.decision_basis_errors` 由结构检查和运行时准入共同调用，拒绝决定自引用、相互循环、重复编号及空依据。两项注入测试在修前出现三个失败，修后通过；当前 55 份合同未发现这类问题。这只证明引用链没有这些结构缺陷，不证明来源内容支持公式。该轮审阅状态为 3 份 `reviewed_for_implementation`、52 份 `drafted`；随后新增 A1 迁入草案，当前数量见 README。

排序边界复核：`kernel_spec/sorting.py` 修复越界维度被取模、布尔 k 被当作计数、空外维度丢失排序轴长度、标量 top-k 的 k=0 崩溃。`tests/test_sorting_boundaries.py` 先复现、修后通过；本机 PyTorch 对照确认边界行为。标量 k=0 的行为目前只有本机实现证据，尚未补查官方源码（指定的本机 27891 代理连接被拒绝）。

排序完整输出检查：`registry.check_output(cid, profile, inputs, observed, ...)` 先走合同启用与参照求值，再调用合同的完整输出比较器。当前 SORT 准入范围只接受有限物理浮点且每行无并列值，逐项核对完整 values、indices、dtype 和 shape。原先的并列允许集合与联合索引枚举继续作为草案帮助函数测试，不属于当前 SORT ground truth。该入口判断的是给定任务的完整输出，不代替真实 kernel 绑定。

GRAPH 排序选择的组合：`graph.evaluate` 对小规模歧义排序枚举完整合法索引张量；同一节点的所有使用共享一次选择，分别求值整张图，返回完整输出的精确集合。共同结果可收敛为唯一输出；超出枚举预算、分支含未定义输出或无法表达精确输出集合时返回 `not_established`，不返回不完整集合。`GR.JOINT_CHOICES`、GR.E9/E10 与 `tests/test_graph_choices.py` 覆盖排列约束、重复使用、排列和不变、独立节点与预算耗尽。范围仍限于该求值器中的排序节点，不代表 lazy 路径、其他非确定性运算或真实 kernel 已完成认证。

GDR 独立复算：`tests/test_gdr_independent.py` 从论文式 (10) 的仿射矩阵乘积展开计算每个时刻的输出与状态，不调用参照的递推或分块求解来生成预期值。覆盖 K=4、V=3、非零前态、17 步跨块边界、默认缩放、状态转置，以及有独立前态的空序列段；3 项测试通过。依据为已保存的 `GDN_PAPER` 与 `FLA_GDR_NAIVE` 来源。它支持这些输入范围内的语义忠实性，不认证完整论文模块、反向梯度或任意 FLA kernel。旧 A1 仍在隔离区，不能因 GDR 通过而启用。

运行时与开发检查器现在共同核对返回值是否符合输出的表示保证：`exact` 不能返回区间，严格参照不能以普通 float 或 NaN 充当答案。ADAMW 的两项注入反例在修改前被错误接受、修改后拒绝；允许包围的 theta、精确状态及合同使用的集合表示仍可返回。变异使用不同公式时可以用严格区间提供差异见证（例如 Sparsemax 被写成 Softmax），但不能仅凭它返回区间而判错；区间与正确答案相容或只给出普通 float 均不构成拒绝见证。这只检查表示形式，不证明区间真正包住目标，也不代替公式、形状和状态核验。

本轮返回保证修复后的验证：15 项必需检查全部通过，包含 78 项单元测试；完整日志为 `.cache/spec_guarantee_final_checks.log`。此前的测试计数保留为各轮记录。

`corpus/dynamic_three_way.py` 的抽样比较仍是诊断，不是完整语义认证。本轮修复了整数经 float 转换丢失差异的问题：捕获值保留整数类型，区间中点差值用有理数计算后才转为显示统计量。局部 `domain_undefined`、`not_established`、`unsupported`、`invalid_input` 分别记录并影响总参照状态；非预期异常继续作为执行失败传播。两种 profile 使用相同坐标，结果保存坐标，且坐标选择不受此前拒绝项影响。

验证：当前单元测试 72 项通过，包括新增的 4 项比较回归。对此前 7 个已抽样记录重新运行，6 个完成比较，1 个因当前 provenance 记录与调用次数不一致被拒绝（`.cache/spec_binding_review_coordinates_20261018.jsonl`）；完成比较的 9 个输出均核对了两种 profile 的坐标一致。这些数字不是规格覆盖率；完成比较的结果仍保留 `semantic_verdict: not_established`。合同来源审阅、完整内存效果对应及严格参照接入仍未全部完成。

## 2026-10-17 代码复核补充

以下是对当前代码和已有记录的复核，后文的 10-15 / 10-16 计数保留其日期，不代表当前进度。

- 当前有 37 份合同、565 个例子；3 份为 `reviewed_for_implementation`，34 份为 `drafted`。新增合同不等于新增认证。
- 唯一检查器、注册入口、v2.5 隔离及子进程失败传播已经实现，本轮不重复实现。
- 新发现并修复：`tools/check_contracts.py:killed` 曾把变异返回 `not_established` 或 `unsupported` 当作拒绝见证。新增两项测试在修前出现三处失败；修后这两类结果与实现异常一样，不作为数学反例。
- **erf 的经验余量已移除**：后续提交已将 `kernel_spec/expr.py:iv_erf` 改为 Arb 求值。复核确认输入转为有理数，Arb 包围求值后以整数尾数与指数取得上下界，再向外转换回区间；不经过显示字符串。新增 `tests/test_reference_guarantees.py`，用带余项界的独立积分级数核对 [-2, 2] 内的点，另测正负大值、零、区间输入与精度恢复。这修复了原先“高精度点值加经验余量”的保证缺口，不等于全部 GELU 调用已经完成任务绑定。后端依据：[Arb 文档](https://python-flint.readthedocs.io/en/latest/arb.html)。
- **绑定仍未完成**：`corpus/bind_inductor.py` 记录的 2110/2120 `bound` 实际检查了源码对应、计算图片段解析和运算名支持，未检查完整输入、输出、状态与 profile。不能当作 2110 个真实语义绑定。
- 已保存的 `corpus/results/three_way_dynamic.jsonl` 共 100 项：22 项 `evaluated`、44 项输入对应不明确、25 项片段引用缺少输入、9 项输出对应不明确。22 项也只是每输出至多 34 个坐标的 K 与 f 中点差异；不是 Gπ − f 的严格裁决或完整输出认证。代码把 `Undefined` 与 `NotEstablished` 合计到 `undefined`，这两类还须分开。
- **工具入口仍需接入完整合同**：`2_tool/src/kernel_analyzer/reference_eval/audit.py` 的 `spec=` 仍接受 float64 函数。现已将其发现标为 `reference_difference`、语义判定标为“无法判断”，报告不再断言差异必然来自公式或数值舍入不能解释。差异和容差统计仍保留；非有限参照值也不再称为数学目标无定义。完整合同与严格参照接入尚未完成。
- **运行时拒绝不完整答案**：`registry.query` 现在核对合同声明的全部输出字段；`activation` 拒绝不存在的来源条款、缺失的来源文件与空的任务选择。`tests/test_runtime_contracts.py` 的五种反例均先复现失败再修复，正常完整结果保留。字段齐全仍不证明维度、布局和内存写回已经正确绑定。
- 本次验证：`run_checks.py` 的 15 项必需检查全部通过（其中检查器等测试 52 项）；审查入口 CPU 回归 1 项通过、7 项 CUDA 测试因设备不可用跳过。以上不是合同语义或真实覆盖的认证计数。

来源抽查：Katharopoulos 等论文 §3.3 式 (10)–(12) 同时包含 S、Z 和归一化分母，§3.3.1 单独推导分子的梯度；因此未归一化计算有依据，但不等于完整归一化注意力。OCP MX v1.0 §6.3 明确允许其他转换算法，因此不能仅凭不同于该节给出的转换方式判错。相关 v2.5 合同仍处于隔离区，本轮未启用。
原文：[线性注意力论文](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf)、[OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。

## 1 已有的证据

| 证据 | 结果 | 出处 |
|---|---|---|
| 参照测试 | 222 项通过：关系实例 9、舍入 oracle 4、phase1 105、phase2 89、phase3 15 | `run_checks.py` |
| 派生记录复现 | 关系检查、目录计数、随机验证、phase1 独立复算四份记录，重算后与提交的一致 | `run_checks.py` |
| phase1 独立复算 | 交叉熵用 mpmath 300 位独立求值；池化与 index/scatter 用另写的蛮力实现；0 处不一致 | `specification_v2/algorithm_specs/phase1/README.md` |
| 关系检查 | 70 项（成立 55、反例 15），26 条关系各 200 组随机输入（含 4 条阴性对照） | `specification_v2/SPECIFICATION_v2.md` §5、§8 |
| ONNX Runtime 对照 | 31,133 次比较。定义相同的运算上没有发现规格错误；发现 ORT 自身问题 3 个（其中 softcap 与掩码的次序已经 onnx#7867 修复） | `evidence/ort_witness/REPORT.md`（本次未重跑） |
| v1 错误的修正 | 8 个角色已改，各有复核值。SGD 与 BN 的 v1 行为已重建，由参数含义推出的关系能抓到 | `SPECIFICATION_v2.md` §3；`evidence/historical_misreadings/` |
| 合同内容审阅 | CE、AdamW 对照 PyTorch 2.10 文档逐条核过；22 个例子直接调用参照，58 个标量通过 | `reviews/2026-10-15/REVIEW.md` |
| Liger CE 对照 | 400 组配置上与 CE 规格一致；全部忽略时约定不同：规格为无定义，Liger 返回 0 | `evidence/declared_semantics/` |

这些证据说明参照代码与它们各自引用的文档、与独立实现大体一致。它们不说明参照已可用于裁决（见 2.1）。

## 2 已知问题

### 2.1 合同层（2026-10-16 规格 v3 第 1 阶段之后）

10-15 审阅的 F1–F5 与 10-16 审阅第 5 项已处理（计划见 `PLAN_v3.md`）：

| | 处理 |
|---|---|
| 检查器 | 新的唯一检查器 `tools/check_contracts.py`：期望值用 AST 解析，不用 eval；精确值与严格包围按 PLAN_v3 §3 比较，不设隐藏容差；每个例子、读法、变异要求都参与结论；矛盾、结构错误、实现错误退出 1，未建立退出 2 |
| 注入失败 | `tests/test_checker.py` 的注入检查包含：错误期望值、不存在的来源、子进程 exit 9、倒置区间、缺输出 / 状态 / 行、错误读法、未审阅或未绑定的自动启用、当前文件与独立复核文件不一致、只靠异常「杀死」的变异、隐藏容差。全部按预期失败 |
| 适配器 | `adapters/CE.py`、`ADAMW.py`、`RANKMEAN.py`：输入先校验（不截断计数、不丢多余项）；CE 返回 loss 与对显式上游的 VJP；ADAMW 读入前态、返回 theta / m / v / t；参照调用 v2.4 代码，误读由独立实现给出 |
| 结果 | 29 个例子全部一致（含 7 个非法输入 / 未支持例）；20 条变异要求全部在指定输出上被拒绝 |
| 来源 | 三个来源的原文快照存于 `sources/`；作者已逐条对照，未发现与条款冲突 |
| 启用 | 唯一入口 `kernel_spec/registry.py`：只读 `contracts/*.yaml`；要求合同已审阅、实现状态已核验、`validated/` 保留的完整合同与适配器和当前文件相同、来源可读取，且需任务选择的 profile 已显式绑定。开发检查不参与准入；CE.R_A 与 RANKMEAN 仍需每次绑定 |

被取代的文件：`contracts/draft/` 的草案检查器与适配器保留作历史；`tools/validate_contracts.py`、`tools/run_review_checks.py`、
`tests/test_contract_validator.py` 已由上面的检查器与测试取代（其中的结构检查已移到 `kernel_spec/schema.py`，原测试逐项移植）。

### 2.2 来源

- **凭记忆写、原文未核对的条目**：
  - CosineAnnealingLR；
  - HF warmup + cosine；
  - GradScaler 的默认值与跳步规则；
  - checkpoint 的重算约定；
  - `torch.var` 的 correction；
  - `torch.matmul` 的广播；
  - Embedding 的 max_norm 与 embedding_bag。

  清单在 `specification_v2/algorithm_specs/phase2/contract_v2_all_families.md` 末尾。
- **合同的两个 PyTorch 2.10 页面**：通过 web 读过，但没有留存原文快照与哈希（`SOURCES.json` 的 verification 字段）。
- **未进合同的 28 个角色**：来源栏多数只写「文档」，没有钉版本（参照文件开头有 URL 与抓取日期）。

### 2.3 记账

- **phase2 的哈希清单**：`specification_v2/algorithm_specs/phase2/SHA256SUMS` 没有随 v2.4 的修改更新，8 个文件校验失败。
- **顶层哈希清单**：`specification_v2/SHA256SUMS` 把自身列了进去，校验必然失败。
- 上面两份清单 `run_checks.py` 都不校验；要修，需要审阅方重新签发。
- **区间实现有三套**：`spec_cross_entropy.py` 内的 Decimal 区间、`rigorous.py`、`rigorous_mp.py`。审阅意见是先统一结果接口，不急于合并。
- **phase2 的 README**：仍是第二阶段 v2.0 的说明，文件内容已是 v2.4。

### 2.4 与 Triton 审查的衔接

- **审查入口不用合同**：`audit(spec=...)` 收一个返回 float64 的函数，用对齐度比较，没有用到合同、保证标签和三值比较。flash attention 案例的 spec 是 `fa_inputs.spec_attention`，这是用 torch 写的实现参照。
- **kernel 与合同怎么绑定，还没有方案**：
  - kernel 常融合多个运算，比如 flash attention 含 matmul、softmax、matmul；
  - kernel 的接口与合同不同，比如 Liger 的 fused linear CE 原地写梯度、不产出 logits；
  - kernel 有角色没定义的输出，比如 flash attention 的 M。
- **很多角色单个 kernel 绑定不到**：31 个角色里 7 个是训练程序级，6 个只在 torch.compile 下才成为 Triton kernel（`roles.md`）。
- **直接调用 phase1 CE 规格的地方**：`2_tool/scripts/essential/common.py` 和 `2_tool/src/kernel_analyzer/contract_v3.py`，路径已改到本目录。
- 审查接入规格的计划是 `2_tool/docs/audit_status.md` 不足 1 的第 2 项。

### 2.5 理论部分的撤回

- `THEORY.md` 中三处已撤回，反例见 `evidence/theory_checks/checks.py`：
  - 定理 4 的「五类互斥」；
  - 定理 5 第 2 条；
  - 定理 6 的「可见集合包含」。
- 不再追求的目标（`METHOD.md`）：唯一刻画、格子表、优越性、按 kernel 库列分母。
- `THEORY.md` 与 `SPECIFICATION_v2.md` 原样保留作记录，其中涉及上述内容的段落不再成立。

PAGEDATTN 现已从未核实 main-branch 草案收窄为安装的 vLLM 0.16.0 causal、未量化、二维输出路径。保留的
release 源码明确拒绝 bidirectional call，并固定分页表、bottom-right query 位置、左窗口、连续 GQA、scale
和 softcap 次序；量化、分段 softmax、sinks、ALiBi、qq_bias 与 multimodal prefix 仍拒绝。18个例子、8个错误
公式、7组768位独立复算、3组 dense PyTorch 对照和5组官方 wrapper GPU 调用通过。直接 Triton 审核先由
dense 对照抓到 source/feature 地址维误合并，修正后4组最大差不超过1.19e-7。正式24次运行的7,680个坐标
全部建立严格参照，Gπ−f 未有坐标排除零；K−Gπ 平均绝对差约1.28e-8，并在对齐规则上检出小的负向数值
作用。输出加0.25的受控版本使全部语义坐标排除零。证据见 `evidence/paged_attention_implementation.md` 与
`evidence/triton_paged_attention_binding.md`。

PACK 现已从“文档 mask、position_ids 与 attention 输出”混合草案改为 FlashAttention v2.8.3
`flash_attn_varlen_func` 的完整前向输出合同。旧文档构造任务已归档；活动合同分别定义 bottom-right causal
与 bidirectional profile，并严格核对累计长度、max_seqlen、连续 GQA、scale、f16/bf16 物理输入及全部输出。
15个例子、8个错误公式、10组768位独立复算、3组 dense PyTorch 公式和4组已安装 vLLM vendored wrapper
GPU 调用通过。直接 Triton f16 正式基线的7,680个坐标全部建立参照，Gπ−f 没有坐标排除零，完整 K−f
平均绝对差约4.02e-5；临时加0.25后全部语义坐标排除零，两类语义规则均检出。证据见
`evidence/pack_implementation.md` 与 `evidence/triton_pack_varlen_binding.md`。

F2_MX 现已从没有固定权威任务的 `ocp_floor`/`ceil_pow2` 草案，收窄为安装的 torchao 0.16.0
`to_mx` MXFP8 FLOOR 任务。三份版本固定源码分别约束接口、scale mode、格式常数和 code-0 边界；
独立位字段与整数格式复算不调用官方实现。13个合同例子、4个错误变体和14组官方函数对照通过。
直接 Triton E5M2 绑定的安全输入在6,144个元素和192个 scale code 上逐项相等；完整有限范围中，
363个坐标的正确任务值有限而 kernel 写成正负无穷，最小尺度固定例另有32个 0 与参考0.5的不一致。
工具现把前者报告为特殊值类别矛盾，不再试图对无穷残差求平均。E4M3 在 RTX A6000 上因类型支持
不足未能形成设备绑定。证据见 `evidence/mx_torchao016/` 与
`evidence/triton_mxfp8_e5m2_floor_binding.md`。当前总数为58份合同、112个 profile、40份已审阅合同、
50个准入 profile、50份调用计划和46份已审阅计划。

## 3 待决事项

| 事项 | 谁决定 | 出处 |
|---|---|---|
| 合同实施 P0–P2 由谁做（交付包写的是 Codex） | 你 | `reviews/2026-10-15/CODEX_TASK.md` |
| RANKMEAN 是否用有效 token 等权；CE 的加权 label smoothing 是否用 R_A | 任务负责人，按任务绑定 | 两份合同的 decisions |
| 范围：要审查哪些训练配置和 Triton kernel，由此决定哪些角色先迁到合同 | 你 | `reviews/2026-10-15/input_package/NEXT_STEPS.md` 第 1 步 |
| 其余 28 个角色的迁移顺序 | 你 | 同上，第 4–6 步 |
| 讨论中、未采纳的方向：算子语义层由 PyTorch 官方分解生成 f，标为实现参照；手写合同作为高价值算子的独立第二路线。可行性实验未做 | 你 | 本次讨论（见下） |

关于最后一项，torch 2.10（ka_main）中的实测：
- `torch._decomp.decomposition_table` 有 1124 条。
- 有分解的：log_softmax、nll_loss、layer_norm、rms_norm、softmax、silu、gelu、embedding、addmm、cumsum、var、SDPA。
- mm 这类本身是 core ATen 算子，语义要自己定。
- `_fused_adamw_` 没有分解。

## 4 版本沿革

| 时间 | 版本 | 内容 | 现在在哪 |
|---|---|---|---|
| 10-07 | 第一阶段 v0.4 | CE、池化、index/scatter、累积；12 条歧义经审阅决定 | `specification_v2/algorithm_specs/phase1`，与旧副本逐字相同 |
| 10-08 | 第二阶段 v2.0 | 第一档 15 个家族 | 旧副本；v2.4 的 phase2 是它的修订 |
| 至 10-15 | v2.1–v2.4 | 31 个角色、phase3、关系检查、THEORY、8 处修正 | `specification_v2/`。仓库里没有 v2.1–v2.3 的记录，v2.4 随 10-15 的规格包入库 |
| 10-15 | 合同层草案 | 三份试点合同、适配器、检查器 | `contracts/draft/` |
| 10-15 | 审阅与交付 | 方法通过；合同补全为 schema 1.1；检查器问题 F1–F5 | `contracts/`、`tools/`、`tests/`、`reviews/2026-10-15/` |

旧副本 `1_experiments/specs`（第一阶段 v0.4 加第二阶段 v2.0）已移入本目录并换成 v2.4，git 历史保留（`git log --follow`）。
- **与 v2.4 的差别**：只在 phase2 的 8 个文件，包括 SGD 的 maximize 次序、BN 计数器、注意力窗口、MoE 联合枚举、RMSprop 区间的平方、长度校验。
- **旧副本仍带着 v1 错误**：它们的修正在仓库外完成，此前没有回到仓库。
- **冻结记录里的引用**：`1_experiments/closure_round_v2`、`general_round_v3_1` 中的 `specs/phase2/... v0.1` 指那份旧副本，可用 `git show ff5e449:1_experiments/specs/phase2/<文件>` 取回。
