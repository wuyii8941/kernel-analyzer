# 角色清单（v2.4，2026-10-15）

一个角色是一类要写 f 的运算或训练步骤。前五列取自 `specification_v2/catalogue.py` 与 `SPECIFICATION_v2.md` §4，后两列是本次整理加的。

- 参照代码的路径相对 `specification_v2/algorithm_specs/`。
- 「依据」的五类：
  - 文档：PyTorch 等官方文档，含优化器的算法框；
  - 论文；
  - 数学：数学定义或恒等式；
  - 标准；
  - 任务：任务定义，没有外部权威，属于任务负责人的选择。
- 「Triton 审查」是整理时的判断，未逐个核实：
  - kernel 级：常有 Triton kernel 直接实现；
  - 编译后：通常只在 torch.compile / Inductor 下生成 Triton kernel；
  - 程序级：训练程序的一步，不是单个 kernel。审查单个 Triton 启动时绑定不到。
- 「待核」指 `phase2/contract_v2_all_families.md` 末尾列出的、第一版按记忆书写、原文未核对的条目。

| 角色 | 内容 | 参照代码 | 依据 | 参照类型 | 合同 | Triton 审查 | 备注 |
|---|---|---|---|---|---|---|---|
| CE | 交叉熵损失与梯度 | phase1/spec_cross_entropy.py | 文档 | 严格区间 | CE.yaml，已审阅 | kernel 级 | |
| AVGPOOL | 平均池化 | phase1/spec_pooling.py | 文档 | 精确 | — | kernel 级 | 越界窗口的除数有两种读法 |
| MAXPOOL | 最大池化 | phase1/spec_pooling.py | 文档 | 允许集合（并列） | — | kernel 级 | |
| INDEXREDUCE | index_add、index_reduce、scatter_reduce | phase1/spec_index_scatter.py | 文档 | 精确；并列为集合 | — | kernel 级 | B020 由它发现 |
| ACCUM | 梯度累积窗口的归一化 | phase1/spec_accumulation.py | 任务 | 精确 | — | 程序级 | |
| ATTN | SDPA 语义：mask、因果对齐、softcap、滑动窗口 | phase2/spec_attention.py | 文档；FlashAttention 的窗口约定 | 严格区间 | — | kernel 级 | v1 的窗口错误已改 |
| GQA | 查询头到 KV 头的映射 | phase2/spec_attention.py | 论文 | 精确 | — | kernel 级 | |
| PACK | FlashAttention 变长序列前向 | contracts/PACK.yaml | 文档（官方接口） | 严格包围 | PACK.yaml，已审阅 | kernel 级 | 累计长度界定独立序列；mask 与 position_ids 另属预处理任务 |
| ROPE | 旋转位置编码，两种布局 | phase2/spec_attention.py | 论文 | 严格区间 | — | kernel 级 | |
| MATMUL | matmul、linear 及其梯度 | phase2/spec_base_ops.py | 数学 | 精确 | — | kernel 级 | 待核：广播规则 |
| REDUCE | sum、mean、prod、amax、amin、var、std、cumsum | phase2/spec_base_ops.py | 数学；correction 参数 | 精确 | — | kernel 级 | 待核：var 的 correction |
| SOFTMAX | softmax、log_softmax、logsumexp | phase2/spec_base_ops.py | 数学 | 严格区间 | — | kernel 级 | |
| ACT | ReLU、SiLU、GELU 两式、SwiGLU、GeGLU | phase2/spec_base_ops.py | 文档 | 区间；GELU-erf 为声明近似；ReLU 零点两种 profile | — | kernel 级 | v1 的 ReLU 零点已改 |
| GATHER | gather、index_select 及伴随 | phase2/spec_base_ops.py | 文档 | 精确 | — | kernel 级 | |
| EMB | embedding、embedding_bag | phase2/spec_embedding.py | 文档 | 精确；renorm 为区间 | — | kernel 级 | 待核：padding_idx、max_norm |
| MOE | 路由、top-k、重归一、容量丢弃、合并、负载均衡损失 | phase2/spec_moe.py | 论文（Switch、Mixtral） | 允许集合（联合枚举，有预算） | — | kernel 级 | v2.2 的枚举已改 |
| NORM | LayerNorm、RMSNorm、GroupNorm | phase2/spec_normalization.py | 文档 | 严格区间 | — | kernel 级 | |
| BN | BatchNorm 输出与 running 统计 | phase2/spec_normalization.py | 文档 | 严格区间 | — | 编译后 | v1 的计数器已改 |
| ADAM | Adam、AdamW 一步与短序列 | phase2/spec_optimizers.py | 文档（算法框） | 严格区间 | ADAMW.yaml，已审阅（只含 AdamW profile；coupled Adam、amsgrad 未进合同） | 编译后 | |
| SGD | SGD | phase2/spec_optimizers.py | 文档（算法框） | 精确 | — | 编译后 | v1 的 maximize 次序已改 |
| RMSPROP | RMSprop | phase2/spec_optimizers.py | 文档（算法框） | 严格区间 | — | 编译后 | v2.2 已改 |
| SCHED | cosine、linear、HF warmup + cosine | phase2/spec_training_program.py | 文档 | 严格区间 | — | 程序级 | 待核：两种调度的公式原文 |
| CLIP | clip_grad_norm_ | phase2/spec_training_program.py | 文档 | 严格区间 | — | 编译后 | |
| SCALER | GradScaler 状态机 | phase2/spec_training_program.py | 文档 | 精确 | — | 程序级 | 待核：默认值与跳步规则 |
| LABELS | 标签右移与计数 | phase2/spec_training_program.py | 任务 | 精确 | — | 程序级 | |
| RANKMEAN | 跨 rank 的 token 平均 | phase2/spec_training_program.py | 任务（HF 文章只是支持例子） | 精确 | RANKMEAN.yaml，已审阅；token 等权需任务绑定 | 程序级 | |
| CKPT | 重算等于保存 | phase2/spec_training_program.py | 数学 | 性质比较器，不是独立参照 | — | 程序级 | 待核：重算约定；v2.2 已改 |
| CAST | 格式转换与舍入（IEEE、FP8） | ../oracle_rounding.py | 标准（IEEE 754-2019、OCP / NVIDIA FP8） | 精确 | — | kernel 级 | v1 已改 |
| EMA | EMA、SWA 参数平均 | phase3/spec_ema.py | 文档 | 精确 | — | 编译后 | |
| KVCACHE | 连续与分页 KV cache：decode 等于 prefill | phase3/spec_kvcache.py | 任务 | 严格区间 | — | kernel 级（推理） | |
| KVCOMP | 分块 KV 压缩 | phase3/spec_kvcomp.py | 论文（arXiv 2609.36322 §3.1） | 精确 / 严格区间 | — | kernel 级 | |

另有两条路径规格，检查同一语义的两条实现路径是否相等，不是独立角色：
- SPARSE（phase3/spec_sparse.py）：COO 稀疏与稠密相等；
- STATEIO（phase3/spec_state_io.py）：保存、加载后继续，与直接继续相等。

## 计数

- **依据**（共 31）：
  - 文档 18；
  - 论文 4（GQA、ROPE、MOE、KVCOMP）；
  - 数学 4（MATMUL、REDUCE、SOFTMAX、CKPT）；
  - 标准 1（CAST）；
  - 任务 4（ACCUM、LABELS、RANKMEAN、KVCACHE）。
- **合同**：此表所列角色中 4 份已审阅，27 个角色没有本表记录的合同。
- **Triton 审查**：
  - kernel 级 19；
  - 编译后 6（BN、ADAM、SGD、RMSPROP、CLIP、EMA）；
  - 程序级 6（ACCUM、SCHED、SCALER、LABELS、RANKMEAN、CKPT）。

两点观察：
- 「任务」类的 4 个角色没有外部权威，它们的条款本质上是任务负责人的选择。按方法应写成需任务绑定的决定；目前只有 RANKMEAN 的合同这样写了。
- 三份试点合同里，只有 CE 属于 kernel 级。
