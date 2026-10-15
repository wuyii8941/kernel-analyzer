# 陌生 Triton 算子库：参照程序清点（2026-10-15）

审查 kernel 时，参照程序默认用 kernel 替换掉的 torch 运算，在 float64 下计算（`2_tool/docs/audit_status.md` 不足 1 第 2 项）。
在没测过的库上批量审查之前，先查清哪些 kernel 已有现成的参照程序、哪些还要定义。

## 范围与做法

- **库**：15 个，都没用工具测过（表 1）。
  - 排除的：Liger、FlagGems、Triton 官方教程与测试，这些用工具测过；Unsloth、torchao、xformers、bitsandbytes、mamba、vLLM
    在别的轮次里出现过，也先排除。
- **代码**：浅克隆到 `.cache/untrusted/<库>_src/repo`，只读，没有导入、安装或运行。提交号与许可在 `inventory/*.json`。
- **清点**：
  - 5 个子任务并行读代码，每个库写一份 `inventory/<库>.json`；
  - 合并后先校验格式，再按规格类别归并（`build_tables.py`：关键词规则，加 65 处读过分族后的人工修正）；
  - 分类都来自读代码，没有运行验证。
- **单位**：入口 kernel 指 Python 端带 grid 启动的 `@triton.jit` 函数，按实现的运算归成分族。每个分族归入四类之一：
  - torch 运算：一个公开的 torch 运算或模块就是参照程序；
  - torch 组合：几个 torch 运算的组合，约定全由 kernel 的接口确定；
  - 需要定义：有 torch 运算定不了的选择，例如布局、量化格式、路由的并列次序；
  - 随机：依赖随机数。

  反向 kernel 的参照程序是前向参照程序的梯度。

## 结果

共 1852 个入口 kernel，797 个分族：

| 类别 | 分族 | 占比 |
|---|---|---|
| torch 运算 | 103 | 13% |
| torch 组合 | 350 | 44% |
| 需要定义 | 329 | 41% |
| 随机 | 15 | 2% |

表 1　各库

| 库 | 入口 kernel | 分族 | torch 运算 | torch 组合 | 需要定义 | 随机 | sm_86 不能跑的分族 | 备注 |
|---|---|---|---|---|---|---|---|---|
| attorch | 25 | 27 | 17 | 6 | 0 | 4 | 0 | torch.nn 子集；目标 Triton 3.0 |
| trident | 52 | 48 | 37 | 8 | 0 | 3 | 0 | 写于 Triton 约 2.1，要移植 |
| DeepSpeed | 34 | 19 | 4 | 14 | 1 | 0 | 0 | 推理用 Triton kernel |
| flash-attention（Triton 部分） | 13 | 7 | 2 | 4 | 0 | 1 | 0 | 部分 kernel 用旧 API，要移植 |
| applied-ai | 36 | 21 | 8 | 10 | 3 | 0 | 6 | |
| kernl | 22 | 21 | 7 | 8 | 6 | 0 | 2 | 已归档；需要 Triton 2.0 |
| Lightning Attention | 16 | 4 | 1 | 1 | 2 | 0 | 0 | |
| flash-linear-attention | 466 | 147 | 2 | 58 | 87 | 0 | 21 | 21 个不能跑的都是昇腾后端 |
| SGLang | 805 | 309 | 16 | 167 | 121 | 5 | 42 | |
| lightllm | 216 | 89 | 8 | 47 | 34 | 0 | 17 | |
| LMDeploy | 89 | 58 | 1 | 20 | 36 | 1 | 16 | |
| OpenAI triton_kernels | 31 | 25 | 0 | 3 | 21 | 1 | 1 | |
| gemlite | 34 | 11 | 0 | 2 | 9 | 0 | 3 | |
| SageAttention（Triton 部分） | 9 | 7 | 0 | 1 | 6 | 0 | 0 | |
| stk | 4 | 4 | 0 | 1 | 3 | 0 | 0 | |

- 按 torch.nn 重写的库（attorch、trident、DeepSpeed、flash-attention 的 Triton 部分）几乎全部已有参照程序。
- 需要定义的集中在三类库：
  - 推理引擎：SGLang、lightllm、LMDeploy；
  - 线性注意力：flash-linear-attention；
  - 量化：gemlite、SageAttention、triton_kernels。

## 需要定义的规格

329 个「需要定义」的分族归成 13 类；同一类在不同库里要定的事大体相同，定一次可以覆盖多个库。逐个分族的待定事项见
`families.md`，各类的分族清单见 `classes.json`。

| | 规格类别 | 分族 | 入口 kernel | 涉及的库 | sm_86 能跑的分族 | 要定的事 | 可用的依据 |
|---|---|---|---|---|---|---|---|
| A | 线性注意力与状态递推 | 103 | 359 | flash-linear-attention、SGLang、lightllm、Lightning Attention | 91 | 每个变体的递推式与参数化（门控用 ln 还是 log2、decay 形式、beta 是否过 sigmoid、由 A_log 与 dt_bias 构造门控的方式）；scale 默认值；初始与最终状态的布局，是否参与梯度；分块实现与逐步递推要相等到什么程度；变长序列与状态池的索引、填充槽；kernel 内部 q/k 的 L2 归一化 eps；value 头到 q/k 头的映射；卷积状态的移位方向 | 各变体的论文（GLA、DeltaNet、Gated DeltaNet、RWKV-6/7、Mamba-2、Lightning Attention-2 等）；fla 各目录的 `naive.py`（fp32，多数不支持变长）；对应模型的 HF 实现 |
| C | 稀疏注意力、索引器与 KV 压缩 | 53 | 146 | SGLang、LMDeploy、lightllm、flash-linear-attention | 35 | 块打分（求和、平均、最大或 logsumexp）；top-k 的并列次序与必选块；索引 −1 的含义；ks / ke 的区间约定；压缩窗口与环形缓冲的布局；fp8 / fp4 KV 的字节布局（如每 token 448 字节 fp8 加 64 维 bf16 rope）；UE8M0 scale 的舍入 | NSA 论文；DeepSeek V3.2 / V4 技术报告与官方推理代码、FlashMLA；MiniMax 公开实现 |
| H | MoE：路由、排序对齐、分组 GEMM、EP 分发 | 39 | 61 | SGLang、LMDeploy、lightllm、triton_kernels、applied-ai | 32 | 路由打分函数（softmax、sigmoid、sqrtsoftplus）与 bias 加在哪里；分组 top-k；并列次序；是否重新归一化；对齐填充（moe_align 的填充 id、+1 桶）；同一专家内 token 的次序（多由原子操作决定，参照只能比较集合）；EP 分发与丢弃；量化专家权重的格式（同 G 类） | DeepSeek-V3、Mixtral、Qwen-MoE、GPT-OSS 的 HF 实现；各引擎自带的 torch 参考 |
| F | 量化格式与量化 / 反量化 | 41 | 84 | SGLang、lightllm、LMDeploy、gemlite、triton_kernels、SageAttention、applied-ai、kernl | 15 | scale 粒度（每张量、每 token、每组 128、每块 128×128、MX 每 32、NVFP4 每 16）；amax 下限 eps（各库从 1e-4 到 1e-12 不等）；scale 的舍入（OCP 规定向下取 2 的幂，多个库向上取）；元素舍入（就近偶数、远离零或截断）；饱和与 NaN；量化前是否先舍入到 bf16 | OCP MX v1.0 规范；NVIDIA FP8 / NVFP4 文档；各库的 torch 对照 |
| B | 分页 / 变长 KV 注意力 | 23 | 67 | SGLang、lightllm、LMDeploy、SageAttention、kernl、flash-linear-attention | 21 | KV 的布局（block table、req_to_token、kv_indptr 加 kv_indices、页大小）；有缓存前缀时的因果对齐；序列长度是否含当前 token；滑动窗口是否含端点；softcap、sink、ALiBi 的形式；MLA 的吸收形式；量化注意力的参照目标（反量化后的精确注意力，还是原输入上的 SDPA） | 布局定下后，多数可以写成「按索引取出 KV，再做 SDPA」的 torch 组合；各引擎的 torch 参考与测试；HF 模型实现 |
| J | 布局、元数据与调度 | 16 | 25 | triton_kernels、SGLang、lightllm | 16 | swizzle 置换；ragged 元数据格式；分片与调度计划；页对齐。多数是整数映射，格式定下后可以精确比较；调度启发式（如 split 个数）不是语义，可以不审 | 各库的格式说明与 torch 参考 |
| E | KV cache 写入、读出与 KV 量化 | 13 | 18 | LMDeploy、lightllm、SGLang | 5 | 写入位置的映射；int8 / int4 的 scale 与零点公式、舍入（截断还是就近）；int4 的打包次序；fp8 scale 的粒度 | 各引擎的 torch 参考与测试 |
| G | 量化与低比特 GEMM | 13 | 32 | gemlite、LMDeploy、SGLang、applied-ai、DeepSpeed | 8 | 权重打包次序（AWQ 是 [0,4,1,5,2,6,3,7]，GPTQ 从低位起）；零点形式（z 或 z+1）；group 大小；按反量化后的值精确相乘，还是中间先舍入；累加类型 | GPTQ、AWQ 原实现；compressed-tensors 格式说明；各库的 torch 参考 |
| I | 采样、投机解码、哈希与水印 | 12 | 32 | SGLang、lightllm | 12 | 给定均匀随机数时的逆 CDF 规则、温度下限、惩罚项的次序；投机解码的接受规则、树的构建与验证；哈希与水印完全由代码定义（MurmurHash 等），参照只能照抄算法 | 投机采样论文（如 arXiv 2403.10444）；各库的测试 |
| M | RoPE 与位置编码变体 | 6 | 6 | kernl、lightllm、SGLang | 6 | 两两配对的方式（相邻交错还是前后两半）；theta；位置偏移；多模态 mrope 中图像 token 的位置 | HF 实现（如 Qwen2-VL 的 mrope）；kernl 自带的 `pytorch_reference.py` |
| O | 其他 | 5 | 9 | SGLang、lightllm、triton_kernels | 3 | mHC 中 Sinkhorn 迭代的次序与 eps；GPT-OSS 的 SwiGLU 变体（列交错、截断、+1）；Triton 的 FPSan 编码 | 对应论文与官方实现 |
| K | 块稀疏矩阵乘 | 3 | 3 | stk | 3 | 块 CSR 元数据格式，包括转置元数据；重复块；块大小固定为 128 | stk 自带的 `to_dense` 后接 `torch.mm` |
| L | LoRA | 2 | 14 | SGLang | 2 | 分段批布局与 adapter 排序；缩放；吸收形式 MLA 中的 LoRA 修正 | LoRA 论文；SGLang 的测试 |

## 随机类（15 个分族）

| 内容 | 分族 | 能确定性检查的部分 |
|---|---|---|
| dropout（单独的，或融合在激活、归一化、注意力里） | attorch 4、trident 3、flash-attention 1、SGLang 1 | 以 kernel 的保留掩码为条件，比较保留下来的元素是否等于 x / (1 − p)。attorch 自己的测试就是这样做的。掩码是否服从概率 p，只能做统计检验 |
| 采样 | LMDeploy 1、SGLang 2 | 固定均匀随机数后按逆 CDF 比较；随机数流本身可按 Philox 复现 |
| 其他 | SGLang 2（模拟均衡路由、测试用的随机等待）、triton_kernels 1（Philox 随机块符号） | 同上 |

需要你定：随机类是否接受「以 kernel 自己的随机数为条件」的比较。

## 清点中看到的疑似问题

下列都只是读代码时看到的，没有运行，大部分没有核实；标「已复核」的由子任务对照源码复核过一次。它们是后续审查的线索，
不是结论。

- applied-ai：softmax 反向把概率和梯度传反了位置（已复核）；三个 GPTQ kernel 有除以零；flash 注意力反向只能编译 bf16。
- SGLang：`compute_identity_kernel` 在 hidden 不是 256 的倍数时漏掉尾部列（已复核）；两个 softcap kernel 用 exp 算 tanh，输入大时出 NaN。
- kernl：主注意力与 split-KV 注意力用错了屏蔽填充 key 的偏移。无 mask、非因果、key 长度大于 128 且不是 128 的倍数时，输出会被缩小。
- attorch：交叉熵与 NLL 在类别数小于 64、batch ≥ 2048 且不能被块大小整除时漏掉最后一块并越界写；L1 损失在差为 0 处的梯度是 +1（torch 是 0）。
- trident：dropout 反向由「输出 > 0」推掩码并乘 (1 − p)，应为除以；RMSNorm 把 eps 加在开方外；GELU 用 tanh 近似，测试却和精确 GELU 比。
- flash-attention（Triton 部分）：`kernel_fwd` 写输出不带 mask，M 或 N 不是块大小倍数时越界写。
- Lightning Attention：并行变体要求 n 是 256 的倍数，否则尾部行不写。
- LMDeploy：量化分页 decode 忽略 `seq_len`，多于一个 query token 时结果错行；`per_token_quant_int8` 远离零舍入，库自己的参考是就近偶数。
- flash-linear-attention：HGRN 与 ABC 的分块反向丢掉最终状态的梯度；RWKV-7 的通道混合把最后一个 batch 行当成状态返回，而不是最后一个 token。
- gemlite：MX GEMM 在 K 不是块大小倍数时越界读。
- triton_kernels：MX scale 默认向上取 2 的幂，不是 OCP 规定的向下取。

## 文件

| 文件 | 内容 |
|---|---|
| `inventory/<库>.json` | 各库的清点：提交号、许可、计数，每个分族的类别、参照程序或待定事项、依据、能否在 sm_86 上跑、说明 |
| `families.md` | 由脚本生成：各库逐个分族的明细，含规格类别列 |
| `classes.md`、`classes.json` | 由脚本生成：按规格类别的汇总与分族清单 |
| `build_tables.py` | 校验清点文件，归类，生成上面三份文件 |

复现：`/data1/tzh/envs/ka_main/bin/python build_tables.py`（只读 `inventory/`，不需要克隆的代码）。

## 下一步

- 先在参照程序已定义的 453 个分族上批量审查：逐位核对、精度偏差、与参照程序比较。优先能在 Triton 3.6 / sm_86 上直接跑的库。
- 13 类需要定义的规格与随机类的处理，等你决定。每定下一类，对应的分族就转为可以审查。
