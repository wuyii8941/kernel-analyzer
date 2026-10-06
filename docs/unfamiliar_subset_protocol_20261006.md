# 陌生组合子集协议：Unsloth 的 Triton kernel（M2，2026-10-06，运行前写定）

本文在对该子集做任何测量之前提交（只做过导入检查）。之后的改动只追加到第 8 节。

## 1. 来源与选择

- 仓库 `github.com/unslothai/unsloth`，提交 `88cfd06`（2026-10-06 浅克隆），Apache-2.0。理由：广泛使用的微调库，
  此前没有被本项目检查过（已检查过的 Liger、vLLM、FLA、flash-attention、mamba、Triton 教程与 Inductor 不在其中），
  且 kernel 都是训练中实际调用的前向与反向。
- 总体：`unsloth/kernels/` 下各文件的公开入口。纳入规则（运行前写定，逐条判定见第 2 节）：
  1. 入口最终启动 Triton kernel（纯 torch 的入口不纳入，例如 `inplace_rope_embedding`）；
  2. 入口只接收普通张量（与范数模块的 weight / eps），不需要量化状态对象或模型对象；
  3. 能在锁定环境（Triton 3.6.0、sm_86）上运行。
  满足规则的入口全部纳入，不再抽样。
- 适配：`scripts/build_unsloth_shim.py` 逐字复制 kernel 文件与 `kernels/utils.py`（launch 设置），只为基础设施
  写桩（设备类型、bitsandbytes 可用性、unsloth_zoo 的版本 / 日志 / 补丁钩子、transformers 的 logger 与
  `LlamaRMSNorm` 基类）。不替换任何 kernel 主体或 launch 配置。

## 2. 逐项判定

| 文件 | 入口 | 判定 |
|---|---|---|
| rms_layernorm.py | `fast_rms_layernorm`（Llama 与 Gemma 两种） | 纳入（U1、U2） |
| layernorm.py | `fast_layernorm` | 纳入（U3） |
| swiglu.py | `swiglu_fg_kernel`、`swiglu_DWf_DW_dfg_kernel` | 纳入（U4、U5） |
| geglu.py | exact / approx 的前向与反向 | 纳入（U6–U9） |
| cross_entropy_loss.py | `Fast_CrossEntropyLoss`（直接调用以测 Triton 写出的逐行 loss 与 dlogits；`fast_cross_entropy_loss` 末尾的求和由 torch 完成） | 纳入：普通（U10）、分块（词表 > 65536，U11）、logit softcapping（U12） |
| rope_embedding.py | `fast_rope_embedding`：不给位置索引时走单张量 kernel（`_rope_embedding`），给索引时走 QK kernel（`_rope_embedding_QK`） | 纳入两种（U13、U14）；`inplace_rope_embedding` 为纯 torch，不纳入 |
| fp8.py、nvfp4.py | FP8 / NVFP4 的反量化与矩阵乘 | 不纳入：float8e4nv 与 NVFP4 需要 sm_89 / sm_100（规则 3） |
| nf4.py、nf4_gemv.py | NF4 反量化与 GEMV | 不纳入：需要 bitsandbytes 量化状态（规则 2） |
| int4_packed.py | INT4 反量化与 GEMV | 不纳入：需要 `Int4QuantState`（规则 2） |
| moe/grouped_gemm | 分组 GEMM 前向与反向 | 不纳入：需要路由元数据对象与 TMA（sm_90，规则 2、3） |

纳入 14 个组合。运行中出现的不支持或中止照实计入 RQ1 的分母，不换组合。

## 3. 条件与单位

| 编号 | 调用 | 形状与精度 | 测量的输出 |
|---|---|---|---|
| U1 | `fast_rms_layernorm`，gemma = False，前向 + 反向 | X (16, 4096) bf16，W bf16，eps 1e-6 | Y、dX、dW |
| U2 | 同上，gemma = True | 同上 | Y、dX、dW |
| U3 | `fast_layernorm`，前向 + 反向 | X (16, 4096) bf16，W、b bf16，eps 1e-5 | Y、dX、dW、db |
| U4 | `swiglu_fg_kernel` | e、g (1, 4, 11008) bf16 | h |
| U5 | `swiglu_DWf_DW_dfg_kernel` | DW、e、g (4, 11008) bf16 | h、df、de |
| U6 / U8 | `geglu_exact_forward_kernel` / `geglu_approx_forward_kernel` | gate、up (1, 4, 11008) bf16 | h |
| U7 / U9 | `geglu_exact_backward_kernel` / `geglu_approx_backward_kernel` | DW、e、g (4, 11008) bf16 | h、df、de |
| U10 | `Fast_CrossEntropyLoss`，前向 + 反向 | logits (4, 32000) bf16，labels 均匀 | 逐行 loss、dlogits |
| U11 | 同上（分块路径） | logits (2, 128256) bf16 | 逐行 loss、dlogits |
| U12 | 同上，logit_softcapping = 30 | logits (4, 32000) bf16 | 逐行 loss、dlogits |
| U13 | `fast_rope_embedding`，不给索引，前向 + 反向 | Q、K (1, 8, 64, 128)（batch, heads, seq, head_dim）bf16，cos / sin (64, 128) 按 θ = 10000 在 float32 计算后转 bf16 | Q′、K′、dQ、dK |
| U14 | 同上，`rope_embedding_indices` = 0..63（int32） | 同上 | Q′、K′、dQ、dK |

- 输入分布：激活与上游梯度 N(0, 1)；范数权重 1 + 0.1·N(0, 1)（Gemma 为 0.1·N(0, 1)，其作用为 1 + W）；layernorm 偏置
  0.1·N(0, 1)；logits 2·N(0, 1)；RoPE 的 Q、K N(0, 1)。全部按 bf16 舍入后作为精确输入。每个种子一次独立抽样。
- 单位：种子 0–95（开发 0–31，确认 32–95）。

## 4. 规格（由我们按公开的数学定义独立书写，不从 kernel 代码生成）

RMSNorm：y = x / √(mean(x²) + eps) · w（Gemma：· (1 + w)）；LayerNorm：(x − μ) / √(var + eps) · w + b；SwiGLU：
h = silu(e)·g；GeGLU：h = gelu(e)·g（exact 用 erf，approx 用 tanh 近似）；反向输出按各入口文档写明的含义（h、
df = DW·f(e)、de = DW·g·f′(e)）；交叉熵：loss = logsumexp(z) − z_label，softcapping 时 z ← 30·tanh(z / 30)，dlogits 为
其梯度；RoPE：q′ = q·cos + rotate_half(q)·sin，反向为其转置。全部用 torch float64 eager（含 autograd）计算，以
`f64_point_spec` 进入比较（筛查级）。

## 5. 问题与指标

- RQ1：每个输出的参照类别分布（分母含报错、不支持、中止）、区间宽度、分段耗时。
- RQ3：e_num 的规则判定（R1、R2、R3、R5）与效应量，按统计校准文档第 4 节处理偏斜。没有先验标签，属发现型数据：
  不计检出率或误报率；检出只作为候选，经独立复算与机制核实后才登记。
- e_sem 与 e_total 同时报告；e_sem 落在候选档的输出按「规格差异候选」单列，逐条核实是规格写法还是实现问题。

## 6. 运行

`scripts/unsloth_subset.py`（绑定）经 `kernel-analyzer check` 的同一引擎运行；报告在 `results/external/unsloth/`。

## 7. 不做

不改 Unsloth 代码；不为提高建立率而换组合或换形状；量化与 MoE kernel 留待下一轮。

## 8. 偏离记录

1. **测量的输出（运行前更正）。** 第 3 节为 U1–U3 列了 dW（与 db），但 Unsloth 的这两个入口在反向中对 W、b 返回
   None（微调时范数权重冻结），不计算这些量；测量的输出改为 Y、dX。
2. **冒烟运行。** 正式运行前用 6 个种子（写到 `.cache/tmp/unsloth_smoke/`，不入库、不报告其判定）确认 14 个绑定都能
   执行。发现 U13 的 dQ、dK 经 `.backward()` 时被 autograd 以 torch 操作复制到叶张量的布局，不是 kernel 写出的
   张量；绑定改用 `torch.autograd.grad` 取反向自身的输出（kernel 写出的缓冲区的视图）。形状、输入与规格不变。
