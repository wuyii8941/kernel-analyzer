# E4 真实缺陷来源（S1 已知缺陷回归、S3 未参与开发的真实缺陷）

2026-10-19 从仓库已有记录整理，未新跑。B = `4_bugs_cases/README.md`（B001 在第 33 行，B024 在第 56 行），
E = `5_specification/corpus/external_batch/README.md`。机制分类按 `docs/SUFFICIENCY_PROTOCOL.md` 第 6 节；
数值缺陷（K−Gπ）不属于 M1–M7，单独标注。

## S1-a：4_bugs_cases

| ID | 实现 | 机制 | 现有完整 f | 检出方式 | 证据 |
|---|---|---|---|---|---|
| B001 RoPE 当成部分旋转 | Liger main | M1 | ROPE tables_half（变异 RO.M4） | Gπ−f | B:33 |
| B002 Phi-4-mini 整头旋转 | Liger ≤0.8.4 | M1 | ROPE tables_duplicated（RO.M3） | Gπ−f | B:34 |
| B003 融合 MoE tl.dot TF32 | Liger 0.8.x | 数值 | MOE | 同机制由 K−Gπ 在 B006 上检出；本 kernel 未跑 | B:35 |
| B004 RMSNorm offset 0 | Liger 0.7.0 | M1 | NORM rms_offset | 未跑（已修复） | B:36 |
| B005 纯 bf16 AdamW 丢增量 | torch AdamW | 数值（状态） | ADAMW | K−Gπ | B:37 |
| B006 tl.dot fp32 默认 TF32 | Triton | 数值 | 未记录 | K−Gπ | B:38 |
| B007 bf16 张量核 FP32 累加缩小 | Triton | 数值 | 未记录 | 仅探针 | B:39 |
| B008 AdamW4bit 矩缩小步长 | torchao 0.16 | 数值（量化设计） | 无（未绑定） | 训练对比 | B:40 |
| B009 随机舍入未用于状态 | torchao 0.16 | 数值（状态） | SROUND（草案） | 探针 | B:41 |
| B010 avg_pool2d ceil 反向除数 | Inductor 2.10 | M2 | POOL | Gπ−f | B:42 |
| B011 教程 attention 非因果反向 | Triton 教程 06 | M3 | ATTN | Gπ−f | B:43 |
| B012 RAdam ρ_t fp32 分支翻转 | PyTorch 2.10 | 其他（低精度改变分支） | RADAM | Gπ−f | B:44 |
| B013 vLLM 双向 SWA 的 V 掩码 | vLLM | M3 | PAGEDATTN | Gπ−f | B:45 |
| B014 池化单元素参数反向 | Inductor 2.10 | 其他（编译失败） | POOL | 异常 | B:46 |
| B015 avg_pool3d ceil 反向 | Inductor 2.10 | M2 | POOL | Gπ−f | B:47 |
| B016 scatter 到单元素目标 | Inductor 2.10 | M3 + 其他 | INDEXREDUCE、INDEX | Gπ−f | B:48 |
| B017 Dynamo 共享 `__code__` 守卫 | Dynamo 2.10 | 其他（图复用） | POOL | Gπ−f 报警，读源码定因 | B:49 |
| B018 SDPA 重写 | Inductor 2.10 | M1 + M3 | ATTN | 人工 + 直接差分 | B:50 |
| B019 bf16 EMA 停滞 | AveragedModel | 数值（状态） | EMA（草案） | 单步作用 | B:51 |
| B020 amax/amin include_self=False 反向 | 2.10 | M4 | INDEXREDUCE | 完整 K−f | B:52 |
| B021 max_pool 空窗口越界索引 | 2.10 eager | M3 + 越界写 | POOL | K−f + 性质 | B:53 |
| B022 max_pool 反向 −1 索引断言 | nightly 2.15 | 其他（崩溃） | POOL | 异常 | B:54 |
| B023 gelu-tanh 反向大 x 出 NaN | 2.10 | 其他（中间溢出） | ACT | K−Gπ 特殊值类别 | B:55 |
| B024 Triton 浮点 `%` | Triton 3.6 | 其他（原语下降） | P 类（任务即程序） | K−K_R | B:56 |

## S1-b：开发批次（attorch、FlagGems 5.0.2）

| ID | 实现 | 机制 | 检出方式 | 是否因此改规格 | 证据 |
|---|---|---|---|---|---|
| D1 L1 在 x=target 的梯度 | attorch | M4 | 完整 K−f | 否（E:30），事先已怀疑 | E:21-30 |
| D2 SmoothL1 beta=0 梯度 | attorch | M4 | 完整 K−f（读源码后复现） | HUBER 为补参照缺口而加，非为缺陷 | E:68-72 |
| D3 addmm beta=0 传播 NaN/Inf | FlagGems | M1 | 完整 K−f + 独立复现 | 否 | E:243-248 |
| D4 标量 bias IndexError | FlagGems | 其他（异常） | 异常 | 否 | E:250 |
| D5 rms_norm eps=None 取 1e-5 | FlagGems | M1 | K−f + 捕获 | 否 | E:283-287 |
| D6 RMSNorm dw 被压平 | FlagGems | M6 | 异常 | 否 | E:288 |
| D7 weight=None 调用 `.contiguous()` | FlagGems | 其他（异常） | 异常 | 否 | E:290 |
| D8 avg_pool ceil 全核除数（前后向） | FlagGems | M2 | 与 PyTorch 对照，后数学复核 | avg profile 后补 | E:331-371 |
| D9 rank-3 池化输入失败 | FlagGems | 其他（异常） | 异常 | 否 | E:349 |
| D10 conv `same` 步长检查、无 batch 被拒 | FlagGems | 其他（异常） | 异常 | 否 | E:451 |
| D11 channels-last 分组卷积反向 | FlagGems | 其他（编译失败） | 编译失败 | 否 | E:454 |
| D12 depthwise 反向编译超时 | FlagGems | 其他（超时；记录未称缺陷） | 超时 | 否 | E:459 |
| D13 SDPA 布尔掩码反向 | FlagGems | M3 | 完整 K−f + 读源码 | VJP profile 为任务而加 | E:507-532 |
| D14 加性掩码未换底 | FlagGems | M1 | 同上 | 同上 | E:532 |
| D15 反向丢掩码 | FlagGems | M3 | 同上 | 同上 | E:533 |
| D16 反向总走因果路径 | FlagGems | M3 | 同上 | 同上 | E:533 |
| D17 GQA dq 偏大 | FlagGems | 未记录 | K−f | 同上 | E:530 |
| D18 nearest 插值忽略 stride | FlagGems | M6 | 完整 K−f | profile 为调用缺口而加 | E:547-565 |
| D19 argmax TILE_N 非 2 的幂编译失败 | FlagGems | 其他（编译失败） | 编译失败 | 否 | E:593 |
| D20 argmax dim=None 非连续输入 | FlagGems | M6 | 完整 K−f | 否 | E:594-597 |
| D21 masked_fill 拒合法广播；非连续输出 | FlagGems | 其他 + M6 | 异常 / 布局检查 | profile 为缺口而加 | E:608-614 |
| D22 pad 无 autograd 路径 | FlagGems | 其他（缺反向路径） | 异常 | profile 为缺口而加 | E:622-637 |

## S3 候选：验收批次首次发现

| ID | 实现 | 机制 | 检出方式 | 是否改规格 | 是否满足 S3 |
|---|---|---|---|---|---|
| A1 clamp 复用此前编译的 bound | TorchInductor 2.10 | 其他（跨调用参数复用） | 完整 K−f + 无工具最小复现 | 否（E:723「没有改规格迁就候选」） | **是** |
| A2 addmm beta=0 NaN | Inductor Triton GEMM | M1 | 完整 K−f | 否 | 否：beta=0 反例开发前已知（E:231） |
| A3 avg_pool2d ceil 尾部反向除数 | Inductor | M2 | 与 PyTorch 对照 + 探针 | 是（之后补参照，E:366） | 否：与 B010 同机制且之后改了规格 |
| A4 kernel_size=[3] 反向编译失败 | Inductor | 其他（编译失败） | 编译失败 | 否 | 否：与 B014 相同 |
| A5 MSE mean 系数 Gπ−f=(45/2^32)·f | FlagGems MSE | 数值（舍入后的 1/287 常数） | 仅 Gπ−f | 否 | 部分：数值系数，非 M1–M7 |
| A6 教程 f32 编译 / 资源拒绝 | Triton 教程 3.6 | 其他（编译 / 资源） | 编译失败 | 否 | 否：记录未称缺陷 |

结论：严格满足 S3 两个条件的只有 A1，且它不属于 M1–M7 中任何一类。S3 样本不足以给出任何机制的检出率，按原样报告。

## 已记录的正常调用（对照）

验收：FlagGems 前向 SmoothL1/Huber 48/48；FlagGems MSE 前向 12/12；index-select+gather 12/12；Inductor：
embedding 12/12、embedding state 12/12、矩阵（显式 Triton）12/16、归一化 16/16、池化 16/20、平均池化 6/8、
唯一最大池化 8/8、卷积模板 10/20、插值 8/8、argmax 10/10、masked_fill 10/10、pad 10/10、where 10/10、
cat 10/10、clamp 8/10、extrema 10/10；trident 0/4（导入失败，环境问题）。未通过的调用对应上表缺陷或编译失败，
没有正常调用被参照误判为语义错误的记录。
