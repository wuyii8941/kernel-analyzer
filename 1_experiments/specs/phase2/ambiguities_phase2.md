# 第二阶段规格的歧义与声明约定清单（v0.2，2026-10-08）

与第一阶段同一规则：有依据的读法全部实现，候选在全部输入上须符合同一读法；没有依据的变体只作错误变体；文档明文的不是歧义。标 C 的是「声明约定」：不同库各自声明不同公式，比较时只对照候选自己声明的那种，混用才是错误。

| 编号 | 情形 | 读法 / 处理 |
| --- | --- | --- |
| BASE-A1 | var/std 自由度 ≤ 0；amax/amin/mean 的空归约 | 无定义，规格返回未建立；sum=0、prod=1 为单位元（文档） |
| BASE-A2 | softmax 全 −inf 行 | 无定义；logsumexp 为 −inf |
| BASE-A3 | GELU 的 approximate | erf 与 tanh 两种声明公式，各自对照；0.044715 与 √(2/π) 是文档公式的一部分 |
| BASE-A4 | matmul 的广播 | 本规格不广播批维，只支持 2-D 与同批 3-D |
| ATT-A1 | 整行被 mask | 无定义；记录 NaN/零的约定 |
| ATT-C1 | q_len ≠ k_len 的 causal 对齐 | SDPA 文档：左上；FlashAttention 文档：右下；按候选声明 |
| ATT-C2 | GQA 的头映射 | 查询头 h 对应 kv 头 h // (H_q/H_kv) |
| ATT-A2 | 滑动窗口边界 | 规格取 i − j < window，记录候选声明 |
| ROPE-C1 | 布局与位置起点 | rotate-half 与交错；位置偏移按声明 |
| NORM-A1 / A2 / A3 | 见 spec_normalization | — |
| EMB-A1 / A2 / A3 | 见 spec_embedding | — |
| OPT-A1 | amsgrad 取 max 的对象 | 近期文档：对未修正的 v 取 max 再修正（主）；旧版文档：对修正后的 v̂ 取 max（R_old）；按锁定版本 |
| OPT-A2 / A3 | maximize 与权重衰减的次序；SGD 第一步不加 dampening | 按算法框 |
| SCH-A1 / A2 | cosine 闭式与递推；计步起点 | 闭式为规格；递推只在外部改 lr 时不同 |
| CLIP-A1 | 1e-6 | 文档公式的一部分 |
| AMP-A1 | inf 检查的范围 | 该次 unscale_ 涉及的全部参数 |
| PROG-A1 | 跨 rank 平均 | 任务要求全局 token 平均；均值的均值为错误变体 |
| MOE-C1 … C4 | 重归一、并列、容量、aux loss | 按声明；并列返回集合 |

严格性标注：spec_base_ops 的 GELU-erf 使用 erf 的高精度点值加 10⁻⁶⁰ 的声明余量，不是严格区间；其余 exp、log、sqrt、cos、sin、tanh 为严格区间；加减乘除、max、min、计数为精确有理数。
