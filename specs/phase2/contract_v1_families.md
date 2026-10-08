# 输入契约 v1（2026-10-08）— 合法、未规定、无定义输入，按家族

版本：对应规格包 specs_phase1 v0.4 与 specs_phase2 v0.1；锁定版本 PyTorch 2.10 文档。执行方不得在看到结果后增删条目；修改走审阅入库。
五类的处理见保证与裁决表第 3 节：合法有定义 → 全部比较；合法未规定 → 全部有依据的读法、候选须一致选一种；合法但无定义 → 记录约定不判对错；不合法 → 规格拒绝；规格未建立 → 单列。

| 家族 | 合法域（之外 → 拒绝） | 文档未规定（读法） | 实数目标无定义（记录约定） |
| --- | --- | --- | --- |
| 交叉熵 | 类别下标 ∈ [0, C) 或 = ignore_index；概率目标与 logits 同形；weights 长 C；label_smoothing ∈ [0,1]；reduction ∈ {none, mean, sum} | CE-A1（权重与平滑：R_A 主、R_B 对照）；CE-A2（分母：D2 主、R_C 诊断）；CE-A6（标量口径两种都报） | CE-A3：mean 且 D2 分母为零（含分子为正的情形）；none/sum 仍有定义 |
| 池化 | kernel/stride/dilation ≥ 1；padding ≥ 0 且 2p ≤ d(k−1)+1；有效核不大于带 padding 的输入；divisor_override 非零整数；avg 的输入有限 | POOL-A1（除数：R2 主、R1 对照）；POOL-A2（并列：集合式）；POOL-A3（NaN：R_prop 主、R_ignore 单列）；MaxPool1d ceil 的两版文档公式按锁定版本 | POOL-A5 几何空窗口 → −∞、空集合；R_ignore 过滤后无值 → 规格未建立 |
| index/scatter | 索引 ∈ [0, size(dim))；dim ∈ {0,1}；1-D/2-D；源与索引形状匹配；sum/prod 输入有限 | IDX-A3（NaN：R_prop 主）；IDX-A4（并列反向：集合式） | IDX-A1/A2 只有一种 mean 读法；全 NaN 过滤后 → 未建立；整数 mean 不在范围 |
| 累加窗口 | 各 micro-batch 的 token 数 ≥ 0 | — | ACC-A1 窗口总 token 数为 0 |
| 优化器（AdamW/Adam/SGD/RMSprop） | 超参为数；lr ≥ 0；betas/alpha ∈ [0,1)；nesterov 需 momentum > 0 且 dampening = 0；梯度有限（非有限梯度由 AMP 契约处理） | OPT-A1（amsgrad 的 max 对象：按锁定版本文档，旧版为 R_old）；OPT-A2（maximize 与 weight decay 的次序按算法框） | grad=None → 跳过（文档明文，不是无定义）；eps = 0 且 v̂ = 0 → 除零，无定义 |
| 归一化 | normalized_shape 与输入匹配；num_channels 能被 num_groups 整除；eps ≥ 0 | NORM-A1（RMSNorm eps=None 取哪个 dtype 的 finfo）；NORM-A3（(1+γ) 变体是另一声明公式） | NORM-A2：BatchNorm 无偏 running_var 在每通道计数为 1 时；N·spatial = 1 时方差为 0、输出为 β |
| 嵌入 | 索引 ∈ [0, V)；per_sample_weights 只允许 mode=sum；offsets 非减且在范围内；include_last_offset 时末 offset 为末尾 | EMB-A1（max_norm 重归一的小常数属接口/数值项） | EMB-A2：mean 的 bag 全为 padding_idx；空 bag → 零（文档明文） |

未入契约的 2b 家族：注意力、序列打包、RoPE、调度、裁剪与 AMP、训练程序层（标签右移与跨 rank 平均）、MoE、matmul/linear、归约与扫描、激活、gather/布局、保存与重算。它们的「无定义」判定在契约补齐前标为「事后裁决」。
