# 输入契约 v2（2026-10-08）— 第一档全部家族

版本：规格包 specs_phase1 v0.4、specs_phase2 v0.2；锁定版本 PyTorch 2.10 文档。执行方不得在看到结果后增删条目；修改走审阅入库。五类处理见保证与裁决表第 3 节。标注「文档待抓取核对」的条目，语义按我所知的官方公式书写，但抓取核对尚未完成，审阅时优先核。

| 家族 | 合法域（之外 → 拒绝） | 文档未规定 / 多种声明约定（读法） | 实数目标无定义（记录约定） | 规格文件 |
| --- | --- | --- | --- | --- |
| 交叉熵 | 类别下标 ∈ [0,C) 或 = ignore_index；概率目标同形；weights 长 C；label_smoothing ∈ [0,1] | CE-A1、CE-A2、CE-A6 | CE-A3 | phase1/spec_cross_entropy |
| 池化 | 2p ≤ d(k−1)+1；有效核 ≤ 带 padding 输入；divisor_override 非零整数；avg 输入有限 | POOL-A1、A2、A3；MaxPool1d ceil 两版文档 | POOL-A5；R_ignore 过滤后无值 → 未建立 | phase1/spec_pooling |
| index/scatter | 索引 ∈ [0,size)；dim ∈ {0,1}；1-D/2-D；形状匹配 | IDX-A3、A4 | 全 NaN 过滤后 → 未建立；整数 mean 不在范围 | phase1/spec_index_scatter |
| 累加窗口 | token 数 ≥ 0 | — | ACC-A1 | phase1/spec_accumulation |
| matmul/linear | 内维一致；bias 长度 = out；本规格支持 2-D 与同批 3-D，不广播批维（BASE-A4） | — | — | spec_base_ops |
| 归约与扫描 | 空归约：sum=0、prod=1 合法；var/std 需 N − correction > 0 | — | amax/amin/mean 的空归约；var 自由度 ≤ 0（BASE-A1）；softmax 全 −inf 行（BASE-A2）；logsumexp 全 −inf = −inf | spec_base_ops |
| 激活 | 输入有限 | GELU 的 erf 与 tanh 是两种声明公式（BASE-A3）；ReLU 在 0 处的次梯度未规定（规格返回集合 {0, 上游}） | — | spec_base_ops（erf 为声明近似，非严格区间） |
| gather / index_select / 布局 | 索引在范围内；2-D，dim ∈ {0,1} | — | — | spec_base_ops |
| 保存与重算 | dropout 固定为零；同输入同参数 | — | — | spec_training_program.prop_checkpoint_equivalence（性质，不需 f） |
| embedding / embedding_bag | 索引 ∈ [0,V)；per_sample_weights 只允许 sum；offsets 非减；include_last_offset 末 offset 为末尾 | EMB-A1（max_norm 重归一的小常数） | EMB-A2（全 padding 的 mean）；空 bag → 零（文档明文） | spec_embedding |
| 注意力 | attn_mask 与 is_causal 不得同时给（文档明文）；bool mask True = 参与；GQA 需 H_q 是 H_kv 的倍数 | ATT-C1（causal 对齐：SDPA 左上，FA 右下，按各自声明）；ATT-C2（GQA 映射）；ATT-A2（滑动窗口含不含边界） | ATT-A1（整行被 mask） | spec_attention |
| 序列打包 | 文档长度 ≥ 1 | 位置重置的起点（0 或声明值） | — | spec_attention（packed_mask、packed_position_ids） |
| RoPE | d 为偶数 | ROPE-C1（rotate-half 与交错；位置偏移；缩放变体按声明公式） | — | spec_attention（rope_apply） |
| 归一化 | normalized_shape 匹配；C 能被 G 整除；eps ≥ 0 | NORM-A1（RMSNorm eps=None 的 dtype）；NORM-A3（(1+γ) 变体） | NORM-A2（BN 单元素无偏方差；N·spatial=1） | spec_normalization |
| 优化器 | 超参为数；nesterov 需 momentum>0 且 dampening=0；梯度有限 | OPT-A1（amsgrad 的 max 对象按锁定版本文档，旧版为 R_old）；OPT-A2 | grad=None → 跳过（文档明文）；eps=0 且 v̂=0 → 除零 | spec_optimizers |
| 学习率调度 | T_max > 0；total > warmup | SCH-A1（闭式与递推）；SCH-A2（计步起点） | — | spec_training_program |
| 梯度裁剪与 AMP | norm_type ∈ {1,2,inf}（本规格） | CLIP-A1（1e-6 是文档公式的一部分）；AMP-A1（哪些梯度参与 inf 检查） | error_if_nonfinite=False 且范数非有限 → 按非有限系数缩放（文档明文，记为数值失败栏） | spec_training_program |
| 训练程序层 | — | PROG-A1（跨 rank 的平均：任务要求为全局 token 平均；DDP 的均值的均值为已知错误变体） | 窗口/所有 rank 无有效 token | spec_training_program、phase1/spec_accumulation |
| MoE | 1 ≤ k ≤ E；capacity ≥ 1 | MOE-C1（重归一）；MOE-C2（top-k 并列：集合）；MOE-C3（容量丢弃或 dropless）；MOE-C4（aux loss 的 α 与是否计丢弃 token） | 区间重叠过宽使 top-k 不可判 → 未建立 | spec_moe |

文档待抓取核对的条目：CosineAnnealingLR 的公式与递推说明；HF get_cosine_schedule_with_warmup 的 lr_lambda；GradScaler 的默认值与跳步规则；torch.utils.checkpoint 的重算约定；torch.var 的 correction 文案；torch.matmul 的广播规则；Embedding/EmbeddingBag 的 padding_idx 与 max_norm 原文（第一版规格已按我所知书写）。
