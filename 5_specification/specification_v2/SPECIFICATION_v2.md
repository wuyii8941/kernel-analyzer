# 统一规格 v2.4：方法与现状（2026-10-15）

所有数字由 catalogue.py、derive.py、fuzz_relations.py 计算，见 MANIFEST.json。理论见 THEORY.md。

## 1 我们怎样做 specification

**原则。**规格里的每一行都要能回答「从哪里来、凭什么成立」。来源只有三个生成器，每个生成器的产物都带来处和证明状态；没有来处的条目不入库。

| 生成器 | 输入 | 产物 | 状态取值 |
|---|---|---|---|
| G1 文档 → 算法规格 | 官方文档、ISA、原论文的公式与前置条件（锁定版本） | 每个角色的 f_alg：精确有理数、严格区间、允许集合或声明近似；文档含糊处登记为读法 | 来源、读法、参照类型、接受域、独立验证、已知问题六个字段 |
| G2 数学 → 关系 | 各角色的 f_alg | 按 7 条推导规则组织的候选关系 | exact / symbolic（固定小形状）/ enclosure（证据，不是证明）/ counterexample |
| G3 实现路径 → 路径等价 | 框架自身的功能轴 | 同一框架两条路径的声明语义应相同 | 来处为 EAGLE 规则编号、文档或任务定义 |

**理论的要点（THEORY.md）。**总差异 K − f_task 唯一地分为执行相对声明（e_num）、程序相对文档算法（e_sem）、算法设计相对任务（e_design）三项（定理 1）；关系只看见随变换变化的误差，盲区是各变换约束的交集（定理 2）；以实现为参照的做法看不见所有参照实现共有的错误（定理 3）；在声明的配置集合上，任何差异必落入三项之一、写明的盲区或执行不合法（定理 4）；绝对意义上的完备不可能（定理 5）；在同一配置上，我们的可见集合包含已有五类做法的可见集合（定理 6）。

**关于「自动推导」要说准。**derive.py 是：候选关系由人按规则指定，检查自动执行，方法与状态自动记录，格子由记录计算。它不是从任意 f 自动产生关系的引擎。「当前模板不支持」不表示数学上没有这种性质。

## 2 精确计数

- 角色 31 个，全部有算法规格（其中 CKPT 是性质比较器，不是独立数值目标；GELU-erf 是声明近似）；带「已改」记录的角色 8 个
- 本版新写的规格：参数平均（EMA/SWA）、KV cache（连续、分页、RoPE）、分块 KV 压缩（均匀与门控聚合，共享与按偏移编号的参数）、稀疏 COO、状态的保存与加载；phase3 测试 15 项
- 候选关系检查 70 项：成立 55，反例 15；精确 29、符号 6、包围 20、反例 15
- 格子（24 个有元数据的角色 × 7 族）168 个：有结果 46，模板不支持 49，未推导 73
- 路径等价轴 10 条，全部有可执行规格（稀疏一轴的工具侧 kernel 尚不在 DSL 范围）
- 随机输入验证：26 条关系 × 200 组，其中阴性对照 4 条；真关系全部成立、实现错误 0，阴性对照各至少失败一次
- 测试：phase1 105、phase2 89、phase3 15、关系实例 9、oracle 4，全部通过

## 3 历次关闭的问题

| 角色 | 问题 | 修正与复核 |
|---|---|---|
| ATTN | v1 的 window 只支持 upper_left 且按查询局部坐标 | v2 复核：Lq=1、Lk=5、bottom_right、window=(1,0)、V_j=j → 7/2 |
| ACT | v1 的 ReLU 零点集合 {0, upstream} 不对应任何文档为两种 profile | 第二阶段 |
| MOE | v2.2 的逐 token 枚举丢失容量状态为联合枚举 | v2.3：两 token 并列、容量 1 → {(1,0),(1,3),(3,0),(3,1)}；无并列、容量 1 → [1,0] |
| BN | v1 的批计数在通道循环内多加 | v2 复核：首批后计数 1、running_mean=[2,12]、running_var=[2,8] |
| SGD | v1 的 maximize 在 weight decay 之后取负 | v2 复核：θ=1, g=0, lr=wd=1/10, maximize → 99/100 |
| RMSPROP | v2.2 的区间类型缺平方，centered 路径第二步抛错 | v2.3：centered + weight_decay 两步可求值，第二步包围含 −0.68617（与 PyTorch 2.10 一致） |
| CKPT | v2.2 的长度不一致被静默截短为拒绝 | — |
| CAST | v1 定向舍入溢出与 E4M3 最大值错误 | test_oracle：与 numpy 4000 组一致；定向溢出与 FP8 边界逐项核对 |

## 4 G1：角色与来源

| 角色 | 来源 | 读法 | 参照类型 | 接受域 | 独立验证 | 已知问题 |
|---|---|---|---|---|---|---|
| CE | torch.nn.functional.cross_entropy 文档（2.x） | ignore_index 与权重并存时的分母：按文档，忽略行不计 | 逐点严格区间（Decimal）；全部忽略且 mean → NaN（集合语义：未定义） | 有限 logits；类别数 ≥ 2 | 第一阶段独立复算；D3 符号证明梯度公式（n=3） | 无 |
| AVGPOOL | torch.nn.functional.avg_pool 文档 | ceil_mode 下窗口越界的分母：R1/R2 两种读法登记 | 精确有理数 | 1D/2D；padding ≤ kernel/2 | 第一阶段独立复算；伴随恒等式精确 | 无 |
| MAXPOOL | torch.nn.functional.max_pool 文档 | 并列时返回集合 | 集合（并列） | 同上 | 第一阶段 | 无 |
| INDEXREDUCE | torch.Tensor.index_reduce_ / scatter_reduce 文档 | NaN 传播读法 | 精确；amax/amin 并列为集合 | 下标在范围内 | 第一阶段；B020 由此发现 | 无 |
| ACCUM | 任务定义：窗口内 token 平均 | 无 | 精确 | 至少一个 token | HF 问题召回 | 无 |
| ATTN | torch SDPA 文档；FlashAttention 的 window 约定 | causal 对齐两种；softcap；window=(left,right) 相对对齐后的对角线 | 严格区间 | 至少一个允许的 key（否则按声明拒绝或置零） | v2 复核：Lq=1、Lk=5、bottom_right、window=(1,0)、V_j=j → 7/2 | v1 的 window 只支持 upper_left 且按查询局部坐标，已改 |
| GQA | GQA 论文 | 无 | 精确映射 | H_q 整除 H_kv | 第二阶段 | 无 |
| PACK | 任务定义：文档独立 | 无 | 精确 mask 与位置 | 长度 ≥ 1 | D2：跨文档权重恒为零 | 无 |
| ROPE | RoFormer；两种布局 | rotate_half 与 interleaved | 严格区间 | 偶数维 | D4：只依赖相对位置 | 无 |
| MATMUL | 数学定义 | 无 | 精确 | 形状匹配 | D5、D3 精确 | 无 |
| REDUCE | 数学定义；torch 的 correction 参数 | var 的 correction | 精确 | 非空 | D7 精确 | 无 |
| SOFTMAX | 数学定义 | 全 −inf 行按声明拒绝 | 严格区间（200 bit） | 至少一个有限元素 | D4、D6、D3 符号证明（n=3） | 无 |
| ACT | torch 激活函数文档 | GELU-erf 使用声明的高精度近似（不是严格区间）；ReLU 零点两种 profile（torch：0；math：次梯度区间） | 区间 / 声明近似 / 集合 | 有限输入 | 第二阶段 | v1 的 ReLU 零点集合 {0, upstream} 不对应任何文档，已改为两种 profile |
| GATHER | torch.gather / index_select 文档 | 无 | 精确 | 下标在范围内 | 伴随恒等式 | 无 |
| EMB | torch.nn.Embedding / EmbeddingBag 文档 | padding_idx 行梯度为零 | 精确 / 区间（renorm） | 下标在范围内 | 第二阶段 | 无 |
| MOE | Switch Transformer / Mixtral 的路由描述；容量按 token 顺序消耗 | 并列时所有合法 top-k 集合均允许 | moe_forward_joint：按容量状态枚举全部允许的联合轨迹（有预算，超出则 not_established）；moe_forward 只给代表值并标 representative_only | k ≤ E | v2.3：两 token 并列、容量 1 → {(1,0),(1,3),(3,0),(3,1)}；无并列、容量 1 → [1,0] | v2.2 的逐 token 枚举丢失容量状态，已改为联合枚举 |
| NORM | torch.nn.LayerNorm / RMSNorm / GroupNorm 文档 | 无 | 严格区间 | 方差分母非零 | D4/D5：LN 平移不变（任意 eps）、缩放不变（仅 eps=0）；RMSNorm 平移不变不成立（反例） | 无 |
| BN | torch.nn.BatchNorm 文档（running 统计、momentum=None 的累计平均） | 训练用有偏方差归一、running_var 用无偏方差 | 严格区间 | N ≥ 2（训练） | v2 复核：首批后计数 1、running_mean=[2,12]、running_var=[2,8] | v1 的批计数在通道循环内多加，已改 |
| ADAM | torch.optim.Adam / AdamW 算法框 | amsgrad 的两种读法登记 | 严格区间（sqrt） | 超参数在文档范围内；theta/grad 长度一致（否则拒绝），grad=None 按文档跳过 | D8：零有效梯度衰减；coupled decay 下不成立（反例） | 无 |
| SGD | torch.optim.SGD 算法框 | 无 | 精确 | nesterov 需 momentum>0、dampening=0；theta/grad 长度一致（否则拒绝），grad=None 按文档跳过 | v2 复核：θ=1, g=0, lr=wd=1/10, maximize → 99/100 | v1 的 maximize 在 weight decay 之后取负，已改 |
| RMSPROP | torch.optim.RMSprop 算法框 | 无 | 严格区间 | 超参数在文档范围内；theta/grad 长度一致（否则拒绝），grad=None 按文档跳过 | v2.3：centered + weight_decay 两步可求值，第二步包围含 −0.68617（与 PyTorch 2.10 一致） | v2.2 的区间类型缺平方，centered 路径第二步抛错，已改 |
| SCHED | torch.optim.lr_scheduler；HF get_cosine_schedule_with_warmup | 无 | 严格区间（cos） | 步数在范围内 | D5：对 base_lr 线性 | 无 |
| CLIP | torch.nn.utils.clip_grad_norm_ 文档（分母含 1e-6） | 无 | 严格区间 | 范数有限 | D5：不齐次（反例） | 无 |
| SCALER | torch.amp.GradScaler 文档 | 无 | 精确状态机 | — | 第二阶段 | 无 |
| LABELS | 任务定义：标签右移与计数 | 无 | 精确 | — | 第二阶段 | 无 |
| RANKMEAN | 任务定义：跨 rank 的 token 平均 | 无 | 精确 | 至少一个 token | 第二阶段 | 无 |
| CKPT | 数学恒等：重算等于保存 | 无 | 比较两份结果的性质，不是独立参照 | 两份结果长度一致（否则拒绝） | — | v2.2 的长度不一致被静默截短，已改为拒绝 |
| CAST | IEEE 754-2019；OCP MX / NVIDIA FP8（E4M3FN 无 inf、最大 448；E5M2 有 inf、最大 57344） | 溢出：RNE → inf/NaN；定向 → 同号最大有限值；saturate=True 强制饱和 | 精确 | 全部有限输入 | test_oracle：与 numpy 4000 组一致；定向溢出与 FP8 边界逐项核对 | v1 定向舍入溢出与 E4M3 最大值错误，已改 |
| EMA | torch.optim.swa_utils.AveragedModel、get_ema_avg_fn、get_swa_avg_fn 文档 | 第一次更新为复制；n_averaged 属于状态 | 精确有理数 | decay ∈ [0, 1]；参数长度一致 | phase3：SWA 等于算术平均、EMA 展开式（随机 50 组）；随机验证 200 组 | 无 |
| KVCACHE | 任务定义：因果解码的增量输出等于同一序列的因果 prefill | RoPE 在缓存前按绝对位置旋转 | 严格区间（连续、分页两种 cache） | 至少一个位置 | phase3：decode = prefill（连续、分页、有无 RoPE）；按位置 0 旋转的变异必须失败 | 无 |
| KVCOMP | arXiv 2609.36322 §3.1 的压缩算法族（窗口 W、步长 S、负载映射、门控、聚合） | 只压缩完整窗口；因果可见性：窗口末位小于查询位置 | 精确（均匀聚合）/ 严格区间（门控） | L ≥ W | phase3：平移步长等变（四种组合）；平移一位与按偏移编号参数下的窗口内置换给出反例；随机验证 200 组 | 无 |

## 5 G2：检查日志（全部 70 项）

| 角色 | 规则 | 命题 | 方法 | 状态 | 见证 |
|---|---|---|---|---|---|
| CE | D2 定义域与无关输入 | 被忽略行的 logits 不进入损失与梯度 | exact | holds |  |
| CE | D2 定义域与无关输入 | 全部忽略且 reduction=mean：未定义（NaN） | exact | holds |  |
| INDEXREDUCE | D2 定义域与无关输入 | include_self=False：被写入位置的原值无关 | exact | holds |  |
| ATTN | D2 定义域与无关输入 | 被 mask 的 key/value 的值无关 | enclosure | holds |  |
| CAST | D2 定义域与无关输入 | 窄格式可表示的值经宽格式往返不变 | exact | holds |  |
| CAST | D2 定义域与无关输入 | 任意在范围内的值往返不变 | counterexample | fails | 1/3 is in range but not representable |
| CAST | D2 定义域与无关输入 | 定向舍入溢出给出同号最大有限值（IEEE 754） | exact | holds |  |
| AVGPOOL | D3 微分（伴随恒等式 ⟨J u, v⟩ = ⟨u, Jᵀ v⟩） | ⟨avg_pool(x), g⟩ = ⟨x, avg_pool_backward(g)⟩（线性算子的伴随） | exact | holds |  |
| MATMUL | D3 微分（伴随恒等式 ⟨J u, v⟩ = ⟨u, Jᵀ v⟩） | 线性层的伴随恒等式 | exact | holds |  |
| SOFTMAX | D3 微分（伴随恒等式 ⟨J u, v⟩ = ⟨u, Jᵀ v⟩） | softmax 的 Jacobian 为 diag(p) − ppᵀ | symbolic | holds | n=3 |
| SOFTMAX | D3 微分（伴随恒等式 ⟨J u, v⟩ = ⟨u, Jᵀ v⟩） | ⟨f(x), g⟩ = ⟨x, Jᵀg⟩ 对 softmax | counterexample | fails | nonlinear f: the identity holds only for linear operators (v1 R23 was wrong) |
| SOFTMAX | D4 对称（平移、置换） | softmax(x + c·1) = softmax(x) | symbolic | holds | n=3, sympy simplify |
| SOFTMAX | D4 对称（平移、置换） | logsumexp(x + c·1) = logsumexp(x) + c（等变，不是不变） | symbolic | holds |  |
| CE | D4 对称（平移、置换） | CE(x + c_n·1) = CE(x) 逐行平移 | enclosure | holds |  |
| CE | D4 对称（平移、置换） | 类别置换：损失不变 | enclosure | holds |  |
| NORM_LN | D4 对称（平移、置换） | LN(x + b) = LN(x)（任意 eps） | enclosure | holds |  |
| NORM_LN | D5 线性与齐次 | LN(a·x) = sign(a)·LN(x)，eps > 0 | counterexample | fails | x=[1,3,7], a=2, eps=1/100 |
| NORM_LN | D5 线性与齐次 | LN(a·x) = sign(a)·LN(x)，eps = 0 | enclosure | holds |  |
| NORM_RMS | D4 对称（平移、置换） | RMSNorm(x + b) = RMSNorm(x) | counterexample | fails | x=[1,0], b=1: [√2,0] vs [2√(2/5),√(2/5)] |
| NORM_RMS | D5 线性与齐次 | RMSNorm(a·x) = sign(a)·RMSNorm(x)，eps = 0 | enclosure | holds |  |
| ROPE | D6 守恒（和式恒等） | ‖rope(x, m)‖² = ‖x‖² | enclosure | holds |  |
| ROPE | D4 对称（平移、置换） | ⟨rope(q,m), rope(k,n)⟩ 只依赖 m − n | enclosure | holds |  |
| ATTN | D4 对称（平移、置换） | 查询行置换 ⇒ 输出行同置换 | enclosure | holds |  |
| MATMUL | D5 线性与齐次 | (sA)B = s(AB) | exact | holds |  |
| AVGPOOL | D5 线性与齐次 | avg_pool(2x) = 2·avg_pool(x) | exact | holds |  |
| SCHED | D5 线性与齐次 | cosine(2·base, 0) = 2·cosine(base, 0)（对 base_lr 线性） | enclosure | holds |  |
| CLIP | D5 线性与齐次 | clip(2g) = 2·clip(g)（齐次） | counterexample | fails | g=[3,4], max_norm=1: clip(2g) ≈ clip(g) (both land on the unit vector), so 2·clip(g) ≠ clip(2g) |
| INDEXREDUCE | D5 线性与齐次 | index_add 对 source 线性 | exact | holds |  |
| ACCUM | D5 线性与齐次 | window_loss 对逐 token 损失线性 | exact | holds |  |
| MAXPOOL | D5 线性与齐次 | max_pool(a·x) = a·max_pool(x)，a > 0（正齐次，不是线性） | exact | holds |  |
| MAXPOOL | D5 线性与齐次 | max_pool(−x) = −max_pool(x)（线性） | counterexample | fails | max(−x) = −min(x) |
| SOFTMAX | D6 守恒（和式恒等） | Σ_k softmax_k = 1 | symbolic | holds | n=3 |
| CE | D3 微分（伴随恒等式 ⟨J u, v⟩ = ⟨u, Jᵀ v⟩） | ∂l/∂x_k = p_k Σa − a_k 等于 −Σ a log softmax 的导数 | symbolic | holds | n=3 |
| CE | D6 守恒（和式恒等） | Σ_k ∂l/∂x_k = 0 | symbolic | holds | n=3 |
| INDEXREDUCE | D6 守恒（和式恒等） | Σ index_add = Σ self + α Σ source | exact | holds |  |
| INDEXREDUCE | D6 守恒（和式恒等） | Σ scatter_reduce(sum) = Σ self + Σ src，include_self=False | counterexample | fails | self=[10], src=[2] -> 2 |
| MOE | D6 守恒（和式恒等） | 重归一的合并权重和为一 | enclosure | holds |  |
| MOE | D6 守恒（和式恒等） | 未重归一的 top-k 权重和为一 | counterexample | fails | raw top-2 of softmax([1,1/2,-1]) sums to < 1 |
| MOE | D8 状态递推 | 并列 + 容量 1：允许的联合轨迹恰为 {(1,0),(1,3),(3,0),(3,1)} | exact | holds | [(Fraction(1, 1), Fraction(0, 1)), (Fraction(1, 1), Fraction(3, 1)), (Fraction(3, 1), Fraction(0, 1)), (Fraction(3, 1), Fraction(1, 1))] |
| ATTN | D6 守恒（和式恒等） | 注意力权重行和为一 | enclosure | holds |  |
| PACK | D2 定义域与无关输入 | 打包后跨文档权重恒为零 | exact | holds |  |
| SOFTMAX | D7 分解（次序、分块、拆批） | Σ 的折叠次序无关 | exact | holds |  |
| SOFTMAX | D7 分解（次序、分块、拆批） | 在线 logsumexp 合并等于整体 | enclosure | holds |  |
| CE | D7 分解（次序、分块、拆批） | reduction=sum：拆批后相加等于整体 | enclosure | holds |  |
| BN | D7 分解（次序、分块、拆批） | 训练模式拆批后拼接等于整体 | counterexample | fails | batch statistics depend on the whole batch |
| ACCUM | D7 分解（次序、分块、拆批） | 微批重划分不改变窗口损失 | exact | holds |  |
| ACCUM | D7 分解（次序、分块、拆批） | 「均值的均值」在长度不等时等于窗口损失 | counterexample | fails | [[1,2,3],[4]] |
| ADAM | D8 状态递推 | 零梯度：m ← β₁m，v ← β₂v（decoupled 或 wd=0） | exact | holds |  |
| ADAM | D8 状态递推 | 零梯度：m ← β₁m，coupled weight decay | counterexample | fails | m = (1-β₁)·wd·θ = 1/100 ≠ 0: the rule needs zero EFFECTIVE gradient |
| SGD | D8 状态递推 | maximize 与 weight decay 的次序（文档：先取负再衰减） | exact | holds | θ=1,g=0,lr=wd=1/10 -> 99/100 |
| BN | D4 对称（平移、置换） | 通道置换（同时置换 γ、β 与 running 状态）⇒ 输出与状态同置换 | enclosure | holds |  |
| BN | D8 状态递推 | momentum=None：累计平均，第一批后计数为 1、running_mean 为批均值 | exact | holds |  |
| EMA | D8 状态递推 | SWA 经 n 次更新等于算术平均（对 n 归纳） | exact | holds |  |
| EMA | D8 状态递推 | EMA 展开式 a_n = d^(n−1)p_1 + Σ(1−d)d^(n−k)p_k | exact | holds |  |
| EMA | D5 线性与齐次 | EMA 对（旧状态, 当前参数）线性：同乘 2 结果乘 2 | exact | holds |  |
| EMA | D8 状态递推 | 源参数恒定时 EMA 停在源值（不动点） | exact | holds |  |
| KVCACHE | D7 分解（次序、分块、拆批） | decode（连续 cache）逐位置等于因果 prefill | enclosure | holds |  |
| KVCACHE | D7 分解（次序、分块、拆批） | decode（分页 cache）逐位置等于因果 prefill | enclosure | holds |  |
| KVCACHE | D7 分解（次序、分块、拆批） | decode（连续 cache，RoPE 绝对位置）逐位置等于因果 prefill | enclosure | holds |  |
| KVCACHE | D7 分解（次序、分块、拆批） | decode（分页 cache，RoPE 绝对位置）逐位置等于因果 prefill | enclosure | holds |  |
| KVCACHE | D7 分解（次序、分块、拆批） | 缓存的 key 按位置 0 旋转时 decode 仍等于 prefill | counterexample | fails | RoPE position must be the absolute position |
| KVCOMP | D4 对称（平移、置换） | 平移 S（步长）个位置：压缩输出平移一个窗口（等变） | exact | holds |  |
| KVCOMP | D4 对称（平移、置换） | 平移 1 个位置：存在某种对应使输出不变 | counterexample | fails | 窗口成员改变：相位依赖是结构性的，属于 f_alg − f_task 一栏 |
| KVCOMP | D4 对称（平移、置换） | 窗口内置换不变（参数在偏移间共享） | exact | holds |  |
| KVCOMP | D4 对称（平移、置换） | 窗口内置换不变（参数按偏移编号） | counterexample | fails | 按偏移编号的参数打破窗口内对称 |
| KVCOMP | D5 线性与齐次 | 均匀聚合对输入线性 | exact | holds |  |
| SPARSE | D7 分解（次序、分块、拆批） | 路径等价：spmm(COO, B) = dense(COO)·B（重复下标求和） | exact | holds |  |
| SPARSE | D7 分解（次序、分块、拆批） | coalesce 不改变稠密值 | exact | holds |  |
| STATEIO | D8 状态递推 | 路径等价：保存、加载后继续等于直接继续（Adam） | enclosure | holds |  |
| STATEIO | D8 状态递推 | 加载时遗漏步数计数器仍等于直接继续 | counterexample | fails | 偏差修正依赖 t |

## 6 G2：格子表（由日志计算）

| 角色 | F1 对称 | F2 无关输入 | F3 线性 | F4 守恒 | F5 分解 | F6 微分 | F7 状态 |
|---|---|---|---|---|---|---|---|
| CE | 成立 2 | 成立 2 | 未推导 | 成立 1 | 成立 1 | 成立 1 | 当前模板不支持（不是数学排除） |
| SOFTMAX | 成立 2 | 未推导 | 未推导 | 成立 1 | 成立 2 | 成立 1，不成立 1 | 当前模板不支持（不是数学排除） |
| NORM_LN | 成立 1 | 未推导 | 成立 1，不成立 1 | 未推导 | 未推导 | 未推导 | 当前模板不支持（不是数学排除） |
| NORM_RMS | 成立 0，不成立 1 | 未推导 | 成立 1 | 未推导 | 未推导 | 未推导 | 当前模板不支持（不是数学排除） |
| BN | 成立 1 | 未推导 | 未推导 | 未推导 | 成立 0，不成立 1 | 未推导 | 成立 1 |
| MATMUL | 未推导 | 未推导 | 成立 1 | 未推导 | 未推导 | 成立 1 | 当前模板不支持（不是数学排除） |
| AVGPOOL | 未推导 | 未推导 | 成立 1 | 未推导 | 未推导 | 成立 1 | 当前模板不支持（不是数学排除） |
| INDEXREDUCE | 当前模板不支持（不是数学排除） | 成立 1 | 成立 1 | 成立 1，不成立 1 | 未推导 | 未推导 | 当前模板不支持（不是数学排除） |
| ACCUM | 当前模板不支持（不是数学排除） | 未推导 | 成立 1 | 未推导 | 成立 1，不成立 1 | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） |
| ADAM | 当前模板不支持（不是数学排除） | 未推导 | 未推导 | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） | 成立 1，不成立 1 |
| SGD | 当前模板不支持（不是数学排除） | 未推导 | 未推导 | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） | 成立 1 |
| CLIP | 当前模板不支持（不是数学排除） | 未推导 | 成立 0，不成立 1 | 未推导 | 未推导 | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） |
| SCHED | 当前模板不支持（不是数学排除） | 未推导 | 成立 1 | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） |
| ATTN | 成立 1 | 成立 1 | 未推导 | 成立 1 | 未推导 | 未推导 | 当前模板不支持（不是数学排除） |
| ROPE | 成立 1 | 未推导 | 未推导 | 成立 1 | 当前模板不支持（不是数学排除） | 未推导 | 当前模板不支持（不是数学排除） |
| MAXPOOL | 未推导 | 未推导 | 成立 1，不成立 1 | 未推导 | 未推导 | 未推导 | 当前模板不支持（不是数学排除） |
| MOE | 未推导 | 未推导 | 未推导 | 成立 1，不成立 1 | 未推导 | 未推导 | 成立 1 |
| CAST | 未推导 | 成立 2，不成立 1 | 未推导 | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） |
| PACK | 未推导 | 成立 1 | 未推导 | 未推导 | 未推导 | 未推导 | 当前模板不支持（不是数学排除） |
| EMA | 当前模板不支持（不是数学排除） | 未推导 | 成立 1 | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） | 成立 3 |
| KVCACHE | 未推导 | 未推导 | 未推导 | 未推导 | 成立 4，不成立 1 | 当前模板不支持（不是数学排除） | 未推导 |
| KVCOMP | 成立 2，不成立 2 | 未推导 | 成立 1 | 未推导 | 未推导 | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） |
| SPARSE | 未推导 | 未推导 | 未推导 | 未推导 | 成立 2 | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） |
| STATEIO | 当前模板不支持（不是数学排除） | 未推导 | 未推导 | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） | 当前模板不支持（不是数学排除） | 成立 1，不成立 1 |

## 7 G3：路径等价轴

| 轴 | 两条路径 | 来处 | 在我们工具里 |
|---|---|---|---|
| 编译 | eager 对 torch.compile/Inductor | EAGLE 规则 1 | K − G 两栏拆分；Inductor 生成的 Triton 在工具范围内 |
| 布局 | 连续对转置/步幅视图；batch-major 对 time-major | EAGLE 规则 10 | 声明层精确比较 |
| dtype | 共同可表示范围内的两种 dtype | EAGLE 规则 12 | CAST 规格给出可表示集合，精确判定 |
| 稀疏 | 稠密对 COO 稀疏表示（含重复下标） | EAGLE 规则 8；torch.sparse 文档 | spec_sparse：spmm 等于稠密矩阵乘、coalesce 与条目次序不改变稠密值（精确）；工具侧稀疏 kernel 尚不在 DSL 范围 |
| 批大小 | 整批对拆批 | EAGLE 规则 15；D7 | CE sum 精确；BN 训练模式不成立（反例已记录） |
| cache | prefill 对 decode；连续 cache 对分页 cache | 任务定义 | spec_kvcache：逐位置相等（严格区间），RoPE 位置错误的变异必须失败 |
| 检查点 | 重算对保存 | 数学恒等 | CKPT 性质 |
| 分布式 | 单卡对跨 rank 平均 | 任务定义 | RANKMEAN 精确 |
| 保存加载 | 保存后加载再继续对直接继续 | EAGLE 规则 16；state_dict 文档 | spec_state_io：Adam 与 EMA 恢复后等于直接继续；遗漏 t 或 n_averaged 的变异必须失败 |
| 内置对外部 | 内置 padding/mask 对显式构造 | EAGLE 规则 2、14 | PACK、ATTN 的 mask 精确比较 |

## 8 随机输入验证

| 关系 | 成立 | 失败 | 拒绝 | 实现错误 |
|---|---|---|---|---|
| SOFTMAX: softmax(x + c) = softmax(x) | 200 | 0 | 0 | 0 |
| SOFTMAX: Σ softmax = 1 | 200 | 0 | 0 | 0 |
| SOFTMAX: online logsumexp = whole | 200 | 0 | 0 | 0 |
| CE: row shift invariance | 200 | 0 | 0 | 0 |
| CE: ignored rows irrelevant | 200 | 0 | 0 | 0 |
| CE: Σ_k grad = 0 per row | 200 | 0 | 0 | 0 |
| CE: reduction=sum splits over batches | 200 | 0 | 0 | 0 |
| INDEXREDUCE: Σ index_add = Σ self + α Σ src | 200 | 0 | 0 | 0 |
| INDEXREDUCE: include_self=False ignores self at written rows | 200 | 0 | 0 | 0 |
| ACCUM: window loss = Σ / count and regroup invariance | 200 | 0 | 0 | 0 |
| MATMUL: (sA)B = s(AB) | 200 | 0 | 0 | 0 |
| NORM_LN: shift invariance, any eps | 200 | 0 | 0 | 0 |
| NORM_RMS: positive scale invariance, eps=0 | 200 | 0 | 0 | 0 |
| ADAM: zero effective gradient -> geometric decay of m, v (decoupled) | 200 | 0 | 0 | 0 |
| AVGPOOL: linearity and adjoint | 200 | 0 | 0 | 0 |
| ROPE: relative position and norm preservation | 200 | 0 | 0 | 0 |
| ATTN: query-row permutation equivariance; masked K/V irrelevant | 200 | 0 | 0 | 0 |
| MOE: renormalised top-k weights sum to 1 | 200 | 0 | 0 | 0 |
| EMA: SWA = mean, EMA closed form | 200 | 0 | 0 | 0 |
| KVCACHE: decode (contiguous / paged, with and without RoPE) = causal prefill | 200 | 0 | 0 | 0 |
| KVCOMP: shift by the stride is equivariant (uniform and gated, shared or per-offset) | 200 | 0 | 0 | 0 |
| SPARSE: spmm = dense matmul; coalesce and entry order do not change the dense value | 200 | 0 | 0 | 0 |
| NEGATIVE CONTROL NORM_RMS: shift invariance (false) | 0 | 194 | 6 | 0 |
| NEGATIVE CONTROL CE: reduction=mean splits additively (false) | 0 | 200 | 0 | 0 |
| NEGATIVE CONTROL KVCOMP: shift by one position is equivariant (false: phase dependence) | 0 | 200 | 0 | 0 |
| NEGATIVE CONTROL ADAM: zero raw gradient decays m with coupled wd (false) | 0 | 200 | 0 | 0 |

## 9 发布范围与边界

**可启用的子集**：G1 中参照类型为精确或严格区间、已知问题为「无」或「已改」、且在日志或随机验证中至少有一项阳性记录的角色，按「角色 + 来源版本 + 模式 + 输入域 + 观察量 + 参照类型」登记启用。

**边界**：
1. 73 个未推导格子是没写的候选；49 个「模板不支持」不是数学排除。
2. 符号证明只在固定小形状上；包围检查是证据不是证明；关系检查不能排除共享的公式错误（THEORY 定理 2 的 β 例）。
3. 没有从 f 自动产生关系的引擎；元数据手填。
4. 关系只在声明层检查；执行层按变换参数分组的条件均值未接进统计层（相位敏感的执行层测量依赖这一步）。
5. MoE 联合枚举有预算；GELU-erf 是声明近似；稀疏轴的工具侧 kernel 不在 DSL 范围。
6. 真实语料上的覆盖比例尚未测量（THEORY 定理 5：这只能测，不能证）。
