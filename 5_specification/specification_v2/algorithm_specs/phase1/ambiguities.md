# 规格歧义与文档不一致清单（第一阶段，v0.4，含 2026-10-07 审阅决定）

规则：文档没有写明的地方，规格不替任何一方做决定，而是把每种**有文档依据**的读法都实现出来。一个读法必须有来源、适用条件和待决定事项；没有依据、只是「某种实现可能这么写」的变体不是读法，只作为带标签的错误变体用于分类。比较时记录候选与 eager 各自符合哪种读法，而且**一个候选在全部输入上必须符合同一种读法**，不能逐个输入挑能通过的读法。两者符合不同读法，记为一个发现（规格歧义），由任务契约的负责人决定采用哪种标准，不由执行方按哪个实现能通过来反推。

## 交叉熵

| 编号 | 情形 | 读法 | 默认 |
| --- | --- | --- | --- |
| CE-A1 | 类别下标目标 + label_smoothing + class weight：平滑出去的那部分概率质量乘哪个权重 | R_A：每个类别用自己的权重 w_c（与概率目标公式 D3 一致）；R_B：整行乘目标类别的权重 w_{y_n}（与 D1 一致）；无权重时两者相同 | **决定：R_A 为主解释**（由 D3 与平滑定义组合而来，依据强度记为「组合解释」）；R_B 保留为对照 |
| CE-A2 | 上述情形下 'mean' 的分母 | D2：Σ_n w_{y_n}·1{未忽略}；R_C：Σ_n Σ_c w_c q_{n,c}·1{未忽略} | **决定：D2**（文档按目标类型明确区分了分母；R_C 是另一种合理的规范化，不是 D2 唯一的一致延伸）。R_C 只作诊断标签；A/D2 与 A/DC 之差是「声明解释下的差异」，不是 bug |
| CE-A3 | 'mean' 且 D2 分母为零（全部忽略，或出现的目标类别权重全为 0） | 有限实数目标无定义。注意不总是 0/0：有平滑时分子可以为正（权重 (0,1)、目标 0、eps=1/2 时为 ln2/4 对 0）。规格返回 loss = NaN、grad_defined = False；候选的返回值记为约定，不判对错；none 与 sum 在全部忽略时仍有定义（为零），不与 mean 一并列为无定义 | 无定义 |
| CE-A4 | ignore_index 落在 [0, C) 内 | 文档允许（"may not necessarily be in the class range"）；规格把该类别下标视为忽略 | — |
| CE-A5 | 概率目标 + class weight，reduction='mean' | 文档 D4 的分母是 N，而不是权重和；与类别下标目标的分母不同，这是文档明文，不是歧义，但最容易被实现成同一种，单列 | N |
| CE-A6 | label_smoothing 的标量取值 | 发布值（如 1/10）与 kernel 实际接收值分开评估，差异记为接口项 | **决定：主 API 比较用实际接收值**；接口项单列，不记为 kernel 的任务错误 |

## 池化

| 编号 | 情形 | 读法 | 默认 |
| --- | --- | --- | --- |
| POOL-A0 | 3d 是否与 1d/2d 同规则 | AvgPool1d 与 AvgPool3d 的文档各自给出公式、padding、ceil 窗口起点与逐维修正，基础规则都有依据；唯一的张力是 AvgPool3d 的 divisor_override 说明把默认除数称为 kernel_size，与 POOL-A1 的「pooling region」解释不一致 | 已有依据；只记录该处张力 |
| POOL-A1 | count_include_pad=True，ceil_mode 产生的窗口越过右侧 padding（越界部分） | R2：除数 = 窗口与显式 padding 域 [−p, L+p) 的交集大小；R1：核大小之积。整个窗口落在 padding 域内时两者相同 | **决定：R2 为主**；R1 保留为尾部读法对照。区分二者的例子：输入 [1,2,3,4]、k=3、s=2、p=0、ceil，第二窗口覆盖 2,3,4，R2 得 7/2、R1 得 7/3。只因过界分母不同，不自动记为已确认 bug |
| POOL-A2 | max pool 并列最大值时返回哪个下标、梯度给谁 | 规格返回并列集合 | **决定**：前向下标必须属于并列集合，不强制 first/last；严格的普通导数评分只在唯一最大者处进行；并列处用集合式检查（check_max_subgradient：系数在集合外为 0、集合内非负、和为 1）；单成员路由与平均拆分不互判错误；任务另外声明「反向使用返回下标」时才逐次检查选路 |
| POOL-A3 | max 窗口内有 NaN | R_prop：输出 NaN；R_ignore：跳过 NaN | **决定：R_prop 为项目诊断 profile**，只传播真正参与当前窗口的 NaN；R_ignore 单列，不因「不传播」自动报错；文档未规定的 NaN 行为不进入主 bug 正确率 |
| POOL-A4 | 文档不一致 | MaxPool2d 的精确公式里没有 dilation，但 dilation 是参数；规格按位置 s·o − p + d·m | — |
| POOL-A5 | 几何空窗口 | 有 dilation 或较大 padding 时，合法的窗口可能只采到 padding 位置。按 P-D3 的 −∞ padding，输出为 −∞，并列集合为空，没有实际输入可路由梯度；不要求返回下标属于空集合。R_ignore 过滤后无值的窗口是另一回事，规格对它返回「未建立」 | −∞ |

## index 与 scatter

| 编号 | 情形 | 读法 | 默认 |
| --- | --- | --- | --- |
| IDX-A1 | reduce='mean'，include_self=False | mean 没有单位元；规格取「只对贡献值求平均」 | 唯一读法 |
| IDX-A2 | reduce='mean'，include_self=True | 文档说 self 的值参与归约，在「对参与值求算术平均」的语义下只有一种读法：(self + Σ)/(1 + k)。(self + Σ)/k 不是 mean（常量输入不保持），v0.2 起不再列为读法，只作为带标签的错误变体 `mean_divides_by_contrib_count` 用于分类 | 唯一读法 |
| IDX-A3 | amax/amin 遇到 NaN | R_prop / R_ignore | **决定：R_prop 为诊断 profile**，只作用于实际参与归约的值（include_self=False 时旧 self 不参与；未被更新的位置不变）；R_ignore 过滤后无值时规格返回「未建立」 |
| IDX-A4 | amax/amin 的反向在并列时 | 规格返回最大者集合 | **决定**：不按唯一向量裁决；严格导数评分只在唯一极值处；并列处用 check_extremum_subgradient（支持集在极值集合内、系数非负、和等于上游）；API 有更具体条款时采用更具体条款 |
| IDX-A5 | 整数类型的 mean（向下取整还是向零取整） | 第一阶段不覆盖，只测浮点 | — |
| IDX-A6 | 文档不一致 | index_reduce_ 的旧版文档把 index 描述为 "indices of source to select from"，现行稳定版为 "indices of self to accumulate into"；规格按后者 | — |

## 梯度累加窗口（spec_accumulation.py）

| 编号 | 情形 | 读法 | 默认 |
| --- | --- | --- | --- |
| ACC-A1 | 整个累加窗口没有有效 token | token 平均目标无定义，记录 skip/zero/NaN 约定 | 无定义；个别 micro-batch 为空而窗口总 token 数 > 0 时目标仍有定义，空 micro-batch 贡献 0 |
| ACC-E1 | 错误变体：对各 micro-batch 的均值再取平均 | 不是读法，只作为带标签的错误变体；当各 micro-batch 的 token 数相等时与正确值相同，所以多数测试发现不了 | — |
