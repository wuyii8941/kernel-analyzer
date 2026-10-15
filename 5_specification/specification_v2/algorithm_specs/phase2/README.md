# 第二阶段收束包 v2.0（2026-10-08）— 覆盖第一档全部家族

内容分两部分。

## A 保证与裁决表（主件）
`guarantee_chain_table.md`：保证链四个箭头各自的前提、规则、代码位置（执行方填）、验证依据与降级处理；差异 / 缺陷 / 重要性三种结论分开；合法、未规定、无定义输入由版本化契约事先决定；用三类组合（纯计算、保存值/分支、状态更新）验证整条链。这是理论闭合的对象。

## A' 让执行方能够独立完成第 1–3 行的材料
| 文件 | 作用 |
| --- | --- |
| numeric_contract_v3.md | 三个问题分开；黑箱模式下精度不变性的可执行判据（含不适用与非有限的处理）；「无法判断」的三条触发规则（S₀、N₀、n_min 由执行方从校准表填入） |
| chain_table_template.json + validate_chain_table.py | 第 1 行的填写模板与自动校验：代码位置、测试、每条降级的触发记录都不能空，引用的文件与名称必须存在 |
| readjudication_procedure.md | 第 2 行：检索模式、复审表格式、改写规则、完成标准 |
| composition_checklist_template.md | 第 4 节三类组合各一份的清单与通过标准 |
| contract_v1_families.md | 第 3 行：7 个家族的契约；其余家族由执行方按文档起草为 v0（标「未审阅」），审阅方在两个工作日内审完入库 |

## B 规格（第一档 15 个家族全部有规格或性质判据）
| 文件 | 内容 | 严格性 |
| --- | --- | --- |
| rigorous.py / rigorous_mp.py | 区间算术：decimal 固定上下文版（含 sqrt）；mpmath.iv 版（exp/log/sqrt/cos/sin 严格；tanh 经 exp 严格；erf 为声明近似） | 见各自文件 |
| spec_optimizers.py | AdamW/Adam、SGD、RMSprop 的一步与短序列；grad=None 与零梯度；amsgrad 的版本相关读法 | 精确 + sqrt 区间 |
| spec_normalization.py | LayerNorm、RMSNorm、GroupNorm、BatchNorm | 精确 + sqrt 区间 |
| spec_embedding.py | embedding、embedding_bag | 精确 + 范数区间 |
| spec_base_ops.py | matmul/linear 及梯度、归约与扫描、softmax/log_softmax/logsumexp、激活（ReLU/SiLU/GELU 两式/SwiGLU/GeGLU）、gather/index_select 及伴随 | 精确；exp/log/tanh 严格；GELU-erf 声明近似 |
| spec_attention.py | SDPA 语义（mask、causal 左上/右下、GQA、softcap、滑动窗口）、序列打包的 mask 与位置、RoPE 两种布局 | 精确 + 严格区间 |
| spec_training_program.py | 学习率调度（cosine、linear、HF warmup+cosine）、clip_grad_norm_、GradScaler 状态机、标签右移与计数、跨 rank 平均、重算等价性质 | 精确 + cos/sqrt 严格区间 |
| spec_moe.py | 路由、top-k 集合、重归一、容量丢弃、合并、负载均衡损失、排列恒等性质 | 精确 + softmax 严格区间 |
| test_specs_phase2.py | 88 项手算检查，全部通过；失败非零退出 | — |
| contract_v2_all_families.md | 第一档全部家族的合法域、读法、无定义情形与规格文件对应 | — |
| ambiguities_phase2.md | 第二阶段的歧义与声明约定清单 | — |

独立性与边界：规格只依据官方文档或原论文的公式；没有查看候选实现的源码来确定语义（clip 的 1e-6 来自 clip_grads_with_norm_ 的文档公式）。标「文档待抓取核对」的条目见契约末尾，审阅时优先核。MoE 没有单一 PyTorch API，规格按各论文声明的公式分读法实现。

规格只依据官方文档书写（URL 与抓取日期在文件开头），没有查看任何实现的源码。歧义与版本相关读法写在各文件的 docstring（OPT-A1 等），统一清单待并入 ambiguities。
