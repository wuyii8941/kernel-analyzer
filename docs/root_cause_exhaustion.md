# 根因推导边界

本页回答一个严格问题：不新增观测、不补造缺失原始量时，现有记录还能否推出更强的
根因结论。机器复算入口是 `scripts/build_root_cause_exhaustion_audit.py`，结果为
`results/property/root_cause_closure_v1/exhaustion_audit.json`。

这是已检查材料的缺口表，**不是证明所有可能的离线分析均已穷尽**。
旧机器汇总记录当时的判断；本轮核对采集代码后发现 Liger 写入范围和 softmax 来源
表述过强，以下收窄解释优先。原始测量和旧汇总保留，不据此宣布全体根因闭合。

| 问题组 | 现有数据能够确定 | 再升级所缺的观测 |
|---|---|---|
| AdamW8bit moment 量化 | 保存残差的传播、干预与限定设置的独立训练收益 | 若声称均值 bias 是唯一中介，才需进一步区分均值、方差和结构；不是所有后续工作的前提 |
| Liger dW 累加 | FP32 顺序改变局部计算，并在长度64/256互斥确认集留下可复现方向结构 | 原坐标实际写入仍未保存；只改变加法顺序的1024步配对训练未显示可解释的长期 loss 后果 |
| softmax saved state | 保存统计与重构概率的行和一致性线索；1024 步声明轨迹已确认恢复会进入 gradient/write 并产生轨迹 non-identity | 独立状态总体中的自然 bias、持续方向和 material loss consequence |
| attention → q-proj 区域 | 一个延迟 BF16 物化的局部来源及传播路径 | 分别干预剩余 upstream logits 与 residual-stream 来源 |
| MM/GEMM | 已有四个具体位置完成同输入来源分解：Qwen128 为 output rounding，Qwen64/Mamba 为 kernel+output rounding，Phi 为 kernel arithmetic | 若要主张跨模型统一 MM 根因，才需要新的共享 factorial；当前不把不同位置合并成一个普遍 bias |
| SiLU backward | 在一个 AST 受检 gate-gradient endpoint 上，显式指数 source variant 贯穿 local、gradient、moment、update 和 write，且使用相同调用前输入 | 自然总体 mean bias，以及 sigmoid、表达式次序、最终 cast 的独立贡献；需要分量中间量和独立自然状态 |
| fused RoPE | 同输入差异与 optimizer-state 条件性 | 单独改变中间物化；固定 step counter 后改变 moments |

## 本轮新增的可用结论

Liger 的旧 joint Gram 可以恢复摘要中的 profile 投影。采集代码使用 8192 维摘要，
update 由 `adam_delta` 计算而没有实际写入；约 `3.08e-9` 是这个摘要空间中的比例，
不是原坐标实际写入 RMS。原 fixed-suite 数据也不因重新运行 Student 检验而变成
随机总体样本。因此保留局部/摘要方向证据，不据此签发实际写入等价或解释已有 loss。

softmax 的 114,688 行重构概率出现非零行和缺陷，重归一化把最大缺陷降到
`8e-16` 以下。这是两个固定状态的一致性诊断；重归一化本身按构造恢复行和，
没有单独验证真实 backward 修改、实际参数写入或自然 bias 的唯一根因。

## 什么情况下才需要新观测

Gram 可以恢复内积与投影，不能恢复从未保存的四种实现输出；总能量摘要不能恢复
SiLU 中间表达式；聚合后的 RoPE 比例不能把同时改变的 moments 和 step counter
拆开。继续给出唯一来源将依赖不可验证假设，不属于数学推导。

下一轮若继续，应逐项运行表中的最小区分实验。它们是新证据，不应回写成历史协议的
事前结论。

## Liger 顺序确认后的状态

此前“需要另采独立状态”的缺口已由长度64和长度256确认运行部分填补：各32个输入状态与长度128
开发银行完全互斥，前向 loss 和 hidden-state gradient 逐位相同，只有 dW 的64块
FP32 加法顺序不同。确认半区的预先声明方向为14/16同向，参数梯度 additive 与
residual 分支以及零矩 AdamW 首步的三个分支均确认。

长度256上的单变量后续干预还比较了 even-then-odd、冻结置换和 FP32 Kahan 外层补偿。
even-then-odd 使三个 update 方向区间均跨零，但总 update RMS 上升；Kahan 减少了
原始梯度向量差异，却没有减少 update RMS。随后1024步小型 FP32 配对训练的验证 loss
差异在 `2.4e-7` 量级并在末步回到零。因此当前证据支持“归约顺序控制有符号 update
分量”，不支持同精度顺序变体带来可测的短程质量改善。

这使 Liger 成为一个已完成“同精度实现差异 → 归约顺序来源 → 新输入 bias 复现”的
案例。它仍使用8192维摘要、只覆盖一个参数范围，且没有该顺序变化的长程 loss 结果；
因此不能与 AdamW8bit 的完整训练干预链合并描述。
