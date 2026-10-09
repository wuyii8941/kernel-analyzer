# 已采纳的审阅决定（2026-10-07）与规格的对应修改

来源：`decisions_reviewed.md`（审阅方，v0.3.1 的审阅）。本文件只记录采纳情况与代码位置；原文随包保留。

| 编号 | 决定 | 规格中的体现 |
| --- | --- | --- |
| CE-A1 | R_A 为主解释；R_B 保留为对照 | `cross_entropy(reading="R_A")` 为默认；R_B、R_C 可选 |
| CE-A2 | 分母用 D2，不切换到 R_C | reading="R_A" 即 D2 分母；R_C 只作诊断标签 |
| CE-A3 | 分母为零时有限实数目标无定义，不写成 0/0；none/sum 仍有定义 | 文档字符串与测试 `ce/A3/positive-over-zero-undefined` |
| CE-A5 | 概率目标 mean 除以 N | 已按 D4 实现 |
| CE-A6 | 主比较用实际接收值；接口项单列 | 协议第 6 节 |
| POOL-A0 | 1d/3d 都有文档依据；只记录 AvgPool3d 的 divisor_override 文案张力 | ambiguities.md |
| POOL-A1 | R2 为主，R1 为尾部对照 | `avg_pool`/`avg_pool_backward` 默认 reading="R2"；审阅例子 7/2 对 7/3 入测试 |
| POOL-A2 | 前向下标须在并列集合内；严格导数评分限唯一最大者；并列处集合式检查；单成员与拆分不互判 | `check_max_indices`、`check_max_subgradient`（非重叠窗口） |
| POOL-A3 | R_prop 为诊断 profile，只传播参与的 NaN；R_ignore 单列 | `max_pool(nan_reading=...)`；过滤后无值 → SpecNotEstablished |
| POOL-A5 | 几何空窗口 → −∞、空集合；与 NaN 过滤后无值分开 | 两条路径分开实现 |
| IDX-A3 | 同 POOL-A3；只作用于实际参与的值 | `_reduce`；全 NaN 过滤后 → SpecNotEstablished |
| IDX-A4 | 不按唯一向量裁决；严格评分限唯一极值；集合式检查 | `check_extremum_subgradient` |
| ACC-A1 | 窗口无 token 无定义；空 micro-batch 贡献 0 | `window_loss` |

审阅方的两点说明一并采纳：独立核对是两种算术实现之间的抽样复现，不是第三方严格区间证明；R_A、R_B、R_C 都能算对各自的定义，核对通过不能决定哪一份是 API 契约。
