# Bias Oracle 已知案例恢复审计

## 结论

更新后的多通道 screen 在当前 development roster 的六个严格 formation positive 中，全部 6/6 能被当前实际接通的筛选路径直接报出风险；Liger 由 reference-relative moving frame 命中，Phi 由四状态 complete-Gram coherence 命中。

八个边界中共有 8/8 得到直接风险/边界判定；false-safe 为 0。自动升级使严格 positive 的 routed recall 为 100.0%，但这只能证明流程安全，不能证明指标完整。

## 逐例结果

| case | 冻结目标 | 当前预测 | 直接命中 | fail-closed |
|---|---|---|---:|---:|
| `liger_fused_ce` | STRICT_POSITIVE | `DIRECT_RISK_REFERENCE_RELATIVE` | yes | yes |
| `phi4_seq64_lmhead_dx` | STRICT_POSITIVE | `DIRECT_RISK_POPULATION_COHERENCE` | yes | yes |
| `qwen64_vproj_mm` | STRICT_POSITIVE | `DIRECT_LOCAL_RISK_DOWNSTREAM_ESCALATE` | yes | yes |
| `qwen128_vproj_mm` | STRICT_POSITIVE | `DIRECT_RISK_SOURCE` | yes | yes |
| `qwen_saved_p_seq128` | STRICT_POSITIVE | `DIRECT_RISK_RESPONSE_RECTIFICATION` | yes | yes |
| `qwen3vl_silu_layer0` | STRICT_POSITIVE | `DIRECT_RISK_RESPONSE_RECTIFICATION` | yes | yes |
| `mamba_seq64_input_proj` | PARTIAL_POSITIVE | `DIRECT_LOCAL_RISK_DOWNSTREAM_ESCALATE` | yes | yes |
| `qwen_layer23_attention_state` | ABSTAIN_BOUNDARY | `ABSTAIN_SOURCE_FIDELITY_FAILED` | yes | yes |

## 指标结构

- Qwen64/128：四次 fixed-condition repeat 已能直接发现 local source risk，但下游层仍需顺序追加 repeat。
- saved-P/SiLU：exact antithetic gradient pair 的 Adam response-even screen 可直接命中。
- Mamba：local risk 可直接命中；真实 gradient/SGD 的条件覆盖仍不完整，所以必须保留 partial。
- layer-23：natural-source fidelity 未通过，正确输出是 abstain，而不是 risk 或 safe。
- Liger：chunk atoms 彼此并不相干；真正可比较的是每个 state 内误差相对 FP32-accumulator reference update 的乘性系数。
- Phi：reference-relative 系数会变号，但四状态 complete-vector Gram 已直接暴露共同参数坐标分量。

因此合适的候选不是一个标量，而是三个互补、均不使用 trajectory label 的风险证据族：

1. conditional event/source asymmetry；
2. transported directional component（complete-vector population、same-state moving frame 或冻结 cross-fit projection）；
3. exact antithetic response non-oddness。

任一 witness 命中即可报风险；全部未命中只能升级或 abstain，不能签发 safe。

## 二级回归

未参与当前六案例指标拟合的旧 Qwen lm_head confirmation 被 cross-fit witness 命中，而 Liger RMSNorm dX 的真实 sign-changing control 未被命中。它们仍是回顾性证据，不能替代下一轮预先冻结的 prospective held-out。

## 边界

这是已参与假设形成的 development recovery audit，不是 held-out accuracy。六个严格 positives 全部命中只说明候选指标值得进入冻结 held-out；不能据此反调阈值或声称通用。
