# Bias Oracle 已知案例恢复审计

## 结论

当前低成本路径不是一个已经完成的通用 oracle。六个严格 formation positive 中，只有 4/6 能被当前实际接通的筛选路径直接报出风险；Liger 与 Phi 只能 fail-closed 升级，不能算作指标直接命中。

八个边界中共有 6/8 得到直接风险/边界判定；false-safe 为 0。自动升级使严格 positive 的 routed recall 为 100.0%，但这只能证明流程安全，不能证明指标完整。

## 逐例结果

| case | 冻结目标 | 当前预测 | 直接命中 | fail-closed |
|---|---|---|---:|---:|
| `liger_fused_ce` | STRICT_POSITIVE | `ESCALATE_MISSING_EVENT_MOMENT` | no | yes |
| `phi4_seq64_lmhead_dx` | STRICT_POSITIVE | `ESCALATE_MISSING_TRANSPORT_JOINT_MOMENT` | no | yes |
| `qwen64_vproj_mm` | STRICT_POSITIVE | `DIRECT_LOCAL_RISK_DOWNSTREAM_ESCALATE` | yes | yes |
| `qwen128_vproj_mm` | STRICT_POSITIVE | `DIRECT_LOCAL_RISK_DOWNSTREAM_ESCALATE` | yes | yes |
| `qwen_saved_p_seq128` | STRICT_POSITIVE | `DIRECT_RISK_RESPONSE_RECTIFICATION` | yes | yes |
| `qwen3vl_silu_layer0` | STRICT_POSITIVE | `DIRECT_RISK_RESPONSE_RECTIFICATION` | yes | yes |
| `mamba_seq64_input_proj` | PARTIAL_POSITIVE | `DIRECT_LOCAL_RISK_DOWNSTREAM_ESCALATE` | yes | yes |
| `qwen_layer23_attention_state` | ABSTAIN_BOUNDARY | `ABSTAIN_SOURCE_FIDELITY_FAILED` | yes | yes |

## 指标缺口

- Qwen64/128：四次 fixed-condition repeat 已能直接发现 local source risk，但下游层仍需顺序追加 repeat。
- saved-P/SiLU：exact antithetic gradient pair 的 Adam response-even screen 可直接命中。
- Mamba：local risk 可直接命中；真实 gradient/SGD 的条件覆盖仍不完整，所以必须保留 partial。
- layer-23：natural-source fidelity 未通过，正确输出是 abstain，而不是 risk 或 safe。
- Liger：缺的是由声明 schedule/operands 自动生成的 event-antithetic moment，旧 24/24 机制结果不能回填为新预测。
- Phi：缺的是联合残差－transport moment；只计算 `J E[epsilon]` 会漏掉 `E[J_e epsilon_e]` 中的 covariance 通道。

因此需要重新思考的是低成本指标的**输入与覆盖**，不是推翻条件反对称分解。下一版 screen 至少必须显式估计：

`E[J_e epsilon_e | c] = E[J_e|c] E[epsilon_e|c] + Cov(J_e, epsilon_e | c)`。

若仍只有 source mean 与局部 HVP，它会系统漏掉 Phi 型 pairing bias；若不能从 schedule 自动构造 event orbit，也会漏掉 Liger。

## 边界

这是已参与假设形成的 development recovery audit，不是 held-out accuracy。coded-group 与 shared-HVP 目前只有 synthetic/semantic-cut feasibility，尚未在这八个自然案例上运行，因此没有计作任何 case 的直接恢复。
