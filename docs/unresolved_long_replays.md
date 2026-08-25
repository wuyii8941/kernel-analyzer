# 长程复核未决项

下面的记录没有被判为“没有 bias”。它们只是没有得到一份可以审计的完整 4096 步结果，因此不进入阳性或阴性分母。

| 记录 | 已完成到哪里 | 未决原因 |
|---|---:|---|
| Gemma 4 RMSNorm/投影反馈区域 | 第 294 步 | 兼容运行包的逐次身份检查在重放中断，不能把部分运行当完整结论 |
| DeepSeek layer-35 `dV` | 没有合法的完整 candidate/repair 长程对照 | 只有形成阶段材料，缺少可重放的精确修复边界 |
| Mamba `in_proj` 的 live candidate/repair 后果阶段 | 4096 步直接审计已完成；live 阶段未产出 | sequential Mamba runtime 未能在当前执行预算内安全写出配对 loss 文件；直接审计未超过自身随机基线 |

上表中的记录保留在主审计表中，并单独保存本文件对应的 JSON。它们不会被标成 negative，也不会被用来计算“没有 bias”的比例。Llama 和 Ministral 已有完整结果，已从未决项移出；它们是同一 `lm_head dX` 实现族在新模型上的长程复现。
