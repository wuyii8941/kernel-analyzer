# Sample completion v1

> 历史协议快照，不是当前案例数量。当前长程审计见
> `docs/all_bias_long_horizon_audit.md`：23 个唯一主矩阵 ID、11 个历史候选、26 行合并审计、4 个最终案例。

这份协议用于补足案例数量，不能把旧实验重新包装成新结果。

## 目标

统一完成至少 20 个案例，其中至少 15 个必须是“确实有数值差异、而且差异能到达参数梯度”的控制。正例至少来自 3 个独立实现家族；如果自然正例不足，必须用真实 kernel 的可控改动证明一个结构条件确实能打开或关闭持续性。

## 每个案例都测同一条链

```text
算子输出差异
    -> 参数梯度差异
    -> SGD/AdamW 更新差异
    -> 32 步参数轨迹分离
```

还要记录当前算子造成的差异和过去两条训练轨迹已经分开后造成的反馈，不能把二者混在一起。

## 成本分层

1. 2 步：确认 candidate、repair、参数坐标和梯度可达。
2. 8 步：去掉完全为零或不可达的候选。
3. 16 步：只做预测，不能给最终标签。
4. 32 步：最终标签；另外随机抽 8–10 个未被 16 步标记的候选，估计漏检。

## 当前冻结名册

完整机器可读名册见 [`roster.json`](../results/property/sample_completion_v1/roster.json)。其中 8 个是已有案例，16 个按 loss、normalization、attention 和 state-space 训练瓶颈预先选出的搜索单元。名册中的 `NOT_YET_UNIFIED` 和 `NOT_STARTED` 是当前状态，不是负例。

## Oracle 规则

Oracle 只能使用前 16 步的算子、梯度和更新差异，以及相邻步相关性。它不能读取 32 步标签、最终漂移、旧的 T4/SEUP 结论或案例名称。输出只有：

- `ESCALATE_TO_32_STEP`
- `NO_ESCALATION_UNDER_PROTOCOL`
- `ABSTAIN_MISSING_REQUIRED_MEASUREMENT`

不能输出 `SAFE`。

## 当前不是完成状态

已有四个模型的 12 个覆盖单元和三个 headline case，但它们还不等于本协议要求的 20 个统一案例。当前补全状态在 [`existing_evidence_snapshot.json`](../results/property/sample_completion_v1/existing_evidence_snapshot.json) 中明确记录为零个统一完成案例、零个统一控制；这避免把旧的不同协议数据混作新分母。

## 停止条件

满足统一案例数、有效控制数、正例家族/真实 kernel 因果构造、冻结 held-out、基线比较和成本报告后才可以结束这一轮。若没有第三个自然正例，结果应诚实收窄为已测范围内的家族性规律，而不是宣布全算子通用。
