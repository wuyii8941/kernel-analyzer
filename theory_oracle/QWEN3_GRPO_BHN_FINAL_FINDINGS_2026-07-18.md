# Qwen3 GRPO B/H/N Oracle：A/B/C 联合发现

状态：`COMPLETE_FOR_MATCHED_SCORER_PROFILE`，但 training-step transition 与 correctness 仍未完成。

## 1. 冻结查询

- reference：grad-enabled eager Trainer scorer；
- candidate：grad-enabled tracked Inductor scorer；
- observable：每个 rollout state 上对齐的 `4 x 128` current-token log-probability field；
- state distribution：预先抽取的 A/B/C 三条 fixed-start 轨迹，共 30 个 state clusters；
- randomness/coupling：对应 eager/compiled call 使用相同 state tensors 并恢复 RNG；每个实现同状态调用两次。

因此以下 B 是 implementation-relative mean-effect magnitude，不是相对真值的 numerical bias。

## 2. 连续 discrepancy profile

| profile | A | B | C | 合并 |
|---|---:|---:|---:|---:|
| state clusters | 10 | 10 | 10 | 30 |
| sign-safe B norm | 1.111e-3 | 1.046e-3 | 1.058e-3 | 6.202e-4 |
| signed global mean | 1.126e-5 | 2.202e-5 | 2.290e-5 | 1.873e-5 |
| repeat-corrected H | 1.401e-5 | 1.068e-5 | 1.066e-5 | 1.176e-5 |
| paired N | 0 | 0 | 0 | 0 |

主要观察：

1. 三条独立轨迹的单轨迹 B magnitude 与 H 处在相近量级，说明该结构不是只由轨迹 A 产生。
2. 合并 B 明显小于每条轨迹 B，而 signed state means 在每条轨迹内都正负变化。这说明 effect field 在 state/trajectory 间存在方向抵消；B 不是实现自身的无条件常数。
3. signed global mean 比 sign-safe B norm 小约两个数量级。只用全局 signed mean 会漏掉稳定存在但在 tensor elements/states 间变号的 effect structure。
4. 在当前固定执行与 RNG coupling protocol 下，全部同状态 repeats 完全一致，所以观测 N 为零。这个结论不外推到其他硬件、autotuning、atomic、stochastic rounding 或未冻结 algorithmic RNG 的协议。
5. B 的 state-cluster 近似区间下界仍为零；30 个 states/3 个 run clusters 不足以作精确总体 prevalence 声明。

## 3. 独立 semantic-event ledger

在 12,288 个 nonzero-advantage tokens 中，检测到 16 个 repeat-stable clipping disagreements：`0->1` 为 6 个，`1->0` 为 10 个。

- finite-bank disagreement：`16 / 12288 = 0.0013021`；
- finite-bank directional difference：`(6 - 10) / 12288 = -0.0003255`；
- A/B/C 均观察到事件，故预声明的 finite-bank event replication 得到支持；
- 由于只有 3 个 run clusters，population prevalence 仍为 `INDETERMINATE`；
- finite-bank `REJECT` 只表示这一个声明过的有限 bank 并非 event-identical，不表示 compiled incorrect。

这组数据同时说明：B/H/N profile 与 semantic-event ledger 必须并列。B/H/N 描述连续 discrepancy 的结构；event ledger 描述它是否跨过特定应用边界。二者不能互相替代。

## 4. 对 Claude 四层方案的裁决

可以保留：

- measurement -> B/H/N/U profile -> propagation -> verdict 的分层；
- matched inputs/states 和 repeated executions；
- operator/observable 与完整 training-step endpoint 分开；
- 没有边界时返回 `UNINSTANTIATED`。

必须修改：

- B 不能只是把 states 与 tensor elements 全部做 signed average，否则会抵消；应同时保存 mean-effect field、方向 summary 和非抵消 magnitude。
- H 必须扣除 repeat noise；N 必须说明是 ref、candidate 还是 paired-difference variability，并声明 coupling protocol。
- sampling uncertainty U 不能混进 H 或 N。
- “零均值 noise 可被 SGD 吸收”和“非零 B 会跨步累积”不是 B/H/N 定义推出的结论。
- `ACCEPT` 需要预声明容忍界限及其置信上界；点估计低于阈值不够。
- 自由运行两条训练轨迹不能估计当前 state 的 implementation effect，因为状态已经分叉。

## 5. Training-level Oracle 尚缺的部分

Run B 的预声明 step-29 snapshot 已通过独立审计：模型、optimizer、scheduler、scaler、RNG、目标 minibatch 和 compiler history 均完整。下一关是在完全相同的起始 state 上分别运行 eager/candidate intervention，测量 loss、gradient 和 parameter update 的 B/H/N/U。

只有这一步才能回答 scorer discrepancy 是否传播为 update discrepancy。即使传播成立，也仍不能直接推出长期不收敛或精度下降；长期训练应是独立 validation ledger。

## 6. 可追溯性说明

冻结 manifest 中两份设计文档的原路径后来被移动到 `theory_oracle/archive/`。归档文件 SHA-256 与 manifest 中冻结值完全一致，但旧路径已失效。这是路径级 provenance 缺陷，不改变数据内容；后续 manifest 应记录 archive relocation，而不是静默覆盖冻结记录。

## 7. 结果文件

- 连续 profile：`results/training_step_oracle/qwen3_grpo_heldout_transport_v0_1/bhn_v0_2.json`
- semantic ledger：`results/training_step_oracle/qwen3_grpo_heldout_transport_v0_1/evaluation.json`
- transition snapshot audit：`results/training_step_oracle/qwen3_grpo_heldout_transport_v0_1/b_transition_snapshot_audit.json`
