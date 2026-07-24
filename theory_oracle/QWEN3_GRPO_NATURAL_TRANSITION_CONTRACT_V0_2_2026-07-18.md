# Qwen3 GRPO natural transition Oracle contract v0.2

状态：`FROZEN_BEFORE_TRANSITION_EFFECT_EXECUTION`

## 查询

从预先选择并验证的 run-B optimizer-step-29 pre-minibatch state 出发，仅把 eager scorer forward 替换为具有相同捕获 specialization history 的 tracked Inductor forward，是否改变同一个自然 GRPO 训练步的连续结果、离散事件或下一状态？

这不是 correctness 查询，也不是长期训练查询。

## 实验单位与重复

- 唯一 state：`data/qwen3_grpo_heldout_transport_v01_b_step29_transition`；
- reference：eager Accelerate FP16 scorer forward；
- candidate：在 final snapshot 上按捕获 inputs 顺序构建的 fresh tracked-Inductor realization；
- candidate history scope：要求有序唯一 graph family 与捕获记录一致，并要求目标 scorer tensor bitwise 一致；当前 snapshot 没有保存十份历史模型/optimizer states，因此不能声称重建了原训练的十次 compile-event lineage；
- common suffix：GRPO loss、scaled backward、unscale、global gradient clipping、captured AdamW step、GradScaler update、linear scheduler step；
- A1/A2/B1/B2 分别在 fresh process 中从同一 snapshot 恢复；
- 每个 arm 两次执行只识别该 selected state 上的 replay/runtime variability，不是两个训练 states。

## 必须通过的有效性门

1. snapshot verifier 为 `VALID`；
2. 四个 arms 的 model/buffer/optimizer/scheduler/scaler/RNG 和 target minibatch 起点完全相同；
3. eager scorer tensor 精确匹配捕获的 `ref_first_sha256`；
4. candidate scorer tensor精确匹配捕获的 `alt_first_sha256`；
5. candidate 的有序唯一 graph hashes/node counts family 精确匹配捕获记录；历史 compile-event multiplicity 明确不在可识别范围内；
6. history replay不改变参数、buffer、optimizer、scheduler、scaler或 RNG；
7. candidate measured call 确实调用 compiled runtime，且 measurement 不产生新 specialization；
8. compiled wrapper 与 optimizer 使用同一组 parameter objects；
9. gradient/update 均 finite。

任一失败均返回 `INVALID`，不能解释为零 effect。

## 连续 endpoints

- 512 维 current-token log-probability field；
- scalar GRPO loss；
- scaled、unscaled、clipped gradient summaries；
- 完整 clipped-gradient effect vector；
- 完整 parameter-update effect vector；
- per-parameter L2、max-coordinate 与 direction/alignment；
- post-step model/buffer、optimizer moments、scheduler、scaler canonical identity。

每个 endpoint 的 profile 均区分：

- B：两个 paired repeats 的 candidate-minus-reference mean effect；
- N：同一 selected state 上两个 paired effects 的 repeat variance；
- H：一个 state 无法识别，必须报告 `UNIDENTIFIABLE_ONE_STATE`；
- U：本设计不作 state-population inference，必须报告 `NOT_ESTIMATED_SELECTED_STATE`。

不得把参数/gradient tensor elements 当作独立 state samples。

## 语义 endpoints

- per-token clipping decisions；
- global gradient-clipping trigger；
- AMP overflow/step skip；
- scaler/scheduler discrete transition；
- non-finite gradient/update。

## Verdict ledgers

- construction：`VALID/INVALID`；
- selected-state exact transition compatibility：
  - self repeats exact且 A/B next state 不同：`REJECT_EXACT_SELECTED_STATE`；
  - self repeats exact且 A/B next state相同：`ACCEPT_EXACT_SELECTED_STATE`；
  - self repeats不稳定：`INDETERMINATE_RUNTIME_VARIABILITY`；
- population prevalence：`NOT_ESTIMATED_ONE_SELECTED_STATE`；
- correctness：`UNINSTANTIATED`；
- operator attribution：`NOT_CLAIMED_SCORER_FORWARD_TREATMENT`；
- long-run harm：`NOT_CLAIMED`。

`REJECT_EXACT_SELECTED_STATE` 只表示这一个受控 transition 并非实现相容，不表示 compiler wrong-code 或实际训练有害。

## Kill criteria

- 不能复现捕获 scorer tensor 或 graph history；
- 只恢复 model 而没有历史 optimizer/scaler/RNG；
- 候选 history warm-up 改变训练状态；
- 用假想 SGD update 替代捕获 AdamW transition；
- 从一个 selected state 估计 H 或 population prevalence；
- 用 next-state hash difference 代替 effect magnitude/direction；
- 把 scorer-forward treatment 称为 operator causal effect；
- 从一步 update discrepancy 推断不收敛、精度下降或训练变长。
