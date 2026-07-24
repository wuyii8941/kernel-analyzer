# Qwen3 transition realization contract v0.1

状态：`FROZEN_PROTOCOL_BEFORE_CALIBRATION_ENDPOINT_EXECUTION`

## 为什么需要这一层

calibration 的 source trajectory 是纯 eager 轨迹。它冻结了 matched state，
但没有、也不应假装拥有一个训练时已经存在的 compiled anchor。因此候选实现
不能由“这一次 fresh `torch.compile` 恰好产生了什么”临时定义。

对每个冻结 state，先做一次 **prospective realization instantiation**：

1. 从完整 pre-step snapshot 恢复模型、optimizer、scheduler、scaler、RNG 和输入历史；
2. 按冻结的 compiler protocol 回放 scorer shape/history；
3. 记录 ordered unique graph family、reference/candidate scorer tensor identity；
4. 验证该过程没有改变训练状态；
5. 在执行 loss、backward、clipping、AdamW 或任何 U/T endpoint 之前写出合同。

后续两个 arm、两个 repeat 必须匹配同一合同。未匹配返回 `INVALID`，不能解释为
零 effect 或 runtime noise。

## 这个合同定义什么

它定义的是：

> 在该 state 和已声明 history 下，由固定 PyTorch/CUDA/Inductor/attention/
> mixed-precision protocol 产生并由 graph family 固定身份的 candidate realization。

合同至少包含：

- snapshot metadata 与 target minibatch digest；
- compiler protocol digest；
- ordered unique graph family 及其 digest；
- eager/candidate scorer tensor digest；
- history replay 和完整 preflight 的 state-preservation gate；
- measured call 未触发新 specialization 的证据。

## 它不定义什么

- 不把首次 realization 选成“差异最大”或“最稳定”的实现；
- 不把 scorer discrepancy 称为 B、correctness error 或训练伤害；
- 不保证不同 states 有相同 graph family；state-specific specialization 是 treatment
  的可记录组成，而不是自动可池化的同一 kernel；
- 不提供 operator attribution；
- 不替代 same-state fresh-process repeats。

## 与 B/H/N 的关系

- B/H 比较的是同一预声明 protocol 在目标 state distribution 上诱导的 paired effect；
- N 比较的是合同固定后 fresh-process paired effects 的重复变化；
- 如果合同不能稳定复现，结果是 realization instability / `INVALID`，不能把 treatment
  变化混入 N；
- 若 graph family 随 state 系统变化，该变化可作为 H 的解释变量，但不能事后改写
  state population。

## 实现入口

`qwen3_grpo_natural_transition_v0_2.py` 支持：

- `--instantiate-realization-contract PATH`：只实例化合同并在 transition endpoint 前退出；
- `--realization-contract PATH`：正常 arm 必须匹配合同；
- 旧 selected-state 实验仍可通过 `--anchor-states` 复现，但两种 identity source 不能混用。

