# Qwen3 common-evaluator selected-state smoke findings v0.1

## Verdict

`VALID SELECTED-STATE T1a/T1b CONSTRUCTION SMOKE; NOT A POPULATION BIAS RESULT`

已有 B/step-29 matched state 的 eager 与 compiled parameter-update artifacts 都能从同一个 frozen
pre-state **逐参数哈希精确**重建记录的 post-state。两个 post-states 随后由同一套 eager、FP16
autocast、SDPA-MATH evaluator 在冻结的 64 个 heldout arithmetic correct-completion examples 上评分。

## Frozen measurement

- endpoint：`T1b = mean correct-completion NLL(candidate post) - mean NLL(reference post)`；
- examples：`forkcert_builtin_arithmetic[9008:9072]`；
- aggregation：先对每个 example 的 completion tokens 平均，再对 64 个 examples 等权平均；
- repeats：每个 arm 两次 fresh model reconstruction；
- common ruler：两边都用 eager evaluator；它是共同量尺，不是正确性真值。

T1b manifest：`QWEN3_COMMON_EVALUATOR_SMOKE_MANIFEST_V0_1.json`。

T1a 另使用 baseline pre-state 生成 frozen rollout bank。sampling 参数全部显式冻结，不继承模型
目录 defaults；同一最终 manifest 下两个 fresh-process banks 的完整 content digest 都是
`997e0a6c...782d41`。32 个 completions 中 21 个得到 exact reward，8 个 prompt groups 中 6 个具有
非零 advantage；2 个 tied groups 原样保留为零 contribution，没有为制造信号而重采样。

## Construction evidence

| gate | result |
|---|---|
| snapshot/result/update artifact SHA256 | exact |
| eager `pre + update = recorded post` | exact parameter and buffer digests |
| compiled `pre + update = recorded post` | exact parameter and buffer digests |
| tokenizer/encoded bank identity | recorded; bank digest `4776eb69...d03a9` |
| eager-post fresh repeats | token-level result digest exact |
| compiled-post fresh repeats | token-level result digest exact |
| output validity | `VALID_SELECTED_STATE_T1B_SMOKE` |
| T1a fresh bank generation repeats | complete content digest exact |
| T1a informative groups | 6/8; tied groups retained |
| T1a reference/candidate scorer repeats | token-level result digest exact |
| T1a output validity | `VALID_SELECTED_STATE_T1A_SMOKE` |

Result：`results/oracle_calibration/qwen3_common_evaluator_smoke_v0_1/t1b_evaluation.json`。

T1a result：`results/oracle_calibration/qwen3_t1a_selected_state_smoke_v0_1/t1a_evaluation.json`。

## Selected-state observation

该状态的 candidate-minus-reference T1b 是 `+0.0002453755587339401`，两次重复完全相同。正号表示
compiled-induced next state 在这套冻结答案上给出的正确 completion likelihood 略低。

这证明了一件构造层面的事：implementation-induced update difference 可以被共同 evaluator 转换成
一个方向明确、可重复的 task-transition scalar。它**不证明**该方向在目标 state distribution 上稳定，
因此不能把这个数叫 `B`，更不能叫 compiler correctness bias。

同一状态的 T1a candidate-minus-reference GRPO surrogate loss 是
`+0.000018542632460594177`，两次 post-state reconstruction/scoring 完全一致。它说明 candidate
next state 在这份 baseline-anchored frozen rollout bank 上有一个可测的 surrogate shift。T1a 与
T1b 在该状态恰好同号，但两者量纲、bank 和含义不同，不能相加，也不能用这一状态的一致性替代
跨轨迹总体估计。

## What changed in the next-step assessment

T1a/T1b common evaluator 都已从“只有规则”推进到“一个真实 Qwen3 selected state 上可执行且
重复精确”。
因此当前主要缺口缩小为：

1. 300-step source trajectory 的 non-mutating multi-state capture 尚未实现；
2. 尚无一条完整 calibration trajectory 的端到端验证；
3. 尚无 4-trajectory calibration，更无 independent confirmation。

arm/pair 两层 record gate 已用该真实 state 的 4 条 arm records 和 2 条 paired-effect records 通过；
bundle 仍明确 `population_eligible=false`。下一步仍不是 operator sweep，而是把同一套 U/T
measurement 接到一条
完整 source trajectory 的预选 states 上。

## Nonclaims

- one selected state does not estimate population average shift；
- T1b only covers the declared arithmetic correct-answer bank；
- a nonzero state effect is not long-run harm；
- common eager evaluation does not make eager truth；
- this result does not identify a bias-contributing operator。
