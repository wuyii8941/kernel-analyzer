# Qwen3 task-transition endpoint rationale v0.1

## Problem

update L2 或 reference-update alignment 只能描述 parameter-space geometry。若要接近“局部差异
是否形成宏观训练方向”，还需要从同一 pre-state 得到的 two next states 上评价具有共同任务
含义的 scalar functional。

单步 greedy reward 不适合作为唯一 functional：小 update 常不改变 token sequence，reward 大量
exact-tie；这只能说明离散输出未翻转，不能说明连续训练作用相同。

## Frozen T1 structure

T1 是两个不合并的 scalar endpoints。任何一个结果都保留自身 B/H/N/U 和 verdict。

### T1a：heldout GRPO surrogate shift

对每个 pre-state `S`，在 matched update 前生成并冻结一个不参与该 update 的 evaluation rollout
bank `E_S`。bank 使用预声明 baseline-anchored generation protocol，保存 prompt/completion、mask、
old log-probability、reward-derived advantage 和 RNG provenance。

generation 不继承模型目录里的 defaults；`do_sample=true, temperature=1, top_p=1, top_k=0,
repetition_penalty=1, use_cache=false, min_new_tokens=max_new_tokens=128` 均显式冻结。这样同一 T1a
不会因 Transformers/TRL 默认值或 checkpoint 自带 `generation_config.json` 改变含义。

两条 arm 完成 natural update 后，在相同 `E_S` 上只做 deterministic scoring：

`T1a(S) = L_GRPO(S'_C; E_S) - L_GRPO(S'_R; E_S)`。

它回答 candidate next state 相对 baseline next state，在独立 frozen GRPO examples 上的 surrogate
objective shift。它不复用 update minibatch，不重新 sampling，不声称 surrogate 等于长期 reward。

两条 post states 都加载到同一个 common eager FP16/SDPA-math evaluation realization 中计算 `F`；
不能各自沿用 training arm 的 eager/compiled scorer，否则 T1 会混入 evaluator implementation
discrepancy。common evaluator 是共同量尺，不是 correctness truth。

### T1b：heldout correct-answer log-likelihood shift

使用固定、外部、从未参与训练 update 的 arithmetic prompt/correct-completion bank `A`：

`T1b(S) = NLL_correct(S'_C; A) - NLL_correct(S'_R; A)`。

teacher-forced correct answer 与 aggregation rule 在 confirmation 前冻结。T1b 是该 arithmetic
subject 的 task proxy；它不是通用语言模型 quality，也不依赖单次 generated token 是否翻转。

## Why both are needed

- T1a 与训练 objective 接近，但 evaluation rollouts 由 baseline-anchored policy 生成；
- T1b 数据完全外部固定，但只覆盖 arithmetic correct-answer likelihood；
- 二者一致时，macro-relevant one-step interpretation 更强；
- 二者冲突时必须分别报告，不能挑选有利 endpoint 或平均抵消。

greedy generation reward、exact-answer rate 和 completion identity 属于 semantic/long-run validation
ledger，不替代 T1a/T1b continuous shift。

## Bias-contributor language

- 仅降低 U1/U2：`contributes to average update shift`；
- 降低 T1a 或 T1b：`contributes to the named one-step task-transition shift`；
- 同时降低 U 和 T：支持一条局部 propagation chain；
- 只有 multi-step repaired training 改善预声明长期 endpoint，才能讨论 long-run contribution；
- 以上均不是 correctness，除非另有 truth/spec authority。

## Remaining identity fields

具体 generation rule 与 static bank 已冻结在 machine-readable C0 manifest。state-derived rollout
artifact hashes 和 common evaluator hashes 在 collection 时写入，缺失则 T1 `INVALID`。
