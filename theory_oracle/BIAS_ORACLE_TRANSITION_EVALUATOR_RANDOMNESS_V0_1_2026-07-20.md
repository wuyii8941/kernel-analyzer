# Bias Oracle：transition repeat 与 evaluator repeat 必须分开

状态：`CALIBRATION_CONTRACT_CLARIFICATION`

## 两种重复不是一回事

对一个冻结 state，存在两个不同层次的重复：

1. **transition repeat `r`**：fresh process 恢复同一 pre-state，分别执行 eager/candidate
   完整一步，得到各自 post-state。这一层识别执行实现本身的 same-state variability。
2. **common-evaluator repeat `e`**：对某一个已经固定的 post-state，用同一个 T1a/T1b
   ruler 重复评估。这一层只识别测量 ruler 自身的 variability。

T endpoint 的 state-level paired effect 应先在每个 transition repeat 内对 evaluator repeats
取均值，然后做 candidate-minus-reference：

`d[s,r] = mean_e T(candidate_post[s,r],e) - mean_e T(reference_post[s,r],e)`。

随后：

- state effect 是 `mean_r d[s,r]`，不是 population B；
- transition runtime component 来自 `d[s,r]` 在 `r` 间的变化；
- evaluator component 来自固定 post-state 内 `e` 的变化；
- 二者不得 pooled 成一个含义不明的 variance。

## T1a bank 的随机性

T1a rollout bank 的生成看似随机，但在本查询中 seed、prompt、sampling protocol 和
pre-state 都已冻结，生成后 bank 是一个固定测量 artifact。它不是 transition runtime
noise。两个 transition repeats 必须使用同一个 bank；若每个 repeat 重新采样 bank，
则测量的是同时改变 transition 和 evaluation sample 的另一个 estimand。

两个 fresh bank generations 只用于验证固定协议可重建相同 artifact。若不能精确复现，
T1a 应标为 `INDETERMINATE_EVALUATION_ARTIFACT`，而不是把差异塞入 N。

## 对现有 selected-state smoke 的限制

旧 smoke 使用一个 self-exact transition post-state，再对该 post-state 做两次 evaluator
reconstruction。这足以验证构造，但不能替代 calibration record 中的
`transition_repeat × evaluator_repeat` 两层索引。新的总体记录必须保存这两个索引。

