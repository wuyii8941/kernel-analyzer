# Qwen3 calibration-0 step-10 end-to-end Oracle findings v0.1

## Verdict

`VALID_POPULATION-ELIGIBLE_CALIBRATION_RECORD; ONE STATE ONLY; NOT B`

预先冻结的 `calibration-0 / early / step-10` 已完成完整链：

`audited pre-state → prospective compiler realization → 2×2 fresh transition arms → U1/U2 → frozen T1a bank → common T1a/T1b evaluator → arm/pair record validation`。

该 state 是 frozen sampling design 中的合法记录，但一个 state、一个 trajectory 不能估计 population B、H 或 trajectory uncertainty。

## Construction evidence

- calibration-0 300-step eager source 正常结束；
- 24/24 snapshot batch audit 为 `VALID`；
- step-10 realization contract 在 loss/backward/update 前冻结；
- history replay 与 complete preflight 都没有改变状态；
- compiled measured scorer 调用 runtime 一次，未产生新 specialization；
- eager/compiled 各两个 fresh-process repeats；
- 两个 eager repeats 与两个 compiled repeats 各自 scorer/update/next-state bitwise exact；
- T1a 两份 fresh bank 的 content digest 完全相同；
- 每个 transition post-state 的两个 common-evaluator repeats 完全相同；
- record validator：4 arm records、2 paired records、U2 artifact hashes 全部有效。

## Observed state effects

以下只描述 step-10；数值大小不能跨 endpoint 直接比较，也没有 materiality tolerance。

| endpoint | step-10 observation | runtime/evaluator variability |
|---|---:|---:|
| training loss candidate-reference | `-2.428889274597168e-06` | paired repeat variance `0` |
| clipped-gradient delta L2 | `0.0089510877162742` | paired vector noise `0` |
| parameter-update delta L2 | `3.5757491034157206e-05` | paired vector noise `0` |
| update delta / reference update L2 | `0.0036017213695277196` | exact repeats |
| update delta cosine with reference update | `-0.000431412340582289` | exact repeats |
| U1 reference-aligned shift | `-1.5538270461532006e-06` | exact repeats |
| T1a heldout GRPO surrogate shift | `+0.00015005911700427532` | transition N `0`; evaluator variance `0` |
| T1b heldout correct-answer NLL shift | `-0.0003450540825724602` | transition N `0`; evaluator variance `0` |

完整 next state 在 implementation 间不同，但以下离散事件均未改变：token clipping decisions、global gradient clipping trigger、AMP scale/skip、nonfinite gradient/update。

## What this calibrates

1. **单 state repeat mean 不是 B。** 它是 `state effect`。只有在 frozen state distribution 上按 trajectory/phase/state 权重聚合后，才可计算 calibration B；独立 trajectories 才提供 top-level uncertainty。
2. **浮点 discrepancy 不是 N。** 本 state 有稳定 nonzero implementation effect，同时 paired runtime N 为零。
3. **无 fork 不等于无 semantic impact。** 训练分支事件没有 fork，但两个 common task endpoints 都有稳定 state effect。
4. **一个 update 标量不够。** U2 magnitude 非零且约为 reference update 的 0.36%，但几乎与 reference update 正交；U1 非常小。U1 不能代替 U2 geometry。
5. **T endpoint 不可合并。** T1a 与 T1b 符号相反并不矛盾；它们定义不同语义。Oracle 必须输出 endpoint-indexed profile，不能压成一个“training bias score”。
6. **exact next-state disagreement 不是 correctness。** 没有独立数学/specification authority，也没有 materiality tolerance；当前只能说 implementation-relative transition discrepancy。

## What remains unknown

- 该 state effect 在其他 early/middle/late states 上是否同向；
- state/phase/trajectory heterogeneity H；
- calibration trajectory mean 是否接近零；
- independent-trajectory uncertainty U；
- T1a/T1b 哪一个与长期训练质量更相关；
- 任何 effect 是否超过实践容忍度；
- 哪些 operator 对未来确认的 population endpoint B 有贡献。

## Consequence for the next stage

calibration-0 剩余 23 states 必须沿同一 record contract 执行，不能按当前 step-10 effect、fork 或
source gradient 筛选。聚合时先在 repeat 内形成 paired effects，再按 state→phase→trajectory 聚合。

只有当至少四条 calibration trajectories 给出可解释的 endpoint-specific B/H/N 尺度后，才能冻结
confirmation 规模；只有 independent confirmation 支持稳定 endpoint B 后，才进入 bias-contributor
operator repair/injection。step-10 本身不能触发算子 root-cause claim。

## Evidence

- `results/oracle_calibration/qwen3_bias_oracle_calibration_0_v0_1/capture_batch_audit.json`
- `results/oracle_calibration/qwen3_bias_oracle_calibration_0_v0_1/step010/realization_contract.json`
- `results/oracle_calibration/qwen3_bias_oracle_calibration_0_v0_1/step010/transition_evaluation.json`
- `results/oracle_calibration/qwen3_bias_oracle_calibration_0_v0_1/step010/task_endpoint_evaluation.json`
- `results/oracle_calibration/qwen3_bias_oracle_calibration_0_v0_1/step010/record_bundle.json`
- `results/oracle_calibration/qwen3_bias_oracle_calibration_0_v0_1/step010/record_validation.json`

