# Matched-State Discrepancy Oracle completion audit

> 2026-07-20 状态说明：本文件保留为“结构定义与 selected-state 实例”的历史审计，
> 不再支持“训练分布上的 bias Oracle 已完成”或“可以直接进入全算子 bias 归因”的结论。
> population bias 校准、训练方向一致的 endpoint、trajectory-cluster uncertainty 与后续启动门，
> 以 `QWEN3_BIAS_ORACLE_CALIBRATION_PLAN_V0_1_2026-07-20.md` 为准。

审计对象：实现一个规范定义、能在 matched states 上区分 average effect、state dependence、runtime variability，并能连接语义与一步训练转移的 Oracle。

## 要求与证据

| 要求 | 状态 | 权威证据 | 审计结论 |
|---|---|---|---|
| 查询必须声明 state distribution、realizations、observable、randomness/coupling | 完成 | `DISCREPANCY_ORACLE_DEFINITION_V3.md`；Qwen contracts/manifests | 缺任一项会 INVALID/UNINSTANTIATED |
| implementation-relative error 与 truth-relative correctness 分离 | 完成 | v3 sections 1/6；结果 correctness=`UNINSTANTIATED` | 未默认 eager 为真值 |
| B/H/N 与 sampling uncertainty U 分离 | 完成 | `src/forkcert/oracle.py`；v3 section 2；33 tests | vector sign cancellation、repeat/H identifiability、finite-state uncertainty 有反例覆盖 |
| B 不被 tensor/state 符号抵消 | 完成 | sign-safe mean-effect norm；Qwen A/B/C 结果 | signed global mean 与 B norm 的真实反例已观察 |
| continuous 与 semantic endpoint 不偷换 | 完成 | v3 L1/L2；held-out semantic ledger | 真实 transition 中 event 不变但 update effect 非零 |
| 一步训练转移从完全相同完整 state 开始 | 完成 | VALID step-29 snapshot audit；A1/A2/B1/B2 pre-state identity | model、optimizer、scheduler、scaler、RNG、minibatch 均匹配 |
| 执行真实 optimizer suffix，而非假想 SGD | 完成 | natural-transition executor；captured AdamW/AMP/clip/scheduler | 旧 hypothetical-SGD script 未被用于该 claim |
| same-state repeats 识别 runtime/replay variability | 完成 | 两个 fresh processes/arm；完整 vector self controls | 本 selected state 的 loss/gradient/update N=0；只限当前协议 |
| H 与 population U 的可识别性 fail closed | 完成 | transition evaluator | 单 transition state 报 H=`UNIDENTIFIABLE`、population=`NOT_ESTIMATED` |
| impact ACCEPT/REJECT 与 correctness verdict 分开 | 完成 | v3 verdict ledgers；evaluation | selected-state impact REJECT，correctness UNINSTANTIATED |
| operator analysis 有明确理论接口 | 完成（定义层） | v3 section 5 | repair/injection 是 intervention estimand；当前 whole-scorer experiment 不称 operator effect |
| operator causal attribution 实例 | 未实例化，非 Oracle 完成前提 | result ledger | 需要 realization-preserving operator intervention；保持 NOT_CLAIMED |
| 长期训练危害 | 未实例化，独立 validation | v3；result ledger | 一步 effect 不外推收敛/精度/时间 |
| exact historical compiler-event lineage | 当前数据不可识别 | invalid attempts v1-v3；manifest revision 4 | snapshot 缺历史参数/optimizer states；claim 已降级为 fresh realization，不隐藏 |
| 经验跨模型/硬件普遍性 | 未证明，定义不承诺 | v3 section 8 | 需要新 state distributions；不影响结构性 Oracle 定义 |

## 实现与验证

- 核心实现：`src/forkcert/oracle.py`；
- 规范：`DISCREPANCY_ORACLE_DEFINITION_V3.md`；
- real scorer bank：30 Qwen state clusters、A/B/C 三条轨迹；
- real transition：4 个 fresh-process arms、完整 clipped-gradient/update vectors；
- 核心 tests：33 passed；
- Qwen/evaluator tests：13 passed；
- transition manifest hashes：0 mismatch；
- transition construction：`VALID`；
- selected-state transition compatibility：`REJECT_EXACT_SELECTED_STATE`；
- correctness：`UNINSTANTIATED`。

## 完成判定

规范 Oracle 的基础结构、selected-state estimands、identifiability、verdict ledgers、实现和一个真实训练转移实例已经完成并相互一致。

该完成判定不包含 target training-state distribution 上的 material bias 估计。现有 selected
states、normal-approximation interval 和 per-state repair evidence 不足以证明 population B，
也不足以启动“贡献长期 bias 的 operator”结论。

“完备”在这里是结构完备，而非宣称覆盖所有模型、事件或 operator。新增 subject 必须实例化同一个 query contract；不能通过新增 endpoint 改写 B/H/N/U、correctness 或 attribution 的含义。

后续 operator repair/injection、独立 correctness authority 与长期训练是建立在该 Oracle 上的研究阶段，不应被倒灌进 Oracle 定义或伪装为当前已经支持的 claim。
