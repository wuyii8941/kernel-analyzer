# Qwen3 bias Oracle calibration contract — draft v0.1

状态：C0 草案。只有 `OPEN` 项被关闭并生成 machine-readable manifest 后，才启动新的
population GPU collection。

## 1. Query identity

- subject：Qwen3-0.6B GRPO training step；
- baseline realization：eager，固定 precision/attention/training configuration；
- candidate realization：TorchInductor，固定 compiler configuration，记录 graph/kernel identity；
- interpretation：implementation-relative shift；eager 不作为 correctness truth；
- primary comparison unit：从同一完整 pre-step state 出发的一次 natural optimizer transition。

任何遗漏 model parameters、optimizer moments/counters、scheduler/scaler、RNG、minibatch、old
policy values 或 compiler realization identity 的 record 均为 `INVALID`。

## 2. Two population queries

### Q-R（primary）

从独立 eager-anchored trajectories 按 phase/step inclusion rule 抽取 frozen states。估计在
baseline visited-state distribution 上，将 realization 换成 candidate 的 average one-step shift。

### Q-C（sensitivity）

从独立 compiled-anchored trajectories 用相同的 phase/step inclusion rule 抽取 frozen states。
在每个 state 上重新执行 eager/compiled matched probes。Q-C 不与 Q-R 池化；两者差异属于
state-distribution feedback evidence。

## 3. Endpoint contract

不存在一个 endpoint 同时充分表示 update geometry 和宏观训练含义。合同使用两个 primary
ledgers；operator claim 必须注明归属哪个 ledger。

### Update ledger U1

对全部实际参与 optimizer update 的 parameter coordinates，按稳定 parameter ordering，以
FP64 accumulation 计算：

primary：`aligned_forcing(s) = dot(U_C-U_R, U_R)`。

描述性相对量：`relative_aligned_shift(s) = dot(U_C-U_R, U_R) / ||U_R||^2`。

- `U_R/U_C` 是实际 AdamW/clip/AMP/scheduler suffix 产生的 parameter update，不是假想 SGD；
- primary 在 `||U_R||=0` 时自然为 0；相对量不用 epsilon 静默填值，标记 `UNDEFINED` 并单独审计；
- full-model energy weighting 是 primary；按 layer/module 的结果只作 conditional ledger，不等权
  平均成另一个 post-hoc primary；
- primary 正负表示对 baseline update 的 aligned forcing，不表示改善/伤害、百分比加速或 correctness。
- primary 依赖冻结的 parameterization、optimizer 与 update scale，不是函数空间或跨训练配置不变量；
  跨 phase 的 scale dependence 必须与 relative profile 一起报告。

### Update ledger U2

`mean_update_field = E[U_C-U_R]`，保存完整向量摘要、layer slices 和预声明 normalized norm。
它识别固定 parameter coordinate frame 中的平均 shift。

### Task-transition ledger T1 — GENERATION/STATIC-BANK RULE FROZEN

T1a：在 update 前由 baseline-anchored protocol 生成、但不参与 update 的 frozen rollout bank
`E_S` 上计算 `L_GRPO(S'_C;E_S)-L_GRPO(S'_R;E_S)`。

T1b：在固定外部 arithmetic prompt/correct-completion bank 上计算 post-update teacher-forced
correct-answer NLL difference。

两者不合并；具体 prompt IDs、generation seed、old-logp/advantage construction、mask、weight 和
scorer identity 由 C0 manifest 冻结。两边 post states 必须用同一个 common eager evaluator，不能
让 evaluator implementation discrepancy 混入 state-transition effect。T1 未实例化时，U1/U2
只能支持 update-bias claim。

### Propagation ledgers

loss、unclipped/clipped gradient、clip/AMP/skip events、optimizer moments 和 next-state hashes。
这些解释传播，不与 P1 合并成一个 verdict。

## 4. Randomness and coupling

- trajectory generation RNG：在 trajectories 之间开放并记录；
- matched-state probe：两实现共享 frozen state、batch/token 和 algorithmic RNG state；
- same-arm repeats：不同 fresh process，使用同一 declared RNG and realization protocol；
- GPU scheduling/atomic behavior：不假定 deterministic，由 repeats 实测；
- autotuning：选择协议固定并记录 chosen realization；identity 不一致不得池化；
- stochastic rounding：默认关闭；若实际开启，必须改为单独 randomness query；
- pairing analysis 基于 `D=C-R`，同时保留 ref-self、cand-self 与 paired-repeat variability。

## 5. Sampling and weighting

### Calibration bank

- 4 independent trajectories；
- early/middle/late 3 个预声明 phase；
- 每 phase 按固定 PRNG 从 eligible steps 抽 8 states；
- 共 96 states/query anchor；每 implementation/state 至少 2 repeats；
- 只冻结尺度、controls、analysis 和 prospective precision，不产生 population verdict。

### Confirmation bank

- 使用 calibration 未出现过的 trajectory/data/RNG seeds；
- 至少 8 independent trajectories，仍按 3 phases × 8 states；
- trajectory count、desired half-width 和 resource cap 在 calibration 后、confirmation 解盲前
  通过 prospective precision calculation 一次性冻结；
- tail coverage target `p_min/alpha`：`OPEN`。若选择 `p_min=0.05, alpha=0.05`，至少需要 59
  independent trajectories；8 条只支持显式 regularity assumptions 下的 first-batch mean analysis；
- confirmation 中途不得依据 effect sign、significance 或当前 interval 追加 trajectory；扩样必须
  作为新版本 study，并显式处理 sequential reuse；
- trajectory 等权；phase 权重预声明等权；phase 内 sampled states 等权；token/coordinate 不作为
  独立 population samples。

Q-R 和 Q-C 各自需要满足 sampling contract。若资源只允许先完成 Q-R，Q-C 明确为
`NOT_YET_INSTANTIATED`，不能暗示 distribution transport。

## 6. Statistical contract

对 scalar U1/T1：

- state effect：same-state paired repeats 的 mean；
- B：按 trajectory→phase→state 的声明权重聚合；
- H：报告 trajectory、phase 和 residual state heterogeneity，不把三者压成一个无法解释的数；
- N：same-state paired-difference variance，并并列 ref-self/cand-self；
- U：trajectory 是最高 sampling unit；trajectory 内 block correlation 通过先生成 trajectory-level
  estimate 处理，不把 step 当顶层自由度；
- primary interval：对 independent trajectory estimates 使用 small-sample t procedure；
- sensitivity：CR2/Satterthwaite 与 wild cluster bootstrap-t；若与 primary 实质冲突则
  `INDETERMINATE_METHOD_SENSITIVITY`；
- tail ledger：`J_tail=ceil(log(alpha)/log(1-p_min))`；未达到时不得声称覆盖该 prevalence 的
  trajectory regime；
- ordinary state-level normal interval 禁止用于 confirmatory verdict；
- confirmation 只运行一次 frozen evaluator；不得反复改方向/权重直到显著。

对高维 U2：完整 mean field 是 estimand；norm inference 的 procedure 必须通过 null、fixed-shift
和 rotating-shift simulations。未经校准的 coordinatewise normal approximation 只作 descriptive。

## 7. Required controls

1. eager/eager；
2. compiled/compiled；
3. fixed signed shift；
4. deterministic state-varying zero-population-mean shift；
5. same-state stochastic sign-changing shift；
6. rare-state deterministic shift；
7. state weighting imbalance trap；
8. trajectory correlation trap。

controls 需要验证 estimate、interval coverage、verdict 和 fail-closed behavior。特别是增加同轨迹
states 不得虚假缩小 trajectory-level uncertainty。

## 8. Verdict ledgers

### Construction

`VALID / INVALID`：state completeness、pairing、realization identity、call counts、artifact hashes。

### Shift existence

`REPRODUCIBLE_AVERAGE_SHIFT / NO_STABLE_AVERAGE_DETECTED / INDETERMINATE`。

H/N 分别报告未约束估计、非负描述值和 trajectory-level uncertainty，不用截断后的 H 点估计修改
shift-existence verdict。若要声称 conditional B，必须使用事前冻结且经 multiplicity 调整的
`endpoint::phase=<phase>` 独立 confirmation；普通 H profile 不自动产生条件方向结论。

### Materiality

`MATERIAL_AVERAGE_SHIFT / DETECTED_BUT_BELOW_MATERIALITY /
PRACTICALLY_EQUIVALENT_AVERAGE_SHIFT / UNINSTANTIATED_MATERIALITY`。

`tau_B` 当前为 `OPEN`。在没有 independent tolerance 时，shift-existence ledger 可以完成，
materiality 必须保持 `UNINSTANTIATED`。

### Correctness

默认 `UNINSTANTIATED_NO_INDEPENDENT_AUTHORITY`。

### Tail/population scope

`TAIL_COVERAGE_SUFFICIENT / TAIL_COVERAGE_INSUFFICIENT /
REGULARITY_CONDITIONAL_ONLY / FINITE_BANK_ONLY`。普通 normality/skewness test 未拒绝不能升级
tail coverage。

## 9. Operator bias-contributor gate

只有 U1/U2 或 T1 获得 independent confirmation 的
`REPRODUCIBLE_AVERAGE_SHIFT` 后启动。候选 repair 在相同 population weighting 上估计：

`C_o = B_full_compiled - B_compiled_with_repair_o`。

要求：

- confirmation trajectories 上降低 `|B|`；
- same-arm runtime controls 有效；
- candidate/repair realization identity 合同有效；
- calibration 与 confirmation 的作用方向不因 post-hoc selection 产生；
- top candidates 检查 pair/coalition interaction；
- fused callable 只能叫 generated-region contribution，除非 source operator 被真正隔离。

本项目不要求定位 N 的 contributor。U-ledger contributor 只能称 average update-shift
contributor；要称 macro-relevant bias contributor，至少需要 T1 transport，长期 harm 仍需
multi-step validation。

## 10. Open items before collection

| item | recommended resolution | consequence if unresolved |
|---|---|---|
| T1 generated artifact hashes | collection 时由冻结规则生成并登记 | 缺 hash 时 T1 `INVALID`，不阻塞 U1/U2 |
| practical `tau_B` | independent perturbation/sensitivity calibration | materiality uninstantiated，不阻塞 shift existence |
| confirmation trajectories / half-width / cap | calibration 后、confirmation 解盲前冻结 | population precision 未声明 |
| method coverage | trajectory-t primary；CR2/wild-cluster sensitivity 经 synthetic audit | audit 失败则 population U/verdict 不可用 |
| tail prevalence `p_min/alpha` | 由目标 claim 与资源共同冻结 | 只能 regularity-conditional 或 finite-bank |
| Q-C budget | Q-R 完成后按同合同执行 | distribution transport 不可声称 |
| source-op intervention unit | 保持 realization 的 tap/override，否则 region | 只能作 region contribution |

## 11. Immediate next action

不启动新的 operator repair GPU sweep。U1 selected-state evaluator 已实现并通过 unit tests；下一步
实现 trajectory-aware record schema、
八类 controls 和 precision simulation；然后将本草案中的 `OPEN` 项冻结为 machine-readable C0
manifest，再采集 Q-R calibration bank。
