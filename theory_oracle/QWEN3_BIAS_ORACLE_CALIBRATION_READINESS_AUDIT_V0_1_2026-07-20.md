# Qwen3 bias Oracle calibration readiness audit v0.1

## Verdict

`C0 THEORY/QUERY STRUCTURE FROZEN; C1 STATISTICAL CORE VALIDATED; CALIBRATION-0
SOURCE AND 24-SNAPSHOT AUDIT VALID; STEP-10/18/57/62/78 FULL RECORDS VALID`

已完成一个旧 B/step-29 state 的 T1a/T1b common-evaluator GPU construction smoke，以及一条完整
300-step calibration-0 eager source trajectory 和 24 个预声明 snapshot。当前 gate 是逐 snapshot
内容审计全部通过；预先冻结的 step-10 state 完整 record 也已通过。阻塞项不是 bias/variance 概念未定义，
而是将同一 record contract 扩展到剩余 states 和 independent trajectories。

## Requirement audit

| requirement | evidence | status |
|---|---|---|
| B/H/N/U separation | population core + 7 tests | READY |
| paired covariance retained | estimator consumes paired effect repeats | READY |
| trajectory is top-level sample | df test; cluster coverage simulation | READY |
| small-cluster method scope | literature rationale + Monte Carlo | READY, REGULARITY-CONDITIONAL |
| rare-trajectory limitation | tail formula/test; 5% example requires 59 | READY, CLAIM CHOICE OPEN FOR CONFIRMATION |
| update direction endpoint U1 | stored paired ratio + reference norm reconstruct denominator-free aligned dot；relative ratio retained as diagnostic | READY |
| U1 is not mistaken for macro effect | U/T endpoint split and retrospective orthogonal counterexample | READY |
| task-transition endpoint structure | T1a heldout GRPO + T1b correct-answer NLL | RULE FROZEN |
| common task evaluator | same eager FP16/SDPA-math evaluator for both post states | T1a/T1b SELECTED-STATE GPU SMOKES VALID |
| Q-R full-training population | pinned base revision, 4 seeds/data blocks, 96 selected pre-steps | DESIGN FROZEN |
| multi-state capture non-mutation | 6-step no-capture/capture final state and streams exact；arbitrary steps 1/4 also exact | READY FOR SOURCE COLLECTION |
| full 300-step state capture | calibration-0 reached step 300; 24/24 content/history/identity audits valid | READY FOR CALIBRATION-0 PROBES |
| source config provenance | recorded source YAML/metadata is bound to plan seed, data slice, 300-step horizon, 24 target digests and zero source compile invocations | CALIBRATION-0 RE-AUDIT VALID; REQUIRED FOR CALIBRATION-1/2/3 |
| prospective compiled treatment identity | per-state pre-endpoint realization contract fixes protocol, graph family and scorer identities | STEP-10 GPU CONTRACT VALID |
| trajectory-aware arm/pair record schema | paired estimands separated from arm outcomes; U2 artifacts hash validated | STEP-10 POPULATION-ELIGIBLE RECORD VALID |
| T1a state-derived bank generation | explicit sampling, two fresh banks exact, tied-group rule enforced | STEP-10 VALID; 8/8 INFORMATIVE GROUPS |
| T1b common post-state evaluation | exact post reconstruction, 64-example bank, two exact repeats | STEP-10 VALID |
| transition/evaluator repeat separation | nested `transition repeat r × common evaluator repeat e` contract and evaluator | STEP-10 BOTH LEVELS EXACT |
| frozen-bank scalar aggregation | exact plan identity/repeats revalidated; unavailable endpoint retained rather than complete-case deleted | READY; AWAITS REMAINING 19 RECORDS |
| signed high-dimensional U2 aggregation | coordinate-wise candidate-reference mean field, sharded artifact, separate H/N traces; per-state L2 labelled magnitude-only | READY; AWAITS REMAINING 19 RECORDS |
| four-trajectory calibration aggregation | complete trajectories aggregated at trajectory level; scalar and signed-vector U2 paths retain top-level df=3 | READY; FAIL-CLOSED ON CURRENT 11/96 STATES |
| reference-boundary conditional diagnostic | signed PPO/GRPO margin reconstructed from frozen old log-probs/advantages; recorded arm decisions validated; explicit tau grid remains calibration-only | READY ON 11 STATES; CONDITIONAL POPULATION CLAIM NOT ALLOWED |
| boundary-family support freeze | write-once selection reads only reference-side support; retains all tau with >=2 exposed states in every trajectory x phase; effect fields excluded | IMPLEMENTED/TESTED; AWAITS FOUR COMPLETE TRAJECTORIES |
| boundary independent confirmation | separate frozen family/spec; reference-repeat-1 condition anchor; paired-repeat N; simultaneous intervals; trajectory sign-flip; no cross-family joint claim | IMPLEMENTED/TESTED; MANIFEST/SPEC/DATA UNINSTANTIATED |
| boundary confirmation resource planning | derives minimum J from frozen family size/alpha and discrete sign-flip resolution only; excludes effect mean/sign/variance; spec binds plan and final bank | IMPLEMENTED/TESTED; AWAITS SUPPORT-COMPLETE FAMILY |
| boundary attribution bridge | requires confirmed operator-ready tau endpoint; freezes exact exposed state census and reference-anchor token-mask hashes; prohibits post-intervention membership change | IMPLEMENTED/TESTED; REAL BRIDGE CANNOT EXIST BEFORE CONFIRMATION |
| four-trajectory execution ordering | calibration-0 external chain is polled by authoritative ledger/records; calibration-1/2/3 launch serially only after predecessor completion; source partials/GPU contention/storage floor fail closed | CAMPAIGN ACTIVE; CALIBRATION ONLY, NO POPULATION CLAIM |
| prospective confirmation precision | variance upper-bound + t half-width + multiplicity/tail/resource gates + sign-flip p-value resolution; observed calibration mean/sign excluded from sizing | PROCEDURE TESTED; HALF-WIDTHS/FLOORS/CAP/TAIL UNINSTANTIATED |
| high-dimensional U2 confirmation | four-trajectory signed mean field can freeze independent directional replication; LOO cross-fitted planning dispersion; fail-closed direction freezer; raw signed delta projection enters the same trajectory B/H/N/U estimator; full-vector omnibus remains separate | PATH TESTED; AWAITS VECTOR CALIBRATION, FLOOR, STABILITY AND PRECISION THRESHOLDS |
| independent confirmation evaluator | precision plan fully recomputed from hashed calibration/spec; planner/estimator/sensitivity/record/U2 dependency hashes frozen; prospective SHA-ranked bank; source config/plan/audit binding; calibration exclusion; exact count; adjusted interval alpha and endpoint abstention | PROCEDURE TESTED; MANIFEST UNINSTANTIATED UNTIL CALIBRATION PRECISION IS KNOWN |
| prospective confirmation trajectory bank | fixed pre-result SHA ranking over 108 eligible non-calibration data blocks; deterministic unique seeds and 8x3 state plans; bank bound to precision J | DESIGN/BUILDER TESTED; NO CONFIRMATION DATA COLLECTED |
| confirmation manifest freeze | precision/bank/config/plan/evaluator hashes bound once; refuses freeze if any confirmation outcome marker already exists | FREEZER SELF-VALIDATES; AWAITS CALIBRATION-DERIVED J |
| confirmation execution campaign | complete manifest/precision/code preflight before collection; serial source→audit→24 records→final evaluator; no automatic deletion/operator launch | DRIVER TESTED; CANNOT START FROM CURRENT UNINSTANTIATED TEMPLATE |
| storage/retention scope | observed 27 GiB/state; 2 TiB safety floor; one prospective SHA-selected replay snapshot per phase/trajectory; raw tensors eligible only after scalar/U2/downstream/hash gates | POLICY FROZEN; NO AUTOMATIC DELETION; NO RAW EVIDENCE DELETED |
| few-trajectory sensitivity | frozen studentized trajectory sign-flip; exact through 16 trajectories, fixed-seed MC above; veto-only conflict rule | PROCEDURE TESTED; DOES NOT PROMOTE PRIMARY FAILURES |
| Bias-contributor confirmation isolation | candidates frozen after endpoint confirmation; contribution-specific precision input spec → reproducible plan; final new contributor trajectories; exact claim unit, multiplicity, interaction and intervention-integrity gates | THEORY/ARTIFACTS/PLANNER/FAIL-CLOSED VALIDATOR READY; 13 PRECISION+DESIGN TESTS PASS; C2 B NOT YET CONFIRMED |
| Bias-contributor result estimand | paired reference/candidate/repair cells form baseline, residual and contribution profiles; C has separate B/H/N; overshoot, absolute-reduction and veto-only sensitivity separated | ESTIMATOR READY; 9 TESTS PASS; REAL CONTRIBUTOR DATA NOT STARTED |
| contributor-pilot summary provenance | planner accepts a frozen dispersion-only summary and is invariant to mean/sign; real raw three-arm records → per-candidate pilot profiles → hashed summary chain | NOT YET IMPLEMENTED; REQUIRED BEFORE C3, NOT A C2 BLOCKER |
| end-to-end synthetic controls | core unit tests and coverage simulation exist | PARTIAL; RECORD/FAILURE GATES MISSING |
| confirmation trajectory count/tail scope | requires calibration variance and claim choice | CORRECTLY DEFERRED |
| practical materiality | no independent tolerance yet | UNINSTANTIATED, DOES NOT BLOCK SHIFT EXISTENCE |
| correctness | no independent authority | UNINSTANTIATED |

## Why old capture could not simply be looped 96 times

Old A/B/C natural-transition machinery proves a full state can be replayed, but the state source is a 30-step
restart and the executor expects a single selected snapshot/history. The new Q-R estimand requires one untouched
eager trajectory to continue through 300 steps while selected pre-states are forked into isolated eager/compiled
probes. Reusing a probe-mutated state in the source trajectory would change `P_R` and invalidate later states.

## Required construction sequence

1. **Source trajectory capture**：在 300-step eager trajectory 中，仅在 manifest-selected pre-steps
   复制完整 model/optimizer/scheduler/scaler/RNG/minibatch；capture 前后 source state hashes 必须相同。
2. **Isolated matched arms**：每个 frozen state 在 fresh processes 中执行 eager/compiled repeats；
   candidate graph identity、state identity 和 natural AdamW transition gates沿用已验证机制。
3. **U evaluator**：repeat 1 保存/流式计算 U1、U2 delta；repeat 2 至少提供完整 hash/self controls。
4. **T evaluator**：在 update 前生成冻结 T1a bank；两个 post states 都交给 common evaluator计算
   T1a/T1b；evaluation 不得改写 post state。
5. **Record validator**：按 C0 duplicate key、phase coverage、artifact hash、two-repeat 和
   trajectory weighting 规则 fail closed。
6. **One-state smoke**：旧 B state 的 exact reconstruction + T1a/T1b common evaluator 已通过；仍需补
   source-capture non-mutation gate 和 population writer integration，之后才执行一条完整
   calibration trajectory，最后再扩到四条。

## Resource discipline

- source process 与 probe processes 不并存持有重复 GPU models；每个 arm 完成后释放 CUDA cache；
- repeat 1 保存 reference update 与 candidate-reference delta 所需 artifact，repeat 2 优先 hash-level
  self check；不为相同目的保存多份 2.3GB vectors；
- snapshot、vector 和 T1 bank 分 manifest 管理；只有在 evaluator、hash 和 downstream aggregate
  全部验证后，才可依据明确 retention policy 删除冗余临时副本；
- calibration 与 confirmation 目录完全隔离，防止后续 evaluator tuning 泄漏 confirmation。

## Immediate next implementation gate

calibration-0 source、24-state audit 和按事先冻结规则选择的
early/step-10、step-18、step-57、step-62、step-78、step-91、step-96、
step-97 以及 middle/step-105 完整记录均已通过。
step-10 显示稳定 nonzero loss/gradient/update/T effects、paired N=0、无离散 fork，且 T1a/T1b 符号相反；
这些都是 state effects，不是 B。下一 gate 是沿完全相同 contract 执行剩余 14 states，再形成一条
trajectory estimate；仍不触发 operator repair sweep。

聚合、precision planning、confirmation isolation、sensitivity 与 contributor design/result gates 已补齐并
本次 B/Mode E/campaign 修正的 49 项直接定向测试通过；nightly 环境补齐兼容的 pytest 8.x 后，完整
测试套件当前 `375 passed`。单条 trajectory 只输出 trajectory mean、phase/state H 和
same-state N；between-trajectory variance、顶层 interval 和 population B 均保持未识别，而不是填成 0。

离散事件聚合也已拆开 signed directional rate 与 nonnegative disagreement rate：step-97 已出现净 clip
count shift 为 0、但一上一下两个 token decision flips 的实际反例。后者只作为 semantic-impact profile，
confirmation planner 会拒绝将其当成 signed Bias endpoint 或据此启动 Bias-contributor 搜索。

clip rate 的权重也已冻结为“state 内按 eligible decisions 归一化，再对 states 等权”，并单独保留
`clip_decision_exposure_count`。该历史字段记录的是 completion-token decision positions，而不是
`advantages != 0` 的 eligibility mask。当前十一个有效 states、两个 repeats 的 position count 均为 512，所以它与
exposure-pooled rate 暂时数值相同；这只是当前 design 的事实，不允许据此省略一般情况下的权重声明。

same-state N 也已绑定固定 realization：同一 arm 的 repeats 必须保持 compiler-config 与 graph-family
digest 一致，当前十一个有效 states 全部满足。该证据不覆盖最终 generated-kernel/autotuning variant identity，
因为现有记录没有该字段；未观测的 variant variation 若存在仍混在 N 中，不能声称已单独排除。研究或
排除 compiler-choice randomness 时必须扩展 instrumentation 或另立 factor/query。

task endpoint 的 randomness scope 也已显式化：transition repeats 识别 conditional transition N；nested
evaluator repeats 检查 common evaluator repeatability；两个 T1a bank files 是同 seed 的 fresh
reproducibility checks，而非独立 bank samples。因此 T1a bank-sampling variance 当前未识别，T1a H
包含 state-adaptive bank content 变化；T1b 才是固定 correct-answer-bank functional。

task evaluator 的 acquisition source 在 campaign 中途有过版本变化，因此当前十一个 records 还接受一条
仅限 calibration 的兼容路径：沿 artifact provenance 重新加载四个 arm，以当前 evaluator 语义重算
T1a/T1b、核对 evaluator-repeat ID 与两个 bank 的 content hash。前八个旧记录被明确标为
`LEGACY_IMPLICIT_SCOPE_NUMERICALLY_REVALIDATED`，step-105/107/108 为 `EXPLICIT_CURRENT_SCOPE`；十一个均通过。
这不能外推成 confirmation 的版本兼容承诺：confirmation acquisition 必须在开始前冻结为一个 source hash。

confirmation 的 threshold source 现有独立协议草案：reference/reference 与 candidate/candidate
fresh-process controls 覆盖 acquisition/runtime null，identical-post-state evaluator control 覆盖 evaluator
null；详见 `QWEN3_CONFIRMATION_THRESHOLD_SOURCE_PROTOCOL_V0_1_2026-07-20.md`。它尚未产生任何数值
half-width、variance floor 或 shift floor，因此 readiness 仍为 `UNINSTANTIATED`，不能据此提前启动
confirmation。

单轨迹与四轨迹 null-control aggregators 已实现并以 5 个直接定向测试覆盖，另有 campaign integration
tests。单轨迹产出在 24-state census 不完整时
fail closed；当前十一个完整 records 仅做只读预检，reference/reference、candidate/candidate、nested
evaluator repeats 以及 update/next-state/event identity 均未观察到变化。由于每层只有两次 repeats，
该结果只支持 `NO_OBSERVED_WITHIN-STATE_VARIATION_AT_R=2`，不支持“runtime variance 为零”。

另外，readiness audit 不再默认 desired half-width 必须被强行实例化。`B existence`、估计精度和 practical
materiality 是三条独立 claim axis：缺少外部精度依据时推荐固定资源的 Mode E，预注册 J/family/floor
后检验 adjusted interval 是否位于 null floor 一侧；未检出保持 fixed-resource indeterminate。只有存在独立
half-width/variance floor 时才使用 Mode P。planner/evaluator 的 Mode E 核心迁移已经完成：fixed-resource
模式不会虚构 desired width，并能对 global 与预声明 phase-conditioned endpoint 产生一致的 interval report
和 attribution eligibility。实际 fixed J、family、multiplicity、tail scope 与 shift floor 尚未冻结，因此仍
不能提前启动 confirmation。

全局 B 门槛也已修正：operator analysis 不再要求 global mean 必须先显著。合法入口包括独立确认的
global B，以及事前冻结、由 reference/pre-intervention variable 定义的 conditional B。当前新增边界诊断
在十个完整 states 上重建 signed clip margin 并逐 token 验证 eager/compiled decisions；它观察到全局平均
很小而 near-boundary 子集出现方向 event change 的可能性，但各 tau 的有效 states 至多 5 个，且 tau 尚未
冻结。因此这里只能作为 calibration discovery，不能写成 conditional Bias，更不能据此启动算子归因。
该 summary 已补齐冻结 plan 的精确 24-state census、record/transition/minibatch 与分析代码哈希；conditional
权重固定为 equal trajectory→phase→exposed state→near-boundary token，任一 phase 无 exposure 即未实例化。
后续 tau 合同禁止按 observed effect 大小/符号选阈值：只按 reference-side support 排除不可定义 tau，所有
support-complete tau 一起进入 independent confirmation multiplicity。独立 evaluator/spec 接线现已完成，
但真实 family manifest 与 confirmation bank 尚不存在；现有 contributor validator 也仍不认识 boundary
endpoint identity。因此该路径当前仍不能产生 population claim 或启动归因。
