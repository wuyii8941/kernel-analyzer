# Oracle GPU Pilot Findings — 2026-07-15

## 1. 这次实验回答什么

本轮不是寻找 compiler bug，而是检查我们提出的 Oracle 结构能否在真实 CUDA eager/compiled execution 上工作。受控 subject 包含 FP16 matmul、GELU、第二次 matmul、bias、cross-entropy、gradient 和 clipped-SGD update。

每个 observation 同时记录：

- compiled path 是否真的执行；
- continuous logits、loss、margin 和 gradient discrepancy；
- `class0 > class1`、argmax、ordered top-2、top-2 set 等不同 semantic endpoint；
- gradient clipping 和一步 update endpoint；
- eager--eager、compiled--compiled 和 eager--compiled；
- batch state、batch 内 case 和 exact repeat 三层单位。

运行环境：PyTorch `2.13.0.dev20260609+cu126`、CUDA 12.6、Tesla T4。pilot 源码 SHA-256 为 `38bf74c8b840d44dfa4c883d04bc58051e9f2310b6a771e06ffa070439613341`。

## 2. 两次独立运行

| | Calibration | Held-out confirmation |
|---|---:|---:|
| batch states | 192 | 192 |
| primary cases | 3,072 | 3,072 |
| exact repeats | 3 | 3 |
| compiled runtime invocations | 576 | 576 |
| backend graph count | 1 | 1 |
| graph code hash | `a4e012...f0aa` | `a4e012...f0aa` |
| nonzero self-pairs | 0 | 0 |

Confirmation protocol 在读取 confirmation output 前冻结于 [ORACLE_GPU_PILOT_CONFIRMATION_CONTRACT.md](ORACLE_GPU_PILOT_CONFIRMATION_CONTRACT.md)。协议审计发现，脚本的 seed 同时改变 base parameters 和 input states；因此 confirmation 是新的 parameter-state instance 加新 cases，不是固定权重条件下只重采样 inputs。

## 3. Measurement gate

两次运行均满足：

- 每个 candidate call 都进入 tracked Inductor callable；
- fixed shape 下只生成一个 graph；
- eager self-pair 和 compiled self-pair 完全为零；
- same-state、same-case 三次重复完全相同；
- 没有 fallback、NaN、Inf、crash 或 missing observation。

因此本轮没有重演旧 online scan 的 silent eager fallback 混淆。它支持把 execution identity 作为 Gate 0.5，而不只是普通环境信息。

## 4. Error、bias 和 variance 的实际结果

| Endpoint | Calibration | Confirmation | 判读 |
|---|---:|---:|---|
| mean signed margin delta | `-8.03e-6` | `-4.31e-6` | 两次 state-bootstrap 95% CI 均跨 0 |
| mean absolute margin delta | `2.02e-4` | `1.68e-4` | 非零，confirmation 在预注册 factor-2 范围内 |
| max absolute margin delta | `2.93e-3` | `1.95e-3` | tail 明显大于 mean |
| between-batch-state variance share | 5.91% | 6.22% | 稳定但不是主要部分 |
| within-batch between-case share | 94.09% | 93.78% | 主要 heterogeneity 来源 |
| exact-repeat variability share | 0% | 0% | 当前 deterministic protocol 下无 runtime noise |

最严格的结论是：

1. `D(s)` 是确定、非零且随 state/case 改变的 implementation discrepancy；
2. 没有证据支持目标分布上的稳定 global signed shift `μ_Q != 0`；
3. 当前 dispersion 是 effect heterogeneity，不是 runtime variability；
4. signed-mean confidence interval 是有限 states 带来的 sampling uncertainty，也不是 runtime variance。

如果把“任何固定实现差”都叫 bias，会把 per-state deterministic discrepancy 与 population-average direction 混为一谈。当前数据支持前者，不支持后者。

### State 单位也必须 endpoint-specific

对 gradient/update 来说，整个 batch 是一步训练 state；对 per-case logit/event 来说，batch 内 case 是更自然的 semantic observational unit。因此表中的 94% 可以叫“batch 内 case heterogeneity”；若 Oracle 的目标是 case-level decision，它又属于 state heterogeneity。不能脱离 endpoint 报一个通用的 state variance。

## 5. Semantic endpoint 不能由一个 margin 代替

| Endpoint | Calibration disagreements | Confirmation disagreements |
|---|---:|---:|
| strict `class0 > class1` predicate | 3 | 2 |
| argmax | 1 | 4 |
| ordered top-2 | 4 | 3 |
| top-2 set | 1 | 0 |

Calibration 中，三个 strict-predicate disagreement 全都没有改变 argmax；唯一 argmax disagreement 又没有改变 strict predicate。原因包括 exact tie 和程序 tie-breaking。另有 top-2 set 改变但 argmax 不变。

这推翻了“给一个 signed margin 就能代表 semantic event”的简化：

- `logit0-logit1` 只对应一个二元 predicate；
- argmax 需要 top-1/top-2 gap 加精确 tie rule；
- ordered top-k 与 top-k set 是不同 event space；
- routing、sampling 和 clipping 同样需要各自真实的 decision map。

现有记录没有保存自然 argmax case 的 top-1/top-2 numerical gap，因此不能用 class0/class1 boundary profile 解释 natural-stratum argmax disagreements。这是 pilot schema 的明确遗漏，下一版必须给每个 semantic endpoint 保存它自己的 decision variable 或充分表示。

## 6. Boundary conditioning 得到什么支持

预先构造的 near/far strata 没有根据 eager/compiled delta 事后筛选。

- 两次运行的 far stratum 都没有 class0/class1 predicate disagreement；
- calibration 的 3 次和 confirmation 的 2 次 predicate disagreement 全部在 near stratum；
- 所有 predicate disagreement 都满足几何必要条件：reference boundary distance 不大于 paired margin shift；
- mean absolute margin delta 在 near、far、natural 三层相近，semantic disagreement 的集中不是因为 near 层数值误差整体更大。

这支持“semantic effect 取决于 discrepancy 与 decision boundary 的联合结构”，而不支持把 raw delta magnitude 或 branch-distance distribution 单独当 Oracle。

near stratum 是受控富集分布，它的 disagreement rate 不能解释为自然 workload 的发生率。

## 7. 没有 persistent semantic direction 证据

- calibration：up/down 为 `0/3`，exact McNemar `p=0.25`；
- confirmation：up/down 为 `1/1`，exact McNemar `p=1.0`。

点估计方向没有 held-out 复现。当前支持 semantic disagreement 的存在，但不支持 stable directional semantic bias。把 calibration 的 `0/3` 单独写成 compiler direction 会是 selection error。

## 8. Transition endpoint

| Endpoint | Calibration | Confirmation |
|---|---:|---:|
| mean relative update L2 delta | `2.498e-4` | `2.439e-4` |
| mean absolute grad-norm delta | `7.67e-5` | `6.92e-5` |
| reference clip rate | 100% | 47.92% |
| candidate clip rate | 100% | 47.92% |
| clip disagreement | 0 | 0 |

相对一步 update discrepancy 在不同 parameter-state instance 上稳定复现，说明 numerical discrepancy 能传播到 transition endpoint。它仍然是很小、没有 application tolerance 的 relative impact，不能称为 harmful update。

Calibration 的 clip threshold `1.0` 使所有 states 都 clipping，因此完全没有辨别力。Confirmation 使用只由 calibration reference median 圆整得到的 `3.9`，获得约 48% clip rate，但 192 个 batch states 中仍无 boundary crossing。这个零结果可能只是对约 `7e-5` grad-norm delta 的低统计功效，不证明 clipping 一般不敏感。

## 9. 当前 Oracle verdict 示例

| Claim level | 当前结果 |
|---|---|
| M0 measurement calibrated | PASS：path identity、self-pair、missing/fallback audit 均通过 |
| M1 numerical discrepancy | YES：absolute/tail discrepancy 和 conditional heterogeneity 可复现 |
| M2 semantic discrepancy | YES on controlled distribution；event-specific、rare、无稳定方向；不能外推自然 rate |
| M3 one-step impact | YES：relative update discrepancy 可复现；无 harm threshold |
| M4 operational non-equivalence | INDETERMINATE：没有外部 acceptance region |
| M5 correctness failure | UNAVAILABLE：没有独立 specification/truth |

这就是多 endpoint Oracle profile 的具体产出。它允许某些层为 YES、某些层为 INDETERMINATE，而不强行压成一个“发生过 fork，所以 fail”的结论。

## 10. 对主计划的影响

### 已得到支持

- execution identity 必须先于统计分解；
- average shift、heterogeneity、runtime variability 和 sampling uncertainty 必须分开；
- deterministic-first 是有效起点；
- numerical、semantic、transition endpoint 不能合并；
- boundary conditioning 和 event direction 都必须显式记录；
- held-out confirmation 能阻止把偶然方向包装成 bias。

### 必须修正

1. 分离 parameter/checkpoint seed 与 input-state seed；
2. 每个 endpoint 声明自己的 sampling/conditioning unit；
3. 保存与真实 event map 对应的 margin、ranking representation 和 tie rule；
4. top-k 同时区分 order、set 和 membership-specific event；
5. 对 rare semantic event 做功效说明，零 disagreement 不直接等于 equivalence；
6. transition 报 relative geometry 和 task projection，不能只报 absolute update norm。

## 11. 下一步应该是什么

受控 pilot 已经完成它的任务，继续堆更多 toy states 的收益很低。下一步应将同一 schema 接到非异常筛选的真实 checkpoint/batch state bank：

- 固定一个真实 checkpoint cluster，先只重采样 batch/token states；
- 再增加多个 checkpoint/trajectory clusters，单独估计 parameter-state heterogeneity；
- 每个 semantic endpoint 保存对应 margin/tie/ranking 表示；
- 保留 fail-closed compiled execution canary；
- 先跑 deterministic profile，再把 algorithmic RNG 和 runtime nondeterminism 作为独立 protocol；
- 在整体 profile 可信后才开始 repair/injection operator attribution。

Operator analysis 目前尚未由本 pilot 验证；本轮只验证了它需要接入的 Oracle endpoint 和 measurement gate。

## 12. Artifacts

- Calibration report: [../results/oracle_pilot/deterministic_20260715_v1/REPORT.md](../results/oracle_pilot/deterministic_20260715_v1/REPORT.md)
- Calibration summary: [../results/oracle_pilot/deterministic_20260715_v1/summary.json](../results/oracle_pilot/deterministic_20260715_v1/summary.json)
- Confirmation report: [../results/oracle_pilot/confirmation_20260715_v1/REPORT.md](../results/oracle_pilot/confirmation_20260715_v1/REPORT.md)
- Confirmation summary: [../results/oracle_pilot/confirmation_20260715_v1/summary.json](../results/oracle_pilot/confirmation_20260715_v1/summary.json)
- Pilot implementation: [oracle_gpu_pilot.py](oracle_gpu_pilot.py)
- Analysis implementation: [analyze_oracle_gpu_pilot.py](analyze_oracle_gpu_pilot.py)
