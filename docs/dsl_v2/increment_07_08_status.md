# DSL v2 第七、八个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 [increment_07.md](increment_07.md)、[increment_08.md](increment_08.md)。
数字是开发证据，**不是盲测成绩**。成绩只按三种状态写：成立 / 不成立 / 无法判断。

## 1. 增量 7（CAS 自旋锁：串行化 + 逆序证据，向量时钟 happens-before）

官方主线下复跑 `test_atomic_cas` 与 `test_atomic_rmw`（`.cache/dsl_v2/w7_capture.jsonl`，归档见 §3）。

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| serialized_add：sem 为默认（acq_rel）或 acquire 时 data 完整，记声明前提 | int32、int64 各 3 种 sem（默认、acquire、acq_rel）共 6 次启动完整，每次 2000 个 program；记了声明前提；happens-before 边各 3998 条 | 成立 |
| serialized_add：sem 为 relaxed 或 release 时报竞争、参照未建立 | 4 次启动部分参照，原因「load 上的跨 program 竞争」 | 成立 |
| change_value（单 program）完整 | 10/10 完整 | 成立 |
| 教程 05 `_layer_norm_bwd_dx_fused`：DX、DW、DB、Lock、Count 完整，记声明前提 | 正在复跑（两种次序、1151 个 program，求值约 30 min） | 无法判断（未跑完） |
| 逐签名测试：锁内累加完整；先到者写入未建立；relaxed 报竞争；不释放的锁不挂起 | `test_signatures_locks.py` 8 条通过 | 成立 |
| 回归不变 | 全套通过（合并后在主检出跑） | 成立 |

声明前提的范围：程序序与逆序两种串行化一致，只是证据，不是对所有串行化的证明。审阅方核对后才能改成已验证。

## 2. 增量 8（W1 证据）

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| 三类齐全且有触发的条目从 335/418 升到至少 400/418 | 416/419（注册表多了一条 CAS 串行化的声明前提条目） | 成立 |
| 剩下的是复合规则或需要设备行为的条目 | 剩 3 条：嵌套 scf.for / scf.if、load / store 别名、跨启动状态。它们是复合规则，没有单一的追踪签名 | 成立 |
| 补测试时发现的缺陷记录并修复 | 3 处缺陷：`ub.poison` 无操作数时解析不出结果类型（崩溃）；gather 一个越界下标让整个 program 中止（改为逐 lane）；负底数加整数指数的 pow 未建立（改为按单调段精确计算）。2 处可靠性缺口：`math.clampf` 沿用了 `tt.clampf` 忽略 NaN 的规则，但它的 maxf / minf 对 NaN 的行为没有核实（NaN 与 min > max 现在未建立）；通用 scan 默认 combine 满足结合律（现在用第二种括号方式在实际输入上核对，不一致的前缀未建立，前提写明） | 成立 |
| 回归不变 | 全套通过 | 成立 |

## 3. 结果文件

- W1：`results/dsl_v2/contracts/coverage.json`（`.cache/dsl_v2/w1_final/` 的带追踪全套测试）。
- CAS 捕获：`.cache/dsl_v2/w7_capture.jsonl`；教程 05 跑完后与之一起归档到 `results/dsl_v2/main_capture/`。
