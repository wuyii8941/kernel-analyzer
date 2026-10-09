# DSL v2 第十个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 [increment_10.md](increment_10.md)（含实现前记录的两处偏离：共享存储按
编译器的 Membar 排序；异步拷贝掩码位置按官方测试零填充）。数字是开发证据，**不是盲测成绩**。成绩只按三种状态写：成立 / 不成立 / 无法判断。

## 1. 评价运行

官方主线下跑官方 Gluon 单元测试 `python/test/gluon/test_core.py`（sm_86 上 426 条通过，其余因需要 Hopper / Blackwell 跳过或失败），
捕获到的启动用增量 10 的求值器从 TTGIR 求值（`results/dsl_v2/main_capture/inc10_gluon_test_core.jsonl`）。

| 状态 | 增量 10 之前 | 增量 10 |
| --- | --- | --- |
| 启动总数 | 428 | 428 |
| 完整 | 0 | 358（83.6%） |
| 部分参照 | 0 | 60 |
| 中止 | 0 | 10 |
| 求值器错误 | 428（没有 TTIR） | 0 |

部分参照的原因全部写明：local atomic 被 mask 掉的 lane 不返回值（36，官方测试也只检查 final）；同一元素被几个 lane 同时更新、整数返回值是集合（12）；
`test_reduce_noncommutative` 的非交换 combine 让 warp 内各 lane 结果不一致（12）。中止的 10 次：逐线程 `ttg.inline_asm` 与 asm 内访问
共享存储（9，登记 F 项），`memdesc_reinterpret` 改元素类型（1）。

## 2. 预期对照

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| 求值器错误从 428 降到只剩新缺陷 | 0 | 成立 |
| 只含已有规则操作的 kernel 的启动大部分完整 | 是（未完整的都属下面几类，原因写明） | 成立 |
| 使用共享存储的启动大部分完整；竞争或未完成的异步拷贝报告为未建立并写明原因 | gather / scatter / subslice / reshape / 共享存储里存指针 / 异步拷贝加 mbarrier 都完整；local atomic 的未完整部分原因写明 | 成立 |
| 用到 F 项操作的启动报告为未建立并写明原因 | 9 次中止，原因写明 | 成立 |
| 427 个唯一 Gluon TTGIR 中按名字完整覆盖的从 222 升到至少 400 | 418/427；剩下的是逐线程 inline asm 6、asm 内访问共享存储 3、warp specialization 1 | 成立 |
| 逐签名测试三类齐全，违反 0；classic 回归不变 | `test_signatures_gluon.py` 13 条（官方 TTGIR，改动的变体写明）通过；全套通过 | 成立 |

## 3. 实现中发现并处理的问题

- 第一次复跑：local atomic 缺 `exch`；集合规则的原因没有传出来；共享存储里的指针丢了所属缓冲。都已修，再复跑得到上表。
- mbarrier：按 PTX 语义实现。没有 noIncrement 时，`async_copy_mbarrier_arrive` 是净零的到达，只让拷贝卡住当前阶段的完成；
  `wait_barrier` 的 parity 指当前阶段或前一阶段。按这个语义逐行走官方 kernel，拷贝恰好在最后一次 wait 时被确认完成。
- libdevice `__nv_fast_fdividef` 按其他 fast 变体的规则（同一数学函数）登记。

## 4. 没有做或不能说的

- 逐线程 inline asm、warp specialization、TMA、tensor memory、wgmma / tcgen05：sm_86 上不运行，没有设备证据；目标语义留在欠账里。
- 共享存储的程序序来自官方编译流水线（Membar），是前提，没有逐条核对 PTX 中的 barrier。
