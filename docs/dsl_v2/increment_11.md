# DSL v2 第十一个增量：登记（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中）。对应 rc3 正式版 04 的 W3（NVIDIA 目标模块：矩阵路径、tensor memory、mbarrier；
「没有对应设备时可先完成源契约、CPU 参照及独立模型测试，明确标 device_validation = pending」）。本登记写于实现之前。

## 1. 选这组的理由

W0 观察清单把 5952 个官方主线 TTIR 重新编译为 sm_90 与 sm_100。TTGIR 中出现、还没有规则的操作只有这几个
（括号内是 sm_90 / sm_100 的出现次数）：

| 操作 | 次数 |
| --- | --- |
| `ttng.warp_group_dot`、`ttng.warp_group_dot_wait`（Hopper wgmma） | 48 / 0、46 / 0 |
| `ttng.fence_async_shared` | 48 / 63 |
| `ttng.tmem_alloc`、`tmem_load`、`tmem_store`（Blackwell tensor memory） | 0 / 63 |
| `ttng.tc_gen5_mma`、`tc_gen5_mma_scaled` | 0 / 58、0 / 5 |
| `ttng.inval_barrier` | 0 / 63 |
| `ttg.memdesc_trans` | 5 / 9 |
| `ttg.fp4_to_fp` | 4 / 0 |

本机只有 sm_86，这些路径不能在设备上运行；按 rc3 先给出 CPU 参照与独立测试，设备验证记为待做。

## 2. 做法

| 项 | 内容 |
| --- | --- |
| A | `warp_group_dot`：d = a·b + c（a 可以是寄存器张量或共享存储，b 是共享存储），与 `tt.dot` 同一精确实数点积规则，记录 inputPrecision 与 maxNumImpreciseAcc；`warp_group_dot_wait` 对值是恒等（异步完成由它的使用点保证） |
| B | `fence_async_shared`：通用代理与异步代理之间的栅栏；在程序序模型里没有值效应，记前提。`inval_barrier`：mbarrier 失效，之后再用记未建立 |
| C | tensor memory：与共享存储同一模型（每个 program 一份，按程序序），`tmem_alloc`（可带初值）、`tmem_store`（带谓词）、`tmem_load` |
| D | `tc_gen5_mma`：D(tmem) = A·B + (useD ? D : 0)，谓词为假时不更新；完成通过参数中的 mbarrier 到达（与增量 10 的阶段模型一致），未确认完成前读 D 记未建立。`tc_gen5_mma_scaled`：按增量 3 的 MX 精确解码后同样处理 |
| E | `memdesc_trans`：视图转置；`fp4_to_fp`：fp4（e2m1，两个一字节，低半字节在前）精确解码 |
| F | 设备验证：全部记为待做（device_validation = pending） |

## 3. 评价设计（实现前写定）

- **同一 kernel 两层对照**：从官方主线转储里选取若干 classic kernel（含 dot、reduce、scan、shared memory 流水线），各自的 TTIR
  用官方主线编为 sm_90 和 sm_100 的 TTGIR（只编译，不运行）。同一组随机输入下，TTIR 参照与两份 TTGIR 参照逐元素比较：
  都完整的元素区间必须相交（实数语义相同，区间外扩各自不同）；任何一方未完整的元素单独计数。
- 名字覆盖：sm_90 / sm_100 重新编译的 TTGIR 中，按名字完整覆盖的比例。
- 逐签名测试三类齐全，违反 0；回归不变。

## 4. 预期

- 两层对照：选取的 kernel 中，TTIR 与 sm_90 / sm_100 TTGIR 参照都完整的元素，区间全部相交；不相交的个数预计为 0。若出现，
  说明某个 TTGIR 规则或 TTIR→TTGIR 的语义对应有缺陷，记录并修复。
- sm_90 / sm_100 TTGIR 名字覆盖：除尚未登记的 TMA / warp specialization 类操作外全部覆盖。
- 设备验证：0（待做）。
