# DSL v2 第十二个增量：登记（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中）。对应 rc3 正式版 04 的 W3（「NVIDIA/AMD 目标模块并行填写语义」；「AMD 覆盖 buffer/LDS、MFMA/WMMA、
块缩放、TDM/异步及 pipeline」）。本机没有 AMD 设备：先给出 CPU 参照与独立测试，设备验证记为待做（device_validation = pending）。
本登记写于实现这些操作的语义之前。

## 1. 选这组的理由

官方 AMD 后端的 LLVM 代码生成需要比本机更新的 glibc，`triton.compile` 对 hip 目标走不完；到 TTGIR 为止的各阶段是官方 pass
流水线，可以原样运行（`scripts/dsl_v2/amd_stages.py`：`make_backend` + `add_stages`，只跑 `ttir`、`ttgir` 两步）。

W0 观察清单的 5952 个官方主线 TTIR 用这个办法编为 gfx942（CDNA3）和 gfx950（CDNA4）的 TTGIR（失败 0）。TTGIR 中出现、
还没有规则的操作只有下面 10 个（括号内是 gfx942 / gfx950 上出现该操作的唯一 kernel 数）：

| 操作 | 次数 |
| --- | --- |
| `amdg.buffer_load`、`amdg.buffer_store` | 4618 / 4617、2859 / 2859 |
| `amdg.buffer_atomic_rmw`、`amdg.buffer_atomic_cas` | 56 / 56、2 / 2 |
| `amdg.buffer_load_to_local` | 0 / 1 |
| `amdg.in_thread_transpose` | 28 / 0 |
| `amdg.scaled_upcast_fp4`、`amdg.scaled_upcast_fp8` | 4 / 0、5 / 0 |
| `rocdl.s.setprio`、`rocdl.sched.barrier` | 7 / 0、7 / 0 |

按名字完整覆盖的 kernel（实现前）：gfx942 922/5952，gfx950 956/5952（`ttir_mapping.kernel_coverage`）。

MFMA / WMMA 在 TTGIR 这一层仍是带 `#mma` 编码的 `tt.dot`，已有规则；TDM、AMD 的 mbarrier / async_wait 在这批官方 TTIR 的
gfx942 / gfx950 编译中没有出现，不在本增量内（留在欠账表）。

## 2. 实现前的发现（登记前的选型运行）

为了选操作，先在官方主线下跑了 test_core 的 dot、reduce1d、reduce2d、scan2d、where、cast，把每次捕获的启动的 TTIR 与 gfx942 /
gfx950 TTGIR 两层求值对照（与增量 11 同一插件，`KA_CROSS_TARGETS=gfx942,gfx950`）。TTGIR 一层因上表操作中止的启动，两层
都建立的元素中出现了区间不相交（gfx942 1568，gfx950 1522；全部是「TTIR 完整、TTGIR 中止」的组合）。

原因是求值器的缺陷，不是 AMD 语义：中止的启动里，参照没写到、而 kernel 写了相同字节（输出与运行前相同，例如 bf16 舍入
测试、`add-matrix` 的 dot、cumprod 下溢为 0）的元素，保留了捕获值，被当作已建立的参照。实数参照不是 kernel 的舍入结果，
所以与另一层的区间不相交。修复（登记前完成）：

- 中止的启动中，kernel 可能写的缓冲里，参照没写到的元素一律未建立。「可能写」= 运行前后有字节变化，或参照已写过其中元素，
  或静态地址流分析表明某个全局写操作的地址可能来自该参数（`ttir_eval._may_write_params`：除内存读以外的操作都传递来源，
  写操作、call、inline asm 的操作数带来源即算可能写；地址经内存逃逸——`tt.int_to_ptr` 或读出指针——时所有参数都算可能写）。
- 修复后只复跑这些不相交的用例：不相交 0。登记的评价是修复后的全新运行（下面第 4 节）。

## 3. 做法

依据官方 `third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td`（提交 e50b186e8bd2）及其下降代码。

| 项 | 内容 |
| --- | --- |
| A | `buffer_load ptr[offsets], mask?, other?`：等同于在地址 `ptr + offsets`（offsets 为 i32 元素偏移，下降时乘元素字节数）上的 `tt.load`，带 mask / other。没有 `other` 时被 mask 掉的 lane 与 `tt.load` 一样未定义（官方下降给 0，但 IR 描述没有规定，不依赖它）。`stride`、`cachePolicy`、`contiguity` 是性能提示，没有值效应 |
| B | `buffer_store value, ptr[offsets], mask?`：等同于同一地址上的 `tt.store` |
| C | `buffer_atomic_rmw kind, sem, scope, value, ptr[offsets], mask?`、`buffer_atomic_cas sem, scope, cmp, val, ptr[offsets]`：等同于同一地址上的 `tt.atomic_rmw` / `tt.atomic_cas`，沿用增量 4 / 7 的原子规则（集合目标、两遍求值、CAS 前提） |
| D | `buffer_load_to_local ptr[offsets] mask = m other = o into dest`：与 `ttg.async_copy_global_to_local` 同一异步拷贝模型（增量 10），地址按 A 项 |
| E | `in_thread_transpose`：线程内寄存器布局变换，是 `ttg.convert_layout` 的特例（官方描述），值恒等 |
| F | `scaled_upcast_fp4 x scale s {axis}`：e2m1 精确解码（低半字节在前，沿 axis，与 `ttg.fp4_to_fp` 相同）后乘 2^(e−127)；`scaled_upcast_fp8`：e4m3fn / e5m2 精确解码后乘 2^(e−127)。scale 是「以 BF16 编码的 E8M0」：e 取 bf16 位型的指数域（i8 时取字节值）。e = 255（NaN 标记）的位置记未建立（官方随后用 `maskNan` 选回 NaN）。结果格式（f16 / bf16）的舍入只在舍入核验模式中检查，与其他转换相同 |
| G | `rocdl.s.setprio`（wave 优先级）、`rocdl.sched.barrier`（编译器调度屏障）：调度提示，没有值或内存效应 |
| H | 设备验证：全部记为待做 |

AMD 的 buffer 操作在硬件上越界读返回 0、越界写丢弃；参照不使用这一点：落在捕获窗口之外的地址照旧记未建立。

## 4. 评价设计（实现前写定）

- **两层对照**：与增量 11 相同的官方 test_core 子集（dot、reduce1d、reduce2d、scan2d、where、cast），每次捕获的启动：TTIR
  参照一次，同一份 TTIR 经官方 AMD 阶段编为 gfx942、gfx950 的 TTGIR 各求值一次，逐元素比较。都建立的元素区间必须相交。
- **名字覆盖**：5952 个官方 TTIR 的 gfx942 / gfx950 TTGIR，按名字完整覆盖的 kernel 数。
- 逐签名测试：A–G 每项三类（正例、边界、前提违反）齐全，违反 0；回归不变。

## 5. 预期

- 两层对照：两层都建立的元素区间全部相交，不相交 0。选型运行中因上表操作中止的启动（gfx942 926，gfx950 659）两层都完整；
  只剩选型运行中两层都是部分参照的 1 次（各目标），原因与 TTIR 一层相同。若出现不相交，说明某条 AMD 规则或求值器有缺陷，记录并修复。
- 名字覆盖：gfx942、gfx950 都从实现前的 922 / 956 升到 5949/5952，剩下的 3 个与 TTIR 一层相同（NVIDIA PTX inline asm 读环境 2 个、
  asm 内访存 1 个）。
- 设备验证：0（待做）。
