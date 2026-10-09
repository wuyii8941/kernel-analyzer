# DSL v2 第十个增量：登记（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中）。对应 rc3 正式版 04 的 W3（classic 与 Gluon 各设导入器、共用语义内核；布局与共享存储先行）
和 W4（异步拷贝的发起、分组与完成）。依据 02 §6.3、§6.9。本登记写于实现之前。

## 1. 选这组的理由

- 官方主线下跑官方 Gluon 单元测试（`python/test/gluon/test_core.py`），捕获到 428 次启动，**全部**是求值器错误：Gluon kernel 没有 TTIR，
  只有 TTGIR，现在的捕获和求值只读 TTIR。
- 同一组测试的 TTGIR 转储有 427 个唯一 kernel，全部能被现有解析器解析。按名字：
  - 222 个（52%）只含已有规则的操作，问题只在读不到 TTIR；
  - 其余要共享存储：`ttg.local_alloc` 196、`local_store` 172、`local_load` 157、`memdesc_subslice` 58、`local_atomic_scatter_rmw` 77、
    `local_gather` 36、`local_scatter` 29、`memdesc_reshape` 10、`memdesc_reinterpret` 1；
  - 异步拷贝与 mbarrier：`async_copy_global_to_local` 15，`ttng.init_barrier` / `arrive_barrier` / `wait_barrier` /
    `async_copy_mbarrier_arrive` 各 15；
  - 其他：`ttg.convert_layout` 5，逐线程的 `ttg.inline_asm` 6，`ttg.warp_specialize` 1。

## 2. 做法

| 项 | 内容 |
| --- | --- |
| A 导入 | 启动没有 TTIR 时，从 TTGIR 求值（同一个语义内核）；classic 启动不变。类型解析保留布局编码；`!ttg.memdesc<…>` 解析为共享存储描述（形状、元素、编码、是否可变） |
| B 布局与共享存储 | `ttg.convert_layout` 对值是恒等。共享存储按 program 建立（每个 program 一份），memdesc 值是指向它的元素下标视图。`local_alloc`（可带初值）、`local_store`、`local_load`、`local_dealloc`；视图操作 `memdesc_subslice`、`memdesc_index`、`memdesc_reshape`；`memdesc_reinterpret` 只接受元素类型不变，否则记未建立 |
| C 共享存储的线程冲突 | 与全局内存同一规则：写入线程由存入张量的布局给出，读取线程由读出张量的布局给出；同一 barrier 阶段内被另一个线程读到刚写入的元素，记执行竞争；布局映射未知时记未建立，不默认安全 |
| D 下标访问 | `local_gather` / `local_scatter` 沿指定轴按下标读写，逐元素检查下标有效；`local_atomic_scatter_rmw` 按增量 4 的规则折叠可交换的更新，返回值按增量 4 的集合 / 点规则 |
| E 异步拷贝 | `async_copy_global_to_local` 在发起时按值写入目标视图，但这些元素标为「未完成」，直到覆盖它的 `async_wait`（按 group 计数）或 mbarrier 等待完成；未完成时被读到记未建立。`ttng.init_barrier` / `arrive_barrier` / `wait_barrier` / `async_copy_mbarrier_arrive` 按阶段计数；等不到的阶段（程序内无法完成）记未建立，不挂起 |
| F 不在本增量内 | 逐线程 `ttg.inline_asm`、`warp_specialize`、TMA、tensor memory、wgmma / tcgen05 等 Hopper / Blackwell 操作（sm_86 上不运行）：记未建立并写明原因 |

## 3. 预算

约 1 天；捕获复跑约 30 min。

## 4. 预期（实现前写定）

- 官方 Gluon test_core 的捕获（sm_86）：
  - 求值器错误从 428 降到只剩尚未识别的新缺陷（若出现，记录并修复）；
  - 只含已有规则操作的 kernel 的启动大部分完整；
  - 使用共享存储的启动大部分完整；有执行竞争或未完成异步拷贝的报告为未建立并写明原因；
  - 用到 F 项操作的启动报告为未建立并写明原因。
- 按名字覆盖：427 个唯一 Gluon TTGIR 中完整覆盖的从 222 升到至少 400。
- 逐签名测试三类齐全，违反 0；classic 回归不变。

## 5. 偏离（实现前记录，2026-10-09）

- **C 改了。** 登记时打算对共享存储做与全局内存相同的线程级冲突检查。实现前查官方编译流水线：TTGIR 之后的 `make_llir` 先分配共享存储，
  再跑 Membar 分析插入 barrier（`triton/backends/nvidia/compiler.py:437-442`，`add_allocate_shared_memory_nv`、`add_membar`），
  classic 与 Gluon 都走这一步。也就是说，转储的 TTGIR 本来就不含这些 barrier；官方契约是同一个 program 的共享存储访问按程序序生效，
  由编译器保证。照登记做会把官方保证有序的访问误报为竞争。
- 改为：一个 program 的共享存储访问按程序序求值，记前提「同一 program 的共享存储访问由编译器的 Membar 分析排序」。
  这个前提来自官方源，不是对 PTX 的逐条核对。Membar 不等待异步拷贝，所以 E（异步拷贝未完成时读取记未建立）照原计划做。
- **E 的补充（实现中、复跑前记录）。** 异步拷贝的掩码：官方 op 定义说其余操作数「与 `tt.load` 相同」，按 `tt.load` 无 `other`
  的掩码位置是未定义。但官方测试 `test_async_copy_mbarrier`（`python/test/gluon/test_core.py:1254-1261`）在初值为 7 的共享存储上
  断言这些位置为 0，即 cp.async 的 src-size 0 下降是零填充。按官方测试零填充，并在结果里记前提，写明来源。
