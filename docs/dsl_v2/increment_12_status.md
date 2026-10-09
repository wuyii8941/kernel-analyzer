# DSL v2 第十二个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 [increment_12.md](increment_12.md)（含登记前选型运行中发现并修复的求值器缺陷）。
数字是开发证据，**不是盲测成绩**。成绩只按三种状态写：成立 / 不成立 / 无法判断。设备验证：0（本机没有 AMD 设备，gfx942 / gfx950
路径只跑官方 AMD 后端到 TTGIR 为止的阶段，不运行，记为待做）。

## 1. 两层对照（登记的评价运行）

官方主线下跑 test_core 的 dot、reduce1d、reduce2d、scan2d、where、cast（2120 条通过）。对每次捕获的启动：TTIR 求值一次；同一份 TTIR
经官方 AMD 阶段编为 gfx942、gfx950 的 TTGIR，在同一组捕获输入上各求值一次，逐元素比较
（`results/dsl_v2/main_capture/inc12_cross_level_amd.jsonl`，求值器为提交 `681beda`，插件 `scripts/dsl_v2/cross_level_plugin.py`，`KA_CROSS_TARGETS=gfx942,gfx950`）。

| | gfx942 | gfx950 |
| --- | --- | --- |
| 启动 | 2138 | 2138 |
| TTIR 与 TTGIR 都完整 | 2137 | 2137 |
| 两边都部分参照 | 1（`test_where[1-*int32]`：经 `tt.int_to_ptr` 的地址不在任何捕获存储内，两层原因相同） | 1（同左） |
| 两边都建立的元素 | 7,344,666 | 7,344,666 |
| 其中区间不相交 | **0** | **0** |
| 只在一边建立的元素 | 0 | 0 |
| 求值器错误 | 0 | 0 |
| 用到的目标操作（启动数） | buffer_load 659、buffer_store 298、in_thread_transpose 267、s.setprio / sched.barrier 9 | buffer_load 659、buffer_store 298 |

选型运行（登记前、修复前）中，这批启动有 926（gfx942）/ 659（gfx950）次在 TTGIR 一层中止，两层都建立的元素中不相交 1568 / 1522。

## 2. 预期对照

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| 两层都建立的元素区间全部相交，不相交 0 | 两个目标各 7,344,666 个元素，不相交 0 | 成立 |
| 选型运行中因 AMD 操作中止的启动（926 / 659）两层都完整；只剩两层都是部分参照的 1 次，原因与 TTIR 一层相同 | 2137/2138 两层都完整；剩下的 1 次是 `test_where[1-*int32]`，两层原因相同 | 成立 |
| 名字覆盖：gfx942、gfx950 从 922 / 956 升到 5949/5952，剩下 3 个与 TTIR 一层相同 | 两个目标都是 5949/5952；剩下的是 NVIDIA PTX inline asm 读环境（`%globaltimer`、`%smid`）2 个、asm 内访存 1 个 | 成立 |
| 逐签名测试 A–G 每项三类齐全，违反 0；回归不变 | `tests/test_signatures_amd.py` 29 条（官方 TTGIR fixture，见 `tests/data/amd_ttgir/README.md`）通过；全套 2930 通过、0 失败（另有 15 个旧测试文件因缺 transformers 收集失败，与之前相同）；W1 仍为 421/421 | 成立 |
| 设备验证 0（待做） | 0 | 成立（按登记） |

## 3. 未登记的补充运行

登记的子集只用到 buffer_load / store、in_thread_transpose 和调度提示。另外跑了用到其余操作的官方测试（同一插件、同两个目标），
只作补充证据，不作登记的结论：

| 运行 | 启动（每个目标） | 两层都完整 | 两层都部分参照 | 两层都建立的元素 | 不相交 | 用到的 AMD 操作（启动数） |
| --- | --- | --- | --- | --- | --- | --- |
| test_core 的 atomic_rmw、tensor_atomic_rmw、atomic_cas、argmax（`inc12_cross_level_amd_supplementary_core.jsonl`） | 454 | 182 | 272 | 7,457 | 0 | buffer_atomic_rmw 56、buffer_store 94、buffer_load 18 |
| test_tensor_descriptor 的 test_tma_gather_dot_pipeline（`..._supplementary_tensor_descriptor.jsonl`） | 1 | 1 | 0 | 256 | 0 | gfx950：buffer_load_to_local；gfx942：buffer_load、local_store |

两个目标数字相同，求值器错误 0，只在一边建立的元素 0。部分参照的原因两层相同：整数原子返回值是集合（240）、同一地址混用原子种类（20）、
CAS 测试中跨 program 的读写竞争（4）、带 NaN 的 argmax 在 warp 内各 lane 结果不一致（8）。`test_scaled_dot` 需要 capability ≥ 9，
本机跳过，所以 `scaled_upcast_fp4 / fp8` 只有逐签名测试（官方 fixture）的证据；`buffer_atomic_cas` 在这批测试的 AMD 编译中没有出现，
同样只有逐签名测试。

## 4. 逐签名测试的内容

每项都在官方 TTGIR fixture 上求值，与测试里独立算出的精确有理数比较，并与同一 kernel 的 TTIR 参照逐元素对照（都完整处区间相交）：

| 项 | fixture | 正例 / 边界 / 前提违反 |
| --- | --- | --- |
| A buffer_load（mask、other） | argmax_masked_other | 被 mask 的 lane 取 other；−inf 与被 mask 的 +inf；地址在捕获之外、去掉 other 的变体（被 mask 的 lane 未定义）不完整 |
| B buffer_store（mask） | masked_copy_bf16 | bf16 复制；最大有限值、−0、mask 掉末元素保持原值；读到捕获之外的 lane 不完整 |
| C buffer_atomic_rmw / cas | atomic_fadd_pairs、atomic_cas_rows | 同址两次 fadd 的精确和、±3e38 相消；NaN 更新；CAS 按行交换、比较不等；返回值缺失时由它寻址的 store 使目标缓冲全部未建立 |
| D buffer_load_to_local | gather_dot_pipeline（gfx950） | 流水线 dot 精确；去掉循环内 `ttg.async_wait` 的变体读到未确认完成的拷贝，不完整 |
| E in_thread_transpose | simple_dot_transpose | dot 精确；65504 与 1/1024；NaN 所在列不完整 |
| F scaled_upcast_fp8 / fp4 | simple_dot_mxfp8、mxfp8_mxfp4_matmul | MX 点积精确；E8M0 = 0（bf16 载体为次正规数 2^−127）与 254；E8M0 = 255（NaN 标记）所在行 / 列不完整 |
| G s.setprio / sched.barrier | matmul_pipelined | 4 次流水线迭代中执行提示，dot 精确；f16 最大值；inf 所在行不完整 |

另有两条测试针对登记前修复的缺陷：中止的启动里，kernel 以相同字节重写的输出不再保留为已建立参照（只读的输入仍保持）；
静态地址流分析在地址经 `tt.int_to_ptr` 逃逸时把所有参数算作可能写。

## 5. 实现中的说明

- `buffer_load` 没有 `other` 时，被 mask 的 lane 与 `tt.load` 一样未定义（登记 A 项）；`buffer_load_to_local` 沿用增量 10 的异步拷贝
  模型，被 mask 的 lane 按增量 10 已记录的偏离零填充（登记 D 项「同一异步拷贝模型」）。
- E8M0 scale 按 bf16 载体位型的指数域解码。编译器的载体是 `max(e << 7, 64)`：e = 0 时得到 bf16 次正规数 2^−127，指数域仍为 0，
  与 2^(e−127) 一致。
- 逐签名测试中一个 fixture 的生成方式出了错（最初用 `strip_locations` 去掉注释时也删掉了布局别名，归约因此拿不到 TTGIR 次序）；
  改为只去掉 `#loc` 行与 `loc(...)` 后缀。不影响求值器与评价运行。

## 6. 没有做或不能说的

- 没有设备证据：gfx942 / gfx950 的结论都只是「CPU 参照与 TTIR 一层一致」，不是设备上的正确性。官方 AMD 后端的 LLVM 代码生成在本机
  不能运行，TTGIR 之后的下降（包括 buffer 操作越界返回 0 的硬件语义）没有核对。
- TDM、AMD 的 mbarrier / async_wait、`masked_load` / `masked_store`、`scaled_downcast_*`、`local_load_packed_transposed`、
  `extract_slice` / `concat` 等其他官方 AMD 操作在这批官方 TTIR 的 gfx942 / gfx950 编译中没有出现，留在欠账表；RDNA（gfx11 / gfx12）
  与 gfx1250 没有编译。
- AMD Gluon 子模块没有在本机运行（需要 AMD 设备）。
