# DSL v2 第十一个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 [increment_11.md](increment_11.md)。数字是开发证据，**不是盲测成绩**。
成绩只按三种状态写：成立 / 不成立 / 无法判断。设备验证：0（本机只有 sm_86，sm_90 / sm_100 路径只编译不运行，记为待做）。

## 1. 两层对照（评价运行）

官方主线下跑 test_core 的 dot、reduce1d、reduce2d、scan2d、where、cast 测试（2120 条通过）。对每次捕获的启动：TTIR 求值一次；
同一份 TTIR 用官方编译器编为 sm_90 和 sm_100 的 TTGIR（只编译），在同一组捕获输入上各求值一次，逐元素比较
（`results/dsl_v2/main_capture/inc11_cross_level*.jsonl`，插件 `scripts/dsl_v2/cross_level_plugin.py`）。

| | sm_90 | sm_100 |
| --- | --- | --- |
| 启动 | 2138 | 2138 |
| TTIR 与 TTGIR 都完整 | 2137 | 2137 |
| 两边都部分参照 | 1 | 1 |
| 两边都建立的元素 | 7,344,666 | 7,344,666 |
| 其中区间不相交 | **0** | **0** |
| 只在一边建立的元素 | 0 | 0 |
| 用到的目标操作（启动数） | warp_group_dot 283（及 wait、fence_async_shared） | tc_gen5_mma、tmem_alloc / load / store、init / wait / inval_barrier 各 287 |

第一次运行中，21 次启动在 TTGIR 一层中止：tf32 dot 前，下降插入了 inline asm `cvt.rna.tf32.f32`。TTIR 一层把 tf32 记为 dot 的精度属性，
在实数目标上不改变值。为了两层共用同一个目标，在数值差异模式下把这条转换按同一规则处理（记 `dot_input_precision:tf32`；
舍入核验模式仍记未建立）。修复后只复跑 tf32 用例：363/363 两层都完整，不相交 0。上表是合并后的结果。

## 2. 预期对照

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| 两层都完整的元素区间全部相交，不相交 0 | 两个目标各 7,344,666 个元素，不相交 0 | 成立 |
| sm_90 / sm_100 TTGIR 名字覆盖：除未登记的 TMA / warp specialization 类外全部覆盖 | W0 观察清单中出现在 sm_90 / sm_100 TTGIR 的 129 个操作，128 个有规则；剩下 `tc_gen5_mma_scaled`（5 次，写明未建模） | 成立 |
| 设备验证 0（待做） | 0 | 成立（按登记） |
| 逐签名测试三类齐全，违反 0；回归不变 | `test_signatures_nvidia.py` 10 条（wgmma、tcgen05 对精确有理点积，以及与 TTIR 参照相交）；全套通过 | 成立 |

## 3. 同一时期的其他结果

- W1：421/421 个支持条目三类测试齐全并有触发证据。复合规则（嵌套控制、store→load 别名、跨启动状态）在实际发生的位置记触发。
- 缺陷（W1 别名测试发现）：经由未建立地址的 store 原来只在启动结束时让目标缓冲失效，同一 program 之后的 load 仍把旧值当作已建立。
  现在立即失效，并对其他 program 的读按「写了每个元素」检查。

## 4. 没有做或不能说的

- 没有设备证据：sm_90 / sm_100 的结论都只是「CPU 参照与 TTIR 一层一致」，不是设备上的正确性。
- TMA（tensor descriptor 在 sm_90 上的异步拷贝）、warp specialization、`tc_gen5_mma_scaled` 不在本增量内。
