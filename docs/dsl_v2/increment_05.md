# DSL v2 第五个增量：登记（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中）。对应 rc3 正式版 04 的 W4（同步与执行顺序）和 W5（近似指令、直方图），依据 02 §5.3、§6.3、
§6.5、§6.8、§6.9。本登记写于实现之前。

## 1. 选这几组的理由

官方主线 `e50b186e` 跑完官方单元测试后（test_core 全部 8584 条，另有 compile_only），转储出 4867 个唯一 TTIR。
按操作名逐条对照映射表，4717 个（96.9%）完整覆盖。挡住其余 150 个的只有这几项（`main_ttir_coverage.py`，一个 kernel 可能同时被几项挡住）：

| 名字 | kernel 数 |
| --- | --- |
| `tt.atomic_load`、`tt.atomic_store`（主线新增） | 44、44 |
| inline asm `div.full.f32` | 36 |
| `tt.atomic_poll`（主线新增） | 29 |
| `tt.reduce` 的通用 combine 区域 | 20 |
| `tt.histogram` | 18 |
| `tt.approx_divf`（主线新增） | 2 |
| 解析失败：返回多个值的函数头 `-> (f32, f32)` | 1（登记前已修，见 §5） |

通用 combine 区域在求值器里已经按 TTGIR 布局的下降次序求值（增量 1，声明前提），只是覆盖统计没有反映。这一项只改统计口径，
不算新语义。

## 2. 目标签名与做法

| 项 | 签名 | 做法 |
| --- | --- | --- |
| A1 | `tt.atomic_store` | 当作返回值不用的 exchange。地址无竞争时为点；有竞争时，最终值是全部候选值的凸包（L_E），整数记未建立。与其他 program 的非原子访问冲突时按竞争处理 |
| A2 | `tt.atomic_load` | 进入两遍求值。地址在本次启动里没有其他原子更新：等于普通 load，为点。有原子更新：取「启动前的值 ⊕ 这些更新的任意子集」的凸包（同增量 4 的 `_atomic_old`，exchange/store 取候选值凸包），为 L_E。第一遍以及超出预算时记未建立。原子读不登记为普通读，所以原子读与原子写之间不算竞争 |
| B1 | `tt.atomic_poll` 的结果 | 第一遍记录每个 poll 的地址和期望值。第二遍：地址上没有其他 program 的写入时，结果是「启动前的值 == 期望值」；无超时又不相等，参照上不会终止，记未建立。有其他 program 写入期望值：无超时时结果为真，记假设「poll 会终止（公平调度，写入方能推进）」，这是 rc3 §6.9 要求写明的终止条件；有超时时结果为集合 {假, 真}，用三态布尔的 MAYBE 表示。地址上有累加类原子更新时记未建立 |
| B2 | happens-before | 第二遍按 poll 依赖重排 program 的求值次序：写入期望值的 program 先于 poll 的 program，同号保持原序。有环则环上的 poll 记未建立（无法证明推进）。poll 以 acquire 成功、且期望值只来自一个 program 的 release（atomic_store release、默认 acq_rel 的原子更新）时，记一条 happens-before 边：之后的普通 load 读到那个 program 在 release 之前的写入，不算竞争。只记直接边，不做传递闭包（漏边只会多报竞争，不会少报） |
| C | inline asm `div.full.f32`、`tt.approx_divf` | rc3 §6.5：数值差异模式下，有文档目标函数的近似指令提升为目标函数，这里是实数除法 x / y（与 `arith.divf` 同一条 π 规则）。舍入核验模式下没有实现误差契约，记未建立 |
| D | `tt.histogram` | 计数精确。官方定义只说 bin 宽 1、从 0 开始，没有说明越界值怎么处理，所以逐样本检查：未被 mask 的值都在 [0, bin 数) 内才建立，否则记未建立（越界行为未定义）。mask 为假的元素不计数 |
| E | 覆盖统计 | `kernel_coverage` 对通用 combine 区域报告「沿 TTGIR 下降次序（声明前提）」，不再算拒绝 |

不在本增量内：poll 的传递 happens-before；CAS 自旋锁的协议摘要；atomic_load 与 release/acquire 链结合后的精确值（现在取集合）。

## 3. 预算

A、B 约 1 天；C、D、E 约半天；捕获复跑约 30 min。

## 4. 预期（实现前写定）

- 4867 个唯一 TTIR 中，按名字完整覆盖的从 4717 升到至少 4860。剩下的预计是带 inline asm 的其他片段，或同一 kernel 里的未实现组合。
- 捕获复跑（`test_atomic_load_store*`、`test_atomic_poll*`、`test_histogram`、带 `div_rn` / 近似除法的测试）：
  - 单 program 的 atomic load/store 与 poll：参照完整（点）；
  - `test_atomic_poll_waits_for_remote_cta`：out 完整，等于 42，记假设「poll 会终止」；payload 的读不报竞争；
  - 带超时、期望值不在内存里的 poll：结果完整（假）；
  - histogram：输入都在范围内时完整。
- 逐签名测试：每项有正例、边界、违反前提三类；包含性违反预计为 0。
- 回归：已有全部测试不变。

## 5. 登记前已经做的修正

- 解析器：函数头的结果是带括号的列表时，原来的正则会把结果类型吞进参数表，报 `bad parameter 'f32'`。改为按配对括号截取参数表。
  这是在官方主线转储上发现的缺陷，不改变任何语义。
