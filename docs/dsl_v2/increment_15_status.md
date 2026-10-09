# DSL v2 第十五个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 [increment_15.md](increment_15.md)（第 5 节是实现中、评价运行前的更正：
整数表示缺陷实际在 CAS 路径，load 本来就换成有符号表示）。数字是开发证据，**不是盲测成绩**。成绩只按三种状态写：成立 / 不成立 / 无法判断。

## 1. 评价运行

增量 6 的全部官方测试文件在增量 15 的求值器下复跑（`results/dsl_v2/main_capture/inc15_broad.jsonl`），逐启动与增量 14 之后的状态运行
（`inc14_broad.jsonl`）比较（`inc15_compare_broad_vs_inc14.json`）。

| | 状态运行（增量 14 后） | 增量 15 |
| --- | --- | --- |
| 启动 | 12,766 | 12,766 |
| 完整 | 12,246 | 12,257 |
| 完整（集合目标） | 260 | 260 |
| 部分参照 | 126 | 118 |
| 中止 | 5 | 2 |
| 求值器错误 | 0 | 0 |

状态改变的启动只有 12 次：`test_bin_op` 中 float % int64 / uint64 的 8 次（部分 → 完整）；`test_libdevice_rint[float64]`、`test_clz`、
`test_popc`（中止 → 完整）；int8 `%` 那次仍是部分参照，现在写明「integer division by zero or INT_MIN / -1 (undefined behavior)」。
剩下的 2 次中止是 `test_inline_asm_with_pointers`（asm 内访存）和 `test_constexpr_if_return`（原子返回值作为 `cf.cond_br` 条件）。

## 2. 预期对照

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| `test_bin_op` 的 8 次 `remf` 部分参照变为完整 | 8/8 | 成立 |
| `test_libdevice_rint[float64]`、`test_clz`、`test_popc` 变为完整 | 3/3 | 成立 |
| int8 `%` 那次仍为部分参照，但写明原因 | 是 | 成立 |
| A 的复跑中没有原来完整的启动变为部分或中止 | 0 次 | 成立 |
| 若有启动的值因 A 改变，逐个列出 | 捕获记录只保存状态，不保存参照值，无法逐个比较值；使用 CAS 的启动状态都没有改变。A 对值的影响由逐签名测试证实（见下） | 无法判断 |
| 逐签名测试违反 0；W1 支持条目增加后仍全部三类齐全并有触发证据 | `tests/test_signatures_inc15.py` 40 条（及新 libdevice 条目由已有生成器得到的 6 条）通过；W1 430/430（新增 `math.ctlz / cttz / ctpop` 与 6 个 libdevice 条目）；全套 3040 通过、0 失败 | 成立 |

## 3. 缺陷与测试

- CAS：修复前，int64 的 2^62 + 1 与比较值 2^62 在 float64 中相等，参照会记一次并不存在的交换；uint32 存储中 0xFFFFFFFE 与比较值 −2
  （同一位型）永远不等，参照会漏掉交换；返回的旧值经过 float64，大 int64 丢精度。三条 CAS 测试在修复前都失败，修复后通过。
- 整数未定义行为（除零、INT_MIN / −1、移位越界）原来记未建立但不写原因；现在写明。
- `test_signed_operations_on_unsigned_storage` 在修复前后都通过：它证实 load 路径本来正确，也就是第 5 节更正的依据。

## 4. 没有做或不能说的

- bessel_y0 / y1 的负数输入与 `remf` 除数为 0：按类别 D 的既定规则记未建立。改成 IEEE 的 NaN / −inf 是语义改动，需审阅方决定。
- 浮点 atomic max / min 的位型下降、同址混用原子种类、argmax 带 NaN 的 lane 不一致、同一 program 内两次原子返回值的大小关系、
  原子返回值作为 `cf.cond_br` 条件、asm 内访存：未处理（原因见登记第 1 节）。
