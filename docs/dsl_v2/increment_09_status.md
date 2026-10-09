# DSL v2 第九个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 [increment_09.md](increment_09.md)。数字是开发证据，**不是盲测成绩**。
成绩只按三种状态写：成立 / 不成立 / 无法判断。

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| 5952 个唯一主线 TTIR 中按名字覆盖的从 5945 升到 5949，剩下 3 个写明原因 | 5949/5952；剩下 2 次环境读取（`%globaltimer`、`%smid`）和 1 次 asm 内访存，原因都写在覆盖报告里 | 成立 |
| 官方 inline asm 测试：shf、packed、multiple_outputs、packed_multiple_outputs 完整；with_pointers 中止并写明原因 | 4 个完整；with_pointers 中止，被改动的元素记未建立 | 成立 |
| `test_dot_max_num_imprecise_acc` 的 fp8e4b15 转换可求值；整个启动是否完整不预设 | 48/48 次启动完整（增量 6 时 24 次中止） | 成立 |
| `test_typeconvert_upcast` 的 2 次中止变为可求值 | 28/28 完整 | 成立 |
| 逐签名测试三类齐全，违反 0；回归不变 | `test_signatures_ptx_bits.py` 17 条通过：fp8e4b15 → fp16 转换按格式定义独立解码核对；全套通过 | 成立 |

结果：`results/dsl_v2/main_capture/inc9_inline_asm.jsonl`（81 次启动，80 完整，1 中止）。

没有做的：asm 内访存（ld / st / atom）、f16x2 算术；时钟与 SM 编号是环境观察，按 rc3 W5 保留为没有参照值的可观察效果。
