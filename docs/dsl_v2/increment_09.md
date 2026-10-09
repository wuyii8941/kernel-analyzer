# DSL v2 第九个增量：登记（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中）。对应 rc3 正式版 04 的 W5（逐指令 inline asm 解析，区分 constraints、pack、寄存器映射、多个输出、
副作用；02 §6.10）。

**时间说明**：解释器与逐签名测试在登记之前实现（`ptx_bits.py`、`tests/test_signatures_ptx_bits.py`），这里如实写明。
评价运行（§4，相关测试族在官方主线下的捕获复跑）在登记提交之后才开始，预期写在运行之前。

## 1. 选这组的理由

- 增量 6 的全量捕获中，`test_dot_max_num_imprecise_acc` 有 24 次启动中止，`test_typeconvert_upcast` 有 2 次，原因都是「inline asm 没有声明语义」。
  这些 asm 是官方 NVIDIA 后端的 fp8e4b15 转换（`triton/language/extra/cuda/utils.py`）：位运算 PTX，`pack=4`，两个输出寄存器。
- 主线 TTIR 按名字覆盖剩下的 7 个都是 inline asm：读时钟、读 SM 编号、asm 内的全局 load/store、打包的位运算、多输出、
  漏斗移位、字节解包加整数转浮点。

## 2. 做法

| 项 | 内容 |
| --- | --- |
| A 解释器 | 寄存器是按元素组展开的数组，存位模式（带「位确定」掩码）或实数值（区间）。需要时互相转换：位到实数按格式精确解码；实数到位只对可精确表示的点（与 `tt.bitcast` 同一政策） |
| B 指令 | mov（含向量打包 / 解包）、and / or / xor / not、shl / shr（逻辑与算术）、add / sub / mul.lo（回绕）、prmt.b32（默认模式）、lop3.b32、shf.{l,r}.{wrap,clamp}.b32、selp、整数与 f16x2 的 setp、整数宽度之间及整数到 f32 的 cvt、f32 与 f16x2 的 min / max；@p / @!p 谓词；.reg 声明、大括号、行注释 |
| C 打包 | 按 Triton 的下降：元素宽 w 的张量占 `max(1, pack·w/R)` 个连续操作数（R 由 constraint 字母给出），组内第 k 个元素在第 `(k·w)//R` 个寄存器的第 `(k·w)%R` 位；先输出后输入 |
| D 环境读取 | `%globaltimer`、`%clock`、`%smid` 等：没有参照值，结果记未建立，程序继续 |
| E 副作用 | asm 内的 ld / st / atom / bar 等不在本增量内：程序中止，被改动的元素按已有规则记未建立 |
| F 覆盖统计 | `kernel_coverage` 用同一个静态分类：子集内的片段算支持；环境读取和副作用写明原因 |

不在本增量内：asm 内的访存；浮点指令以外的近似函数；f16x2 的算术（只有比较和 min / max）。

## 3. 预算

已用约半天。评价运行约 20 min。

## 4. 预期（评价运行前写定）

- 主线转储的 5952 个唯一 TTIR 中，按名字覆盖的从 5945 升到 5949；剩下 3 个是两次环境读取和一次 asm 内的访存，都写明原因。
- 捕获复跑（`test_inline_asm*`、`test_dot_max_num_imprecise_acc`、`test_typeconvert_upcast`）：
  - 官方 5 个 inline asm 测试中，shf、packed、multiple_outputs、packed_multiple_outputs 完整；with_pointers（asm 内访存）中止并写明原因；
  - `test_dot_max_num_imprecise_acc` 的 fp8e4b15 用例：转换部分可求值；整个启动是否完整取决于 dot 部分，不预设；
  - `test_typeconvert_upcast` 的 2 次中止变为可求值。
- 逐签名测试三类齐全，违反 0；回归不变。
