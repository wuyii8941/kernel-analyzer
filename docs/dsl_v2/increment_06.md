# DSL v2 第六个增量：登记（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中）。对应 rc3 正式版 04 的 W2（通用区域）、W3（前端绑定）、W5（外部函数、格式）。

**时间说明**：本增量的五项（§2 的 A–E）都在登记之前实现，并已有逐签名测试（`tests/test_signatures_inc6.py`、libdevice 的
Bessel 用例）。按增量 2 的做法如实写明。评价运行（§4，官方单元测试的全量捕获复跑）在登记提交之后才开始，预期写在运行之前。

## 1. 选这几组的理由

增量 4 的代码在官方主线上跑了一次全量捕获（`test_core` 全部，另有 random、standard、libdevice、conversions；
tensor_descriptor 在跑时还用旧捕获代码），12,202 次启动中完整 11,180 次。除增量 5 已覆盖的原子与同步操作外，其余问题按次数：

| 原因 | 次数 | 归属 |
| --- | --- | --- |
| 捕获把元组参数记成一个参数，TTIR 参数数对不上（tensor descriptor 测试的 shape/strides、test_cat_nd 的张量列表、test_const） | 340 + 10 + 5 次求值器错误 | W3 前端绑定 |
| argmax / argmin 遇到 NaN 直接中止 | 6 | W2 |
| libdevice 的 Bessel 函数（j0、j1、y0、y1、cyl_bessel_i0、cyl_bessel_i1，含 f 版本）没有语义 | 12 | W5 |
| bf16 / fp8 与整数之间的位转换 | 4 | W5 |
| `tt.map_elementwise`（转储中 3 个 kernel；捕获的测试选择里没有） | — | W2 |

另外两类不是语义问题：「部分参照、无原因」的 72 次是插件把 NaN / ±inf 参照值算作未完成（增量 5 已改插件）；「什么都没写」的
43 次核对后都是正确结果（mask 全假的 store、或没有张量参数的 kernel）。

CAS 自旋锁（教程 05 的 layer-norm 反向、test_atomic_cas）需要协议摘要的设计，留到增量 7。

## 2. 目标签名与做法

| 项 | 签名 | 做法 |
| --- | --- | --- |
| A | 元组参数 | 捕获按前端的方式展开元组（`shape` → `shape.0`、`shape.1`，可嵌套），每个元素用它自己的签名项；值为 1 而被特化成 constexpr 的元素随之去掉。`_after` 用同一展开 |
| B | argmax / argmin 遇到非有限值 | 官方 combine 是 `v1 > v2` 加 where，遇到 NaN 不满足交换律（NaN 在左被丢弃，在右会传播），不是次序无关的。有 TTGIR 布局时改走下降树（增量 1 的通用区域路线）；各 lane 结果不一致时，按现有的 lane 凸包规则记未建立（结果被哪个 lane 使用尚未建模） |
| C | Bessel 函数 | Arb 球算术（python-flint，200 位）对包含区间的球求值，端点转 float64 后外扩一 ulp；y0、y1 只在 x > 0 上建立；±inf 与 NaN 按极限 |
| D | 位转换 | int16 ↔ bf16、int8 ↔ fp8（e5m2、e4m3fn）。浮点到整数沿用「位确定」政策（可精确表示的点或无穷）；整数到浮点按位解码 |
| E | `tt.map_elementwise` | 区域是纯标量函数，按 `pack` 个连续元素一组解释（区域参数按「每个输入的组内元素」排列，结果同理）；区域里可以有 cf.br / cf.cond_br。某组的分支无法判定时，只记该组未建立，程序继续 |

## 3. 预算

已用约半天。评价运行约 1 h（六个官方测试文件，GPU 1）。

## 4. 预期（评价运行前写定）

- tensor descriptor 测试（test_tensor_descriptor 文件）：不再出现「TTIR 参数数对不上」的错误。sm_86 上主线把 descriptor
  改写成指针运算加 mask，所以预计大部分启动参照完整；越界填充为 NaN 的选项给出 NaN 特殊值（算作已建立）。
- test_cat_nd、test_const：可以求值，预计完整。
- argmax / argmin 带 NaN 的 6 次：不再中止。各 lane 结果一致时完整，不一致时索引输出记未建立、值输出完整。
- Bessel 12 次：可以求值；正常输入完整，接近溢出的输入记未建立。
- 位转换 4 次：可以求值。
- 求值器错误（status = error）：除了尚未识别的新缺陷，预计为 0。若出现新缺陷，记录并修复，修复前不计入完成。
- 全量的完整比例（complete / 全部启动）高于增量 4 那次的 11,180 / 12,202；具体数字不预设。
