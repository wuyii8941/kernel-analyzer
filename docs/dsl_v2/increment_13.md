# DSL v2 第十三个增量：登记（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中）。对应 rc3 正式版 04 的 W3（NVIDIA 目标模块：矩阵路径、块缩放）。本登记写于实现之前。

## 1. 选这组的理由

增量 11 之后，W0 观察清单中 sm_90 / sm_100 TTGIR 出现的 129 个操作只剩 `ttng.tc_gen5_mma_scaled` 没有规则（sm_100 上 5 个唯一
kernel：官方测试的 `simple_dot_mxfp` 1 个、`mxfp8_mxfp4_matmul_tma` 4 个流水线变体；AMD 观察清单里它只出现在 sm_86 的 TTGIR 中，
即同一批 dump 本身）。本机只有 sm_86，这些 kernel 的官方测试要求 capability ≥ 9 或 10，不能在本机运行。

## 2. 做法

依据官方 `include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td`（提交 e50b186e8bd2）：
`$d += matrix_multiply(scale($a, $a_scale), scale($b, $b_scale))`，操作数 a、b、d、a_scale、b_scale、useD、pred、lhs / rhs 格式、
可选的 barrier 与谓词、`is_async`。

| 项 | 内容 |
| --- | --- |
| A | 解码与 `tt.dot_scaled`（增量 3）相同：fp8 e4m3 / e5m2 精确值，fp4 e2m1 每字节两个、沿 K 打包、低半字节在前（a 为 [M, K/2]，b 为 [K/2, N]）；a_scale 为 [M, K/32]、b_scale 为 [N, K/32] 的 E8M0，乘 2^(e−127)，e = 255 为 NaN（含它的行 / 列未建立）；bf16 / fp16 不缩放 |
| B | 累加、谓词与完成与 `tc_gen5_mma`（增量 11）相同：useD 为假时 D = A·B，pred 为假时不更新；异步时 D 在某个 barrier 的 wait 观察到完成之前未建立 |
| C | 两个求值共用一个解码函数（`tt.dot_scaled` 的实现抽出），不另写一份 |
| D | 设备验证：待做（device_validation = pending） |

## 3. 评价设计（实现前写定）

- **两层对照（合成捕获）**：5 个 kernel 的官方 TTIR 用官方主线编为 sm_100 TTGIR（只编译）；在同一组随机输入上 TTIR 参照与 TTGIR
  参照逐元素比较，都完整的元素区间必须相交；另在选定元素上与测试中独立算出的精确有理数比较。
- **名字覆盖**：W0 观察清单中 sm_90 / sm_100 TTGIR 的操作按名字全部有规则。
- 逐签名测试三类齐全，违反 0；回归不变。

## 4. 预期

- 5 个 kernel 的 TTGIR 参照在正例输入下全部完整，与 TTIR 参照都完整的元素不相交 0；选定元素包住精确值。
- sm_90 / sm_100 名字覆盖 129/129。
- 设备验证：0（待做）。
