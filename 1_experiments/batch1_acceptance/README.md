# 第一批验收（共同语义内核：来源 / 精度 / 安装 / 查询）

协议 [protocol.md](protocol.md) 与保留集、预先写定的答案（[items.json](items.json)、[predictions.json](predictions.json)）在运行之前冻结于
提交 `da4b64e`（工具代码 `d26d5f6`）；运行与结果在其后，逐项报告与日志在 [results/](results/)（`results.json` 为三态汇总）。逐项账见
[3_audits/batch1_semantic_core](../../3_audits/batch1_semantic_core/README.md)。

**结果：24 项中成立 24，不成立 0，无法判断 0。** 来源 14 项中没有一项在真值不干净时给出 call 级（可靠性违反 0）；3 项真值干净而给
kernel 级（H-P1b、H-P7、H-P12），与事先登记的保守预期一致，是能力缺口。精度两项的达标级别与独立预测（60 位精度下的界）一致；bias 两项
与舍入方向推出的符号和等价判断一致；干净 wheel 在源码树外安装、两个入口与一次真实测量通过，审计钩子没有看到对 `2_tool/` 的访问；
公开 v1.1 回归 28 个程序与上一轮逐程序相同。

| 项 | 组 | 谱系 | 结构 | 真值 | 观察 | 状态 | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H-P1a | provenance | Gluon (TTGIR) | ATen in-place add, bytes unchanged, on an input | not clean | kernel-level | 成立 |  |
| H-P1b | provenance | Gluon (TTGIR) | ATen in-place clamp that is the identity on the data | clean | kernel-level | 成立 | conservative (truth clean, as pre-registered) |
| H-P2 | provenance | classic Triton, 2D strided | transposed view of a declared input | clean | call-level | 成立 |  |
| H-P3 | provenance | classic Triton, 2D strided | overlapping windows (unfold) of an input | clean | call-level | 成立 |  |
| H-P4 | provenance | ATen where + classic Triton | elements of x or exact zeros | clean | call-level (producer record) | 成立 |  |
| H-P5 | provenance | ATen cat + classic Triton | x followed by exact ones | clean | call-level (producer record) | 成立 |  |
| H-P6 | provenance | ATen index_put_ + classic Triton | partial constant write over a rounded product | not clean | kernel-level | 成立 |  |
| H-P7 | provenance | ATen index_put_ + classic Triton | an exact copy through index_put_ | clean | kernel-level | 成立 | conservative (truth clean, as pre-registered) |
| H-P8 | provenance | Inductor-compiled reader | cross-input bytes written through .data | not clean | kernel-level | 成立 |  |
| H-P9 | provenance | Inductor-compiled reader | declared input read by compiled code | clean | call-level | 成立 |  |
| H-P10 | provenance | classic Triton, inference-mode input | input without a version counter | clean | call-level | 成立 |  |
| H-P11 | provenance | classic Triton, inference-mode input | in-place add without a version counter | not clean | kernel-level | 成立 |  |
| H-P12 | provenance | classic Triton, two launches | ATen write to the other half between launches | clean | kernel-level | 成立 | conservative (truth clean, as pre-registered) |
| H-P13 | provenance | classic Triton, two launches | exact ATen scaling between launches | not clean | kernel-level | 成立 |  |
| H-R1 | precision | classic Triton | axis-1 tile sum, n = 256, cancelling pairs up to 2^90, sum ~2^-60 | — | met，第 2 级 | 成立 |  |
| H-R2 | precision | classic Triton | reverse prefix sum, n = 256, alternating +-2^22 plus [0.25, 4) | — | met，第 2 级 | 成立 |  |
| H-R3 | precision | classic Triton | elementwise exp, target 1e-9 ulp | — | backend limit，第 1 级 | 成立 |  |
| H-R4 | precision | classic Triton | float64 row sum, target 1/8 ulp | — | backend limit，第 1 级 | 成立 |  |
| H-R5 | precision | classic Triton | 16 programs atomic_add one int32 counter and store the old value | — | intrinsic set width，第 1 级 | 成立 |  |
| H-Q1 | bias | classic Triton | bf16 rounding toward zero of 3x + 1 | — | fixed_mean.R1.approximate: nonzero (positive)；fixed_mean.R1.bounded: nonzero (positive)；aligned.R2.approximate: nonzero (positive)；aligned.R2.bounded: nonzero (positive)；fixed_mean.R1.equivalence: not shown | 成立 |  |
| H-Q2 | bias | classic Triton | bf16 rounding to nearest even of 3x + 1 | — | fixed_mean.R1.equivalence: within delta；aligned.R2.equivalence: within delta | 成立 |  |
| H-Q3 | bias | declaration | equivalence axis requested without delta | — | equivalence.rel (required because query.axes includes equiva | 成立 |  |
| H-K1 | packaging | wheel | clean wheel install outside the source tree | — | 0 | 成立 |  |
| H-G1 | regression | public v1.1 programs | 28 programs, mode B, 96 units | — | 28 个程序，差异 0；命中 27/28 | 成立 | evaluated by compare_regression.py (protocol section 4) on run reg20261010T1241; execution deviation: the 4 race programs were launched on the GPU by mistake, their results are not in the record (carried job files of reg20261009T1051 kept) |

执行偏差（不影响上表的判定，但必须记录）：H-G1 的回归运行把 4 个竞争程序（T5）误在 GPU 上运行（运行脚本按测量通道选全部程序，上一轮靠
预先放入沿用作业排除它们）。这违反任务书「不绕过权限运行竞争 kernel」；运行没有经过权限提示。4 份结果不入记录，记录沿用 reg20261009T1051
的作业文件；运行脚本已改为默认跳过竞争程序（冻结之后对运行脚本的改动，工具代码未改）。

捕获抽查（协议第 5 节）：在官方主线 Triton 环境下重跑教程 01、02、05 的捕获，5 次启动的状态、完整元素数、分类、原因与前提字段与
`auditfix_tutorials` 记录逐项相同（[results/capture_spot_tutorials_compare.json](results/capture_spot_tutorials_compare.json)）。

本批的保留集由执行方在代码冻结前编写，不能替代外部审阅方的保留集（待验收）；没有 AMD 设备，AMD 设备验证仍为 pending。
