# DSL v2 开发记录（工具 4.0，增量 1–15）

依据参照 DSL v2 rc3 正式版（`reference_dsl_v2_rc3_official_20261009.zip`，SHA-256 8050903f…6faf；设计文档副本在 [design_rc3/](design_rc3/)）。
分支 `dsl-v2`，工具 4.0，**未发布、未冻结**。这里的数字是开发证据，**不是盲测成绩**；成绩只按三种状态写：成立 / 不成立 / 无法判断。

本文件合并了原来 15 份增量登记与 12 份结果文件中的结果部分：每个增量的结果原文按顺序收在下面。登记原文可由下表的提交取回
（`git show <提交>:docs/dsl_v2/increment_NN.md`）；增量 1–9 的登记与实现在同一批提交里，增量 10–15 的登记各在实现之前单独提交。

## 当前状态（截至增量 15）

| 项 | 结果 | 数据 |
| --- | --- | --- |
| 官方测试广泛捕获（官方主线 e50b186e，sm_86，6 个官方测试文件；外部审计修复版与证书之后重跑，提交 503a19e） | 12,766 次启动：完整 12,100、完整（集合目标）260，合计 96.82%（无条件）；完整（未证明前提下）157（官方 roll 等 z3 不能证明结合的 scan），与前者合计 98.05%；部分 118、中止 2、求值器错误 0 | [captures/auditfix_broad.jsonl](captures/auditfix_broad.jsonl.gz) |
| TTIR 与 NVIDIA sm_90 / sm_100 TTGIR 两层对照（审计修复版之后重跑） | 每个目标 2,138 次启动、7,344,666 个元素两层都建立，不相交 0；两层都无条件完整 1,981 次，roll 等 156 次两层都在前提下 | [captures/auditfix_cross_nvidia.jsonl](captures/auditfix_cross_nvidia.jsonl.gz) |
| TTIR 与 AMD gfx942 / gfx950 TTGIR 两层对照（审计修复版之后重跑） | 同上：7,344,666 个元素不相交 0，1,981 / 156；补充集 454 次启动两层都完整 362 次（含官方 test_atomic_cas 的锁证书） | [captures/auditfix_cross_amd.jsonl](captures/auditfix_cross_amd.jsonl.gz) |
| 官方 TTGIR 名字覆盖 | sm_90 / sm_100 观察到的 129 个操作 129/129；gfx942 / gfx950 的 5952 个 kernel 5949/5952 | [w0/](w0/) |
| W1 规则契约（3.6.0 回归 profile） | 430/430 个支持条目三类测试齐全并有触发证据 | [w1_contracts/coverage.json](w1_contracts/coverage.json) |
| v1.1 程序回归（非盲测） | 见增量 1 | [regression_v11/](regression_v11/) |
| 设备验证（sm_90 / sm_100 / gfx942 / gfx950） | 0（本机只有 sm_86；这些路径只编译、在 CPU 上求值） | — |
| 外部审计（基线 1aee15e）发现的 F01–F07 | 已修复（提交 5fe8e56 起），欠账补齐（证书、生产者记录、按需精度）；受影响的捕获全部重跑 | [3_audits/fix_1aee15e](../../3_audits/fix_1aee15e/README.md) |

## 增量一览

「状态」是该结果文件「预期对照」表中成立 / 不成立 / 无法判断的条数（多增量合写的文件只计一次）。

| 增量 | 内容 | 登记提交 | 结果提交 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | 通用 combine 区域与执行有效性；v1.1 程序回归 | `3be713e` | `a355614` | （无对照表） |
| 2 | 触发追踪、规则契约基线、有界路线与设计敏感性 | `3d94b9e` | `f3d8a57` | 4 / 2 / 1 |
| 3 | dot_scaled、无竞争 CAS、无符号原子；W6 八类校准 | `50b0606` | `f3d8a57` | （同上） |
| 4 | 原子返回值（点 / 集合目标）、位视图原子、次序律 | `23939a6` | `27237ff` | 13 / 0 / 0 |
| 5 | 原子 load / store / poll 与 happens-before、直方图、近似除法 | `94f530f` | `27237ff` | （同上） |
| 6 | 元组参数、NaN 的 argmax、Bessel、窄格式位转换、map_elementwise | `00fd9db` | `27237ff` | 5 / 1 / 0 |
| 7 | CAS 自旋锁：串行化与逆序证据、向量时钟 happens-before | `954e9af` | `53d754b` | 10 / 0 / 0 |
| 8 | W1 逐签名证据 | `126c398` | `53d754b` | （同上） |
| 9 | 位级 inline PTX（约束、打包、多输出） | `53d754b` | `16bd408` | 5 / 0 / 0 |
| 10 | Gluon 导入（TTGIR）、共享存储、异步拷贝、mbarrier | `91f285f` | `cba4ee7` | 6 / 0 / 0 |
| 11 | NVIDIA Hopper / Blackwell TTGIR（wgmma、tcgen05、tensor memory）与两层对照 | `40bb916` | `eae7968` | 4 / 0 / 0 |
| 12 | AMD gfx942 / gfx950 TTGIR（buffer 操作、scaled upcast、in-thread transpose） | `a92ac31` | `96fc2ed` | 5 / 0 / 0 |
| 13 | tc_gen5_mma_scaled | `c9472ad` | `1aee15e` | 4 / 0 / 0 |
| 14 | 整数原子返回值的集合目标（L_E） | `3fb81bf` | `1b61e55` | 3 / 1 / 0 |
| 15 | CAS 整数缺陷、remf（被除数绝对值小于除数）、位计数、libdevice rint / clz / popc | `0439b6c` | `e4f1b68` | 5 / 0 / 1 |

## 数据目录

- [captures/](captures/)：官方测试捕获与两层对照（每行一次启动），运行脚本在 [captures/run/](captures/run/)。只保留最新版本：外部审计修复版与证书
  之后（提交 503a19e）全部重跑，`auditfix_broad`（广泛捕获，含 test_core 的原子测试）、`auditfix_gluon_test_core`（含 Gluon 原子测试）、
  `auditfix_cross_nvidia`、`auditfix_cross_amd*`、`auditfix_tutorials`，运行脚本 `run/auditfix_rerun_all.sh`；与各自前一版的逐启动对照在
  `auditfix_capture_comparisons.json`。被替换的增量文件（`inc15_broad`、`inc14_test_core_atomic`、`inc14_gluon_test_core_atomic`、
  `inc10_gluon_test_core`、`inc11_cross_level_combined`、`inc12_cross_level_amd*`、`inc4_tutorials`、`inc7_tutorial05`）在 git 历史里
  （本次记录提交之前的 `1_experiments/dsl_v2/captures/`），下文各增量一节提到它们时指的是当时的运行。合成捕获 `inc13_scaled_mma_synthetic`
  与 `inc1x_compare_*`（各增量当时的逐启动比较）保留。
- [w0/](w0/)：W0 官方清单（源码、注册表枚举、观察到的操作，NVIDIA 与 AMD）与覆盖结果。
- [w1_contracts/](w1_contracts/)：W1 规则契约与完成度。
- [regression_v11/](regression_v11/)：v1.1 程序回归运行。
- [calibration/](calibration/)：W6 八类校准。

## 第一个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。依据：参照 DSL v2 rc3 正式版。登记见 increment_01.md（登记，提交 `3be713e`）。
general-v3.1 和结构验收 v1.1 的结果都没有改；v1.1 的计分仍只绑定运行 r20261008T2030。v1.1 的程序现在是开发集与回归集，
下面的数字**不是盲测成绩**。

### 1. 做了什么

| 项 | 内容 | 验证 |
| --- | --- | --- |
| A 通用 combine 区域（02 §6.2，04 W2） | 任意 combine 区域沿 TTGIR 布局给出的 lowering 次序逐步解释，目标是顺序特定的；Welford 闭式保留为快速路径；没有布局时记未建立 | 8 条测试；按新规则改写的 4 条，用测试里独立写的有理数归约树逐元素核对包含性 |
| B 执行有效性（02 §6.3，04 W4） | 跨 program 的读后写与写后读；同一 program 内无屏障的写后读按 TTGIR 布局做线程级判定（同线程有序，异线程竞争，映射未知则未建立）；`gpu.barrier` 分阶段 | 5 条测试，只在 CPU 上求值编译产物，有竞争的 kernel 从不运行；标量 store 只由线程 0 执行，在 PTX 上可见 |
| C 重复启动（02 §8.7） | `check.run` 每个输入至少启动 2 次，含浮点原子时 8 次；只重取 K，不重算参照。逐位一致照常；判过竞争的不做统计；原子程序按输入内平均；原因不明的只诊断 | 4 条 CPU 测试，包括用有理数核对输入内平均的外向舍入 |

注册表：`2_tool/data/rule_registry.json`，只覆盖 3.6.0 回归 profile 的规则，不是 rc3 官方目标面的覆盖率。
任意 combine 区域登记为「声明前提」，前提是 lowering 次序模型（浮点求和已逐位核对，combine 内的操作数次序按源码），交审阅方。

### 2. 回归结果（v1.1 的 32 个常规程序，主轮种子，模式 B）

运行 `reg20261009T1051`（增量 1）。外部审计修复版之后重跑为 `reg20261009T2319`（记录见提交 3532066）；第一批共同语义内核之后重跑为 `reg20261010T1241`（[regression_v11/reg20261010T1241](regression_v11/reg20261010T1241/SUMMARY.md)，与 reg20261009T2319 逐程序 28/28 相同；这一轮 4 个竞争程序被误在 GPU 上运行，结果不入记录，见该页与 3_audits/batch1_semantic_core）。reg20261009T2319 当时的结果在 `1_experiments/dsl_v2/regression_v11/reg20261009T2319`（`SUMMARY.md`、逐程序作业）：28 个常规程序的观察、命中、K−G 判定与执行状态与 reg20261009T1051 逐程序相同；4 个竞争程序没有在 GPU 上重跑，它们在 reg20261009T1051 的作业文件放在该目录的 `carried_from_reg20261009T1051/`。下面的数字是 reg20261009T1051 的。

- **预计命中 31/32。** 唯一未命中的是 prog_29：我预计它的均值与方差因 0/0 未建立，实际参照完整。实际布局下，每个线程先合并 4 个有效元素，
  再合并 4 个被 mask 的元素，不会出现两个零权重相遇。这也解释了上一轮设备输出没有 NaN。
- **安全标准：违反 0 处。** 判为竞争或不一致、统计不做的输出上，没有出现「非零」。

| 组 | v3.1（r20261008T2030） | 4.0 |
| --- | --- | --- |
| T1 Welford 变体（prog_08、17、20、29、32） | 10 个输出参照全部未建立（语义缺失），只能补跑 K−f | 10/10 参照完整，可以分解。prog_08、17、20、29 的 variance：K−G 两类规则都判非零，G−f 未确认（实现的数值作用）。prog_32 的 variance：G−f 固定均值非零（语义上少了 δ² 项），K−G 恒为零、判退化 |
| T5（prog_01、13、22、23） | K−G 固定均值判为非零（实际是竞争） | 4/4 判为执行竞争，统计不做；重复启动全部不一致 |
| T7（prog_11、18、19、27） | 适配器把参照重算 8 遍 | 工具内每个输入启动 8 次、只重取 K，按输入内平均；判断与 v3.1 相同（prog_18 对齐类非零） |
| 其余 19 个 | — | 参照质量与 K−G 规则记录（逐单位投影端点）都与 v3.1 逐项相同 |

成本：作业墙钟合计 19,858 s，同样 32 个程序在 v3.1 主轮模式 B 下是 41,712 s。T7 每个程序从约 6,300–6,900 s 降到 850–1,200 s。
16 个进程并行，墙钟 26 min。

### 3. W0 已开始的部分

- **官方基线**：Triton 主线 `e50b186e8bd2d16ae3f564311ef868aa3bd45a2d`，浅取到 `.cache/upstream/triton`，作为数据读取，没有执行。
  它钉住的 LLVM 是 `b010a18d…`。本机 glibc 是 2.31，ubuntu-x64 预编译包跑不起来，所以用 almalinux-x64 包，SHA-256 与官方值一致。
- **公共 API 清单**：`1_experiments/dsl_v2/w0/api_e50b186e8bd2.json`，493 项，用 AST 读取，未声明完整。其中 `triton.language` 148 项，
  与 rc3 包里的经典导出快照数目一致。
- **源定义清单**：`1_experiments/dsl_v2/w0/source_ops_e50b186e8bd2.json`。用钉住版本的 `mlir-tblgen --print-records` 展开树里全部
  10 个 `*Ops.td`，得到 215 个操作定义，每个附规范化契约的哈希。
- **名字差集**（`official_vs_legacy_names.json`）：
  - 主线 `tt` 方言比 3.6.0 新增 6 个：`approx_divf`、`atomic_load`、`atomic_poll`、`atomic_store`，以及两个 grid dependency 操作；
  - 移除 3 个：`advance`、`cat`、`make_tensor_ptr`，主线已经没有 block pointer；
  - 其余方言（ttg、ttng、amdg、nvws、nvg、proton、proton_gpu、tti、gluon）的 162 个操作，旧 profile 都没有登记。
- **还缺**：构建期注册（要编译官方 Triton）、真实编译日志（要语料 B）、上游 MLIR 方言的清单。所以四方核账目前是 INCOMPLETE，
  这只是名字差集，还不是语义路线。

### 4. 下一个增量（候选，登记时再定）

1. W0 收尾：编译官方 Triton，导出构建注册；用审阅方给的语料 B 取编译日志；跑 rc3 的核账工具。
2. 结合律 / 交换律证书（有理函数恒等式加分母非零），使没有 TTGIR 次序的归约也能接纳；官方主线的 `ReduceOp::isCommutative()` 可作参照。
3. 统计 W6：有界路线；s = 0 时允许 Hoeffding。
4. 需要审阅方提供：与工具无关的随机合法 / 不合法程序生成器和精确执行器（04 分工）；冻结后的盲测 v2。

### 5. 没有做或不能说的

- 没有运行 prog_26（环境权限检查），同类模式由 CPU 合成用例覆盖。
- 线程级判定依赖布局模型；只建模了 blocked 与 slice 布局，其他布局记「线程映射不可得」。
- 「顺序特定的目标」依赖 lowering 次序这一声明前提。
- 4.0 没有冻结，这里的结果都不能当验收成绩。

## 第二、三个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 increment_02.md（登记，提交 `3d94b9e`）、increment_03.md（登记，提交 `50b0606`）。
general-v3.1 与结构验收 v1.1 的冻结结果都没有改。下面的数字都是开发证据，**不是盲测成绩**。成绩只按三种状态写：成立 / 不成立 / 无法判断。

### 1. 预期对照

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| 增量 2 D：至少 80% 的支持签名得到三类测试和触发证据 | 317/401 = 79.1%（有触发证据 326，有违反前提测试 320） | 不成立（差 0.9 个百分点） |
| 增量 2 D：包含性违反为 0 | 逐签名测试发现求值器缺陷 5 处，都已修复并加了回归测试（见 §2） | 不成立（预期过于乐观；缺陷已修） |
| 增量 2 E：构建注册与 TableGen 源定义一致或差异可解释 | 主线 `e50b186e` 的构建注册导出 1281 个操作，全部登记（`registered_e50b186e8bd2.json`）；TableGen 源定义 215 个 Triton 方言操作，上游方言 1066 个。四方核账要等观察清单 | 无法判断（核账未跑完） |
| 增量 3 A1/A2：主线 TTIR 中 assume、ttg.barrier、atomic_cas、dot_scaled 不再挡住任何 kernel | 原 473 个 TTIR 的集合无法从转储目录精确复原（后续运行改写了文件时间）。在 13:55 的 4867 个唯一 TTIR 上，这四个名字都不再出现在阻塞表里（`pre-reorg-20261009:results/dsl_v2/w0/main_ttir_coverage_20261009T1355.json`） | 成立（在更大的集合上） |
| 增量 3 A3：教程 vector-add、softmax、layer-norm 参照完整；matmul 完整并记录 dot 精度 | vector-add、softmax、matmul 完整，matmul 记 `dot_input_precision:tf32`。layer-norm、fused-attention、grouped-gemm、persistent-matmul 正在跑 | vector-add/softmax/matmul 成立；其余无法判断（未跑完） |
| 增量 3 A3：捕获模块能在主线运行时工作 | 能。test_core 子集 5176 次启动：完整 4677，部分 274，中止 80，求值器错误 145。错误全部是工具缺陷，在增量 4 修复（整数 dot 120 次、fp8e4b15 未解包 25 次） | 成立（附缺陷） |
| 增量 3 B：含 CAS 的 kernel 如实报告 | 无竞争时直接求值，有竞争时报执行竞争（比登记多做了一步） | 成立 |
| 增量 3 C：dot_scaled 包含性违反为 0 | 6 条逐签名测试（fp8 e4m3 与 fp4 e2m1，三类各一），违反 0 | 成立 |

### 2. 发现并修复的缺陷

| 来源 | 缺陷 | 修复 |
| --- | --- | --- |
| 逐签名测试 | 包围离开 float64 有限范围时仍标为完整 | 记未建立 |
| 逐签名测试 | 特殊值路径上，非精确的有限极限没有外扩 | 外扩一个 ulp |
| 逐签名测试 | erfc、asinh、acosh、cbrt、exp10、saturate、tan 缺极限规则 | 补齐 |
| 逐签名测试 | 整数操作数的 abs 走了浮点路径 | 改走整数规则 |
| 逐签名测试 | 初等函数包围在溢出时抛异常 | 捕获后记未建立 |
| 主线捕获 | 整数 `tt.dot` 让求值器崩溃 | 精确整数矩阵乘；超出累加器范围记未建立（饱和还是回绕取决于下降） |
| 主线捕获 | `fp8e4b15` 操作数没有解包 | 按 IR 中的 i8 解包 |
| 代码审查 | 非 fadd 的原子贡献没有建立值时，仍按已建立折叠 | 对所有种类记未建立 |
| 主线转储 | 函数头的结果是带括号的列表时解析失败 | 按配对括号截取参数表 |
| 主线转储 | 指针地址空间为字符串（`"constant"`）、顶层属性别名行 | 都能解析 |

### 3. 五栏报告（rc3 04 要求）

数字针对本轮登记的目标签名，不是 rc3 官方目标面的全部。

| 项 | 官方目标 | 语义定义 | 实现 | 证据核验 | 设备测试 |
| --- | --- | --- | --- | --- | --- |
| 3.6.0 回归 profile 的支持签名 | 401 | 401 | 401 | 317 有三类测试与触发证据 | 只在 sm_86 上捕获过 |
| 主线 `assume`、`ttg.barrier` | 2 | 2 | 2 | 逐签名测试 | 主线捕获（sm_86） |
| `tt.atomic_cas` | 1 | 无竞争：有；有竞争：交错关系待定义 | 无竞争部分 | 1 条测试 | 未单独测 |
| `tt.dot_scaled` | 1 | 1（精确 MX 解码） | 1 | 6 条测试 | 待测（sm_86 不支持 fp8e4nv，按 sm_90 只编译） |
| W0 清单 | API / 源定义 / 构建注册 / 观察 | 前三项已导出 | — | 观察清单与核账未完成 | — |

### 4. W6 八类校准（`1_experiments/dsl_v2/calibration/eight_classes.json`）

每个设置 300 次重复，开发 32 个、确认 64 个单位，d = 64，α = 0.05。FM 是固定均值规则类的「非零」比例，AL 是对齐规则类的。

| 类 | μ = 0：FM / AL | μ = 0.1：FM / AL | μ = 0.4：FM / AL |
| --- | --- | --- | --- |
| 稠密同号 | 0.050 / 0.040 | 1.000 / 0.047 | 1.000 / 0.060 |
| 稀疏单坐标 | 0.073 / 0.043 | 0.770 / 0.057 | 1.000 / 0.037 |
| 组内抵消 | 0.080 / 0.043 | 0.797 / 0.023 | 1.000 / 0.037 |
| 均值零但对齐非零 | 0.037 / 0.040 | 0.043 / 1.000 | 0.043 / 1.000 |
| 状态相关 | 0.053 / 0.047 | 0.053 / 1.000 | 0.037 / 1.000 |
| 离散并列 | 0.027 / 0.040 | 0.817 / 0.020 | 1.000 / 0.027 |
| 罕见大值 | 0.033 / 0.037 | 0.463 / 0.030 | 1.000 / 0.030 |
| 原子执行波动（每输入 8 次取平均） | 0.060 / 0.027 | 1.000 / 0.047 | 1.000 / 0.033 |

- μ = 0 时，前五类的数据生成过程相同，都是纯噪声。合并 1500 次，FM 的误报率是 0.059（蒙特卡洛标准误 0.006），AL 是 0.043。
  FM 比名义 0.05 高 1.4 个标准误，不足以判定偏高，也不能判定没有偏高。
- 组内抵消和稀疏单坐标由 R5（学到的方向）检出；均值零但对齐非零、状态相关只有对齐类能检出，与设计一致。
- 无法判断的比例在全部设置里都是 0。每次重复约 25 ms（罕见大值约 40 ms）。

### 5. 没有做或不能说的

- 设备验证只有 sm_86；dot_scaled 只编译、没有运行。
- 语料 B、独立的随机程序生成器与精确执行器、盲测 v2 都要审阅方提供；W7 没有开始。
- 推送需要用户确认。本地提交到 `94f530f` 为止都未推送（origin/dsl-v2 = `a355614`）。

## 第四、五个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 increment_04.md（登记，提交 `23939a6`）、increment_05.md（登记，提交 `94f530f`）。
冻结结果都没有改。下面的数字都是开发证据，**不是盲测成绩**。成绩只按三种状态写：成立 / 不成立 / 无法判断。

主线捕获的原始结果在 `pre-reorg-20261009:results/dsl_v2/main_capture/`（每次启动一行，附运行脚本与 README）。

### 1. 增量 4（原子返回值、位视图原子、次序律）

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| test_atomic_rmw 中 fadd 类（4 种浮点）的 `Old` 由未建立变为集合完整（L_E） | 80/80 次启动完整，全部带 `set:` 原因 | 成立 |
| 中性值为 ±inf 的 max/min 仍未建立（凸包非有限） | float32/float64 的 max/min 仍未建立；机制不同：前端把它们降成整数位视图上的操作，`Old` 是整数集合，所以未建立，不是因为凸包非有限 | 结果成立，登记的机制不准确 |
| 整数 add/max/min 的 `Old` 仍未建立 | 180 次启动部分参照，原因是「集合值整数结果无法用点表示」 | 成立 |
| 80 个中止启动变为可求值：`Z` 完整、`Old` 未建立 | 80 次全部变为部分参照；`Z` 完整（次序律证书 {max ≥ 0, umin < 0} 成立），`Old` 未建立 | 成立 |
| 单地址无竞争的返回值为点值完整 | 逐签名测试通过（浮点和整数，单 program 和 grid） | 成立 |
| 包含性违反为 0 | 12 条测试按独立枚举的全部交错核对，违反 0 | 成立 |
| 混合种类且无证书时报告未建立 | 逐签名测试通过 | 成立 |
| 回归：已有测试不变 | 全套通过。有 2 条旧测试写的是旧规则（「返回值被使用就永不完整」），按新规则改写为「集合包含全部交错」，改动在同一提交里 | 成立（改写 2 条按旧规则写的测试） |

增量 4 期间发现并修复的缺陷：整数 `tt.dot` 崩溃（120 次启动）、`fp8e4b15` 操作数未解包（25 次）、非 fadd 原子的未建立贡献被当作已建立折叠（代码审查）、
带括号结果列表的函数头解析失败。修复后同一组测试复跑：错误 0。

### 2. 增量 5（原子 load/store/poll、直方图、近似除法）

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| 4867 个唯一 TTIR 中按名字完整覆盖的从 4717 升到至少 4860 | 原集合没有逐项保存（只存了计数和阻塞名字表），无法逐项复算。阻塞表里的每个名字现在都有规则，按推断这 4867 个都已覆盖。在转储全部完成后的 5952 个唯一 TTIR 上，增量 6 代码覆盖 5945 个，剩下 7 个都是 inline asm | 原集合：无法判断（只能推断）；更大集合上 99.9% |
| 单 program 的 atomic load/store 与 poll 参照完整 | test_atomic_load_store 36、coalesced 12、test_atomic_poll 18、no_timeout 1、tensor_results 8、timeout 2 次，全部完整 | 成立 |
| `test_atomic_poll_waits_for_remote_cta`：out = 42 完整，记假设「poll 会终止」，payload 读不报竞争 | 完整；记了这条假设；happens-before 边 1 条；无竞争原因 | 成立 |
| 带超时、期望值不在内存里的 poll 结果完整（假） | 完整 | 成立 |
| histogram：输入都在范围内时完整 | test_histogram 7、mask 7、compare_mask 2 次完整 | 成立 |
| 逐签名测试三类齐全，包含性违反 0 | 18 条测试，违反 0 | 成立 |
| 回归 | 全套通过 | 成立 |

**偏离（已在复跑前记录，increment_05.md §6）**：直方图越界值。登记时认为官方没有说明越界值怎么处理，捕获复跑时两个官方测试报为部分参照。查官方源后发现，
官方解释器写明「GPU 丢弃所有越界值」，官方测试也断言这一点，所以规则改为丢弃越界值。两种口径：

- 按登记的规则：`test_histogram_out_of_range` 与 `test_histogram_silent_data_corruption` 为部分参照（2/108 次启动）；
- 按偏离后的规则：19/19 次启动完整（`inc5_histogram_after_deviation.jsonl`）。

增量 5 期间还修了：指针地址空间为字符串（`"constant"`）、顶层属性别名行、`triton.reinterpret` 传 torch dtype、无符号 64 位标量、
原子操作的地址未建立时求值器崩溃。

### 3. W0 核账（rc3 工具 `reconcile_official_inventory.py`）

- **源定义**：TableGen `--print-records` 展开树中全部 `*Ops.td` 与 triton-opt 加载的上游方言，共 1281 个操作。
- **构建注册**：用 triton-opt 自己的编译与链接命令编了一个小程序（`2_tool/scripts/dsl_v2/registry_dump.cpp`），在同一套方言注册下调用
  `MLIRContext::getRegisteredOperations()`，枚举出 1281 个操作。与源定义两个方向都没有差集。
  - 此前的 `registered_e50b186e8bd2.json` 只是逐个探测源定义里的名字，构建里有、源定义里没有的操作它找不到。它的说明里写了这个限制，
    却仍标了 `complete_for_profile: true`，这个声明过强。现在由枚举版 `registered_enum_e50b186e8bd2.json` 取代；旧文件保留，不改。
  - 注册表只给名字，所以枚举版每条记录的契约哈希是从源定义复制的（记录里写明）。工具的「源定义 / 注册契约比较」在这里没有实际作用。
- **观察**：官方教程与 7 个官方单元测试文件的转储（5952 个唯一 TTIR，sm_86），再为 sm_90、sm_100 重新编译。共 130 个操作，全部已注册。
  观察解析器最初把属性键（`tt.divisibility = 16`）当成操作，已修正。
- **工具输出**：`READY_FOR_MANUAL_REVIEW`（`1_experiments/dsl_v2/w0/reconcile_enum_20261009/reconcile_output.json`）。
  - 目标 1281 条都有路线。其中 141 条是「已实现、未验证」（3.6.0 回归 profile 的规则），其余是「计划中」，各自写明 rc3 设计章节与工作包。
  - 「已验证」的路线声明为 0。
  - 按工具说明，这个状态只表示输入清单内部一致、目标都有路线。它不是语义证书，也不表示语言全覆盖或实现完成。
  - 观察清单不是语料 B。

### 4. 没有做或不能说的

- CAS 自旋锁（教程 05 的 layer-norm 反向、test_atomic_cas）仍未建立：中止或判为竞争。协议摘要留到增量 7。
- inline asm 的逐指令解析（读时钟 / SM 编号、多输出、打包的位运算、asm 里的全局 load）没有做，这是主线 TTIR 名字覆盖剩下的 7 个。
- 设备验证仍只有 sm_86。
- 推送需要用户确认；本地提交都未推送（origin/dsl-v2 = `a355614`）。

## 第六个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 increment_06.md（登记，提交 `00fd9db`）。数字是开发证据，**不是盲测成绩**。
成绩只按三种状态写：成立 / 不成立 / 无法判断。

### 1. 评价运行

官方主线 `e50b186e` 下跑 6 个官方单元测试文件的全部测试（test_core、test_random、test_standard、test_libdevice、test_conversions、
test_tensor_descriptor），每次捕获的启动都用增量 6 的求值器求值（`pre-reorg-20261009:results/dsl_v2/main_capture/inc6_broad.jsonl`）。
运行中发现两处捕获绑定缺陷：常量元组被展开，主机侧 tensor descriptor 没有按官方 ABI 展开。修复后（`2e30681`）只复跑受影响的文件和测试族
（`inc6_rerun_after_capture_fixes.jsonl`），合并结果在 `inc6_combined.jsonl`。

| 状态 | 增量 4（`inc4_broad`） | 增量 6（合并） |
| --- | --- | --- |
| 启动总数 | 12,766 | 12,766 |
| 完整 | 11,387 | 12,210 |
| 完整（集合目标，L_E） | 80 | 80 |
| 部分参照 | 279 + 72（无原因，旧插件把特殊值算作未完成） | 285 + 15（同上） |
| 中止 | 173 | 47 |
| 求值器 / 绑定错误 | 707 | 1 |
| 什么都没写（mask 全假或无张量参数，参照正确） | 46 | 106 |
| 超出捕获大小上限 | 22 | 22 |

完整 + 集合目标占全部启动的 96.3%（增量 4 是 89.8%）。登记里引用的「12,202 次中完整 11,180 次」是增量 4 那次运行还没跑完
（test_tensor_descriptor 尚在进行）时的计数，上表是它跑完后的最终数字。

剩下的 1 个错误是 `test_constexpr_assignment[literal2-None]`：带 `tl.constexpr` 注解的元组参数；该 kernel 没有张量参数，也不写任何东西。尚未修。

### 2. 预期对照

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| tensor descriptor 测试不再出现参数数对不上的错误，大部分完整 | 第一次评价运行仍有 332 次绑定错误：主机侧 descriptor 对象和常量元组没有覆盖。修复后复跑，test_tensor_descriptor 1096 次、test_standard 507 次启动，错误 0（test_tensor_descriptor 有 3 次什么都没写） | 按登记时的实现：不成立；修复后：成立 |
| test_cat_nd、test_const 可求值并完整 | test_const 5/5 完整；test_cat_nd 修复后完整 | 成立（cat_nd 在修复后） |
| argmax/argmin 带 NaN 的 6 次不再中止；lane 一致时完整，否则索引未建立 | 6 次都变为部分参照：值完整，索引未建立（lane 结果不一致）；原因文字在后来的提交里补上 | 成立 |
| Bessel 12 次可求值，正常输入完整 | 8 次完整；4 次部分参照，原因是 y0/y1 的输入含 x ≤ 0（实数上无定义），不是溢出 | 成立（未建立的原因与登记写的不同） |
| 位转换 4 次可求值 | 4 次都可求值；同一测试族另有 2 次中止，原因是 inline asm，不属于本项 | 成立 |
| 求值器错误除新缺陷外为 0 | 修复两处新缺陷后为 1（上面那一个） | 不成立（剩 1 个未修） |
| 全量完整比例高于增量 4 | 96.3%，增量 4 是 89.8% | 成立 |

### 3. 求值器错误与中止的变化

- 错误 707 → 1：元组参数展开（增量 6）、常量元组与主机侧 descriptor（`2e30681`）、地址未建立的原子与 uint64 标量（增量 5）。
- 中止 173 → 47：原子 load/store/poll（增量 5）、Bessel、位转换、argmax 带 NaN（增量 6）。剩下的主要是 CAS 自旋锁（增量 7 处理，结果另报）、
  inline asm（fp8e4b15 转换的位运算片段）和 1 处执行竞争。

### 4. 没有做或不能说的

- 这次评价仍只覆盖官方单元测试和教程，不是语料 B。
- 设备验证只有 sm_86。

## 第七、八个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 increment_07.md（登记，提交 `954e9af`）、increment_08.md（登记，提交 `126c398`）。
数字是开发证据，**不是盲测成绩**。成绩只按三种状态写：成立 / 不成立 / 无法判断。

### 1. 增量 7（CAS 自旋锁：串行化 + 逆序证据，向量时钟 happens-before）

官方主线下复跑 `test_atomic_cas` 与 `test_atomic_rmw`（`.cache/dsl_v2/w7_capture.jsonl`，归档见 §3）。

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| serialized_add：sem 为默认（acq_rel）或 acquire 时 data 完整，记声明前提 | int32、int64 各 3 种 sem（默认、acquire、acq_rel）共 6 次启动完整，每次 2000 个 program；记了声明前提；happens-before 边各 3998 条 | 成立 |
| serialized_add：sem 为 relaxed 或 release 时报竞争、参照未建立 | 4 次启动部分参照，原因「load 上的跨 program 竞争」 | 成立 |
| change_value（单 program）完整 | 10/10 完整 | 成立 |
| 教程 05 `_layer_norm_bwd_dx_fused`：DX、DW、DB、Lock、Count 完整，记声明前提 | 完整：1151 个 program、11,002,048 个写入元素全部建立，记了声明前提（两种次序合计求值 795 s）。同一教程的前向与 `_layer_norm_bwd_dwdb` 也完整 | 成立 |
| 逐签名测试：锁内累加完整；先到者写入未建立；relaxed 报竞争；不释放的锁不挂起 | `test_signatures_locks.py` 8 条通过 | 成立 |
| 回归不变 | 全套通过（合并后在主检出跑） | 成立 |

声明前提的范围：程序序与逆序两种串行化一致，只是证据，不是对所有串行化的证明。审阅方核对后才能改成已验证。

### 2. 增量 8（W1 证据）

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| 三类齐全且有触发的条目从 335/418 升到至少 400/418 | 416/419（注册表多了一条 CAS 串行化的声明前提条目） | 成立 |
| 剩下的是复合规则或需要设备行为的条目 | 剩 3 条：嵌套 scf.for / scf.if、load / store 别名、跨启动状态。它们是复合规则，没有单一的追踪签名 | 成立 |
| 补测试时发现的缺陷记录并修复 | 3 处缺陷：`ub.poison` 无操作数时解析不出结果类型（崩溃）；gather 一个越界下标让整个 program 中止（改为逐 lane）；负底数加整数指数的 pow 未建立（改为按单调段精确计算）。2 处可靠性缺口：`math.clampf` 沿用了 `tt.clampf` 忽略 NaN 的规则，但它的 maxf / minf 对 NaN 的行为没有核实（NaN 与 min > max 现在未建立）；通用 scan 默认 combine 满足结合律（现在用第二种括号方式在实际输入上核对，不一致的前缀未建立，前提写明） | 成立 |
| 回归不变 | 全套通过 | 成立 |

### 3. 结果文件

- W1：`1_experiments/dsl_v2/w1_contracts/coverage.json`（`.cache/dsl_v2/w1_final/` 的带追踪全套测试）。
- CAS 捕获：`pre-reorg-20261009:results/dsl_v2/main_capture/inc7_atomic_cas_rmw.jsonl`；教程 05：`1_experiments/dsl_v2/captures/inc7_tutorial05.jsonl`。

## 第九个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 increment_09.md（登记，提交 `53d754b`）。数字是开发证据，**不是盲测成绩**。
成绩只按三种状态写：成立 / 不成立 / 无法判断。

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| 5952 个唯一主线 TTIR 中按名字覆盖的从 5945 升到 5949，剩下 3 个写明原因 | 5949/5952；剩下 2 次环境读取（`%globaltimer`、`%smid`）和 1 次 asm 内访存，原因都写在覆盖报告里 | 成立 |
| 官方 inline asm 测试：shf、packed、multiple_outputs、packed_multiple_outputs 完整；with_pointers 中止并写明原因 | 4 个完整；with_pointers 中止，被改动的元素记未建立 | 成立 |
| `test_dot_max_num_imprecise_acc` 的 fp8e4b15 转换可求值；整个启动是否完整不预设 | 48/48 次启动完整（增量 6 时 24 次中止） | 成立 |
| `test_typeconvert_upcast` 的 2 次中止变为可求值 | 28/28 完整 | 成立 |
| 逐签名测试三类齐全，违反 0；回归不变 | `test_signatures_ptx_bits.py` 17 条通过：fp8e4b15 → fp16 转换按格式定义独立解码核对；全套通过 | 成立 |

结果：`pre-reorg-20261009:results/dsl_v2/main_capture/inc9_inline_asm.jsonl`（81 次启动，80 完整，1 中止）。

没有做的：asm 内访存（ld / st / atom）、f16x2 算术；时钟与 SM 编号是环境观察，按 rc3 W5 保留为没有参照值的可观察效果。

## 第十个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 increment_10.md（登记，提交 `91f285f`）（含实现前记录的两处偏离：共享存储按
编译器的 Membar 排序；异步拷贝掩码位置按官方测试零填充）。数字是开发证据，**不是盲测成绩**。成绩只按三种状态写：成立 / 不成立 / 无法判断。

### 1. 评价运行

官方主线下跑官方 Gluon 单元测试 `python/test/gluon/test_core.py`（sm_86 上 426 条通过，其余因需要 Hopper / Blackwell 跳过或失败），
捕获到的启动用增量 10 的求值器从 TTGIR 求值（`1_experiments/dsl_v2/captures/inc10_gluon_test_core.jsonl`）。

| 状态 | 增量 10 之前 | 增量 10 |
| --- | --- | --- |
| 启动总数 | 428 | 428 |
| 完整 | 0 | 358（83.6%） |
| 部分参照 | 0 | 60 |
| 中止 | 0 | 10 |
| 求值器错误 | 428（没有 TTIR） | 0 |

部分参照的原因全部写明：local atomic 被 mask 掉的 lane 不返回值（36，官方测试也只检查 final）；同一元素被几个 lane 同时更新、整数返回值是集合（12）；
`test_reduce_noncommutative` 的非交换 combine 让 warp 内各 lane 结果不一致（12）。中止的 10 次：逐线程 `ttg.inline_asm` 与 asm 内访问
共享存储（9，登记 F 项），`memdesc_reinterpret` 改元素类型（1）。

### 2. 预期对照

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| 求值器错误从 428 降到只剩新缺陷 | 0 | 成立 |
| 只含已有规则操作的 kernel 的启动大部分完整 | 是（未完整的都属下面几类，原因写明） | 成立 |
| 使用共享存储的启动大部分完整；竞争或未完成的异步拷贝报告为未建立并写明原因 | gather / scatter / subslice / reshape / 共享存储里存指针 / 异步拷贝加 mbarrier 都完整；local atomic 的未完整部分原因写明 | 成立 |
| 用到 F 项操作的启动报告为未建立并写明原因 | 9 次中止，原因写明 | 成立 |
| 427 个唯一 Gluon TTGIR 中按名字完整覆盖的从 222 升到至少 400 | 418/427；剩下的是逐线程 inline asm 6、asm 内访问共享存储 3、warp specialization 1 | 成立 |
| 逐签名测试三类齐全，违反 0；classic 回归不变 | `test_signatures_gluon.py` 13 条（官方 TTGIR，改动的变体写明）通过；全套通过 | 成立 |

### 3. 实现中发现并处理的问题

- 第一次复跑：local atomic 缺 `exch`；集合规则的原因没有传出来；共享存储里的指针丢了所属缓冲。都已修，再复跑得到上表。
- mbarrier：按 PTX 语义实现。没有 noIncrement 时，`async_copy_mbarrier_arrive` 是净零的到达，只让拷贝卡住当前阶段的完成；
  `wait_barrier` 的 parity 指当前阶段或前一阶段。按这个语义逐行走官方 kernel，拷贝恰好在最后一次 wait 时被确认完成。
- libdevice `__nv_fast_fdividef` 按其他 fast 变体的规则（同一数学函数）登记。

### 4. 没有做或不能说的

- 逐线程 inline asm、warp specialization、TMA、tensor memory、wgmma / tcgen05：sm_86 上不运行，没有设备证据；目标语义留在欠账里。
- 共享存储的程序序来自官方编译流水线（Membar），是前提，没有逐条核对 PTX 中的 barrier。

## 第十一个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 increment_11.md（登记，提交 `40bb916`）。数字是开发证据，**不是盲测成绩**。
成绩只按三种状态写：成立 / 不成立 / 无法判断。设备验证：0（本机只有 sm_86，sm_90 / sm_100 路径只编译不运行，记为待做）。

### 1. 两层对照（评价运行）

官方主线下跑 test_core 的 dot、reduce1d、reduce2d、scan2d、where、cast 测试（2120 条通过）。对每次捕获的启动：TTIR 求值一次；
同一份 TTIR 用官方编译器编为 sm_90 和 sm_100 的 TTGIR（只编译），在同一组捕获输入上各求值一次，逐元素比较
（`1_experiments/dsl_v2/captures/inc11_cross_level*.jsonl`，插件 `2_tool/scripts/dsl_v2/cross_level_plugin.py`）。

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

### 2. 预期对照

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| 两层都完整的元素区间全部相交，不相交 0 | 两个目标各 7,344,666 个元素，不相交 0 | 成立 |
| sm_90 / sm_100 TTGIR 名字覆盖：除未登记的 TMA / warp specialization 类外全部覆盖 | W0 观察清单中出现在 sm_90 / sm_100 TTGIR 的 129 个操作，128 个有规则；剩下 `tc_gen5_mma_scaled`（5 次，写明未建模） | 成立 |
| 设备验证 0（待做） | 0 | 成立（按登记） |
| 逐签名测试三类齐全，违反 0；回归不变 | `test_signatures_nvidia.py` 10 条（wgmma、tcgen05 对精确有理点积，以及与 TTIR 参照相交）；全套通过 | 成立 |

### 3. 同一时期的其他结果

- W1：421/421 个支持条目三类测试齐全并有触发证据。复合规则（嵌套控制、store→load 别名、跨启动状态）在实际发生的位置记触发。
- 缺陷（W1 别名测试发现）：经由未建立地址的 store 原来只在启动结束时让目标缓冲失效，同一 program 之后的 load 仍把旧值当作已建立。
  现在立即失效，并对其他 program 的读按「写了每个元素」检查。

### 4. 没有做或不能说的

- 没有设备证据：sm_90 / sm_100 的结论都只是「CPU 参照与 TTIR 一层一致」，不是设备上的正确性。
- TMA（tensor descriptor 在 sm_90 上的异步拷贝）、warp specialization、`tc_gen5_mma_scaled` 不在本增量内。

## 第十二个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 increment_12.md（登记，提交 `a92ac31`）（含登记前选型运行中发现并修复的求值器缺陷）。
数字是开发证据，**不是盲测成绩**。成绩只按三种状态写：成立 / 不成立 / 无法判断。设备验证：0（本机没有 AMD 设备，gfx942 / gfx950
路径只跑官方 AMD 后端到 TTGIR 为止的阶段，不运行，记为待做）。

### 1. 两层对照（登记的评价运行）

官方主线下跑 test_core 的 dot、reduce1d、reduce2d、scan2d、where、cast（2120 条通过）。对每次捕获的启动：TTIR 求值一次；同一份 TTIR
经官方 AMD 阶段编为 gfx942、gfx950 的 TTGIR，在同一组捕获输入上各求值一次，逐元素比较
（`1_experiments/dsl_v2/captures/inc12_cross_level_amd.jsonl`，求值器为提交 `681beda`，插件 `2_tool/scripts/dsl_v2/cross_level_plugin.py`，`KA_CROSS_TARGETS=gfx942,gfx950`）。

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

### 2. 预期对照

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| 两层都建立的元素区间全部相交，不相交 0 | 两个目标各 7,344,666 个元素，不相交 0 | 成立 |
| 选型运行中因 AMD 操作中止的启动（926 / 659）两层都完整；只剩两层都是部分参照的 1 次，原因与 TTIR 一层相同 | 2137/2138 两层都完整；剩下的 1 次是 `test_where[1-*int32]`，两层原因相同 | 成立 |
| 名字覆盖：gfx942、gfx950 从 922 / 956 升到 5949/5952，剩下 3 个与 TTIR 一层相同 | 两个目标都是 5949/5952；剩下的是 NVIDIA PTX inline asm 读环境（`%globaltimer`、`%smid`）2 个、asm 内访存 1 个 | 成立 |
| 逐签名测试 A–G 每项三类齐全，违反 0；回归不变 | `2_tool/tests/test_signatures_amd.py` 29 条（官方 TTGIR fixture，见 `2_tool/tests/data/amd_ttgir/README.md`）通过；全套 2930 通过、0 失败（另有 15 个旧测试文件因缺 transformers 收集失败，与之前相同）；W1 仍为 421/421 | 成立 |
| 设备验证 0（待做） | 0 | 成立（按登记） |

### 3. 未登记的补充运行

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

### 4. 逐签名测试的内容

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

### 5. 实现中的说明

- `buffer_load` 没有 `other` 时，被 mask 的 lane 与 `tt.load` 一样未定义（登记 A 项）；`buffer_load_to_local` 沿用增量 10 的异步拷贝
  模型，被 mask 的 lane 按增量 10 已记录的偏离零填充（登记 D 项「同一异步拷贝模型」）。
- E8M0 scale 按 bf16 载体位型的指数域解码。编译器的载体是 `max(e << 7, 64)`：e = 0 时得到 bf16 次正规数 2^−127，指数域仍为 0，
  与 2^(e−127) 一致。
- 逐签名测试中一个 fixture 的生成方式出了错（最初用 `strip_locations` 去掉注释时也删掉了布局别名，归约因此拿不到 TTGIR 次序）；
  改为只去掉 `#loc` 行与 `loc(...)` 后缀。不影响求值器与评价运行。

### 6. 没有做或不能说的

- 没有设备证据：gfx942 / gfx950 的结论都只是「CPU 参照与 TTIR 一层一致」，不是设备上的正确性。官方 AMD 后端的 LLVM 代码生成在本机
  不能运行，TTGIR 之后的下降（包括 buffer 操作越界返回 0 的硬件语义）没有核对。
- TDM、AMD 的 mbarrier / async_wait、`masked_load` / `masked_store`、`scaled_downcast_*`、`local_load_packed_transposed`、
  `extract_slice` / `concat` 等其他官方 AMD 操作在这批官方 TTIR 的 gfx942 / gfx950 编译中没有出现，留在欠账表；RDNA（gfx11 / gfx12）
  与 gfx1250 没有编译。
- AMD Gluon 子模块没有在本机运行（需要 AMD 设备）。

## 第十三个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 increment_13.md（登记，提交 `c9472ad`）。数字是开发证据，**不是盲测成绩**。
成绩只按三种状态写：成立 / 不成立 / 无法判断。设备验证：0（本机只有 sm_86，sm_100 路径只编译不运行，记为待做）。

### 1. 两层对照（合成捕获）

5 个官方 kernel 的 TTIR（W0 dump）与官方主线为 sm_100 编出的 TTGIR（fixture 见 `2_tool/tests/data/nvidia_ttgir/README.md`），在同一组随机
正例输入上各求值一次（`1_experiments/dsl_v2/captures/inc13_scaled_mma_synthetic.json`，脚本 `inc13_scaled_mma_eval.py`）：

| kernel | 输出元素 | TTGIR 完整 | TTIR 完整 | 都完整 | 不相交 | 执行的 tc_gen5_mma_scaled |
| --- | --- | --- | --- | --- | --- | --- |
| simple_dot_mxfp（e4m3 × e4m3） | 16,384 | 16,384 | 16,384 | 16,384 | 0 | 1 |
| mxfp8_mxfp4_matmul_tma，BLOCK_N 128，1 级流水 | 16,384 | 16,384 | 16,384 | 16,384 | 0 | 2 |
| 同上，BLOCK_N 128，3 级流水 | 16,384 | 16,384 | 16,384 | 16,384 | 0 | 3 |
| 同上，BLOCK_N 256，1 级流水 | 32,768 | 32,768 | 32,768 | 32,768 | 0 | 2 |
| 同上，BLOCK_N 256，2 级流水 | 32,768 | 32,768 | 32,768 | 32,768 | 0 | 2 |

### 2. 预期对照

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| 5 个 kernel 的 TTGIR 参照在正例输入下全部完整，与 TTIR 参照都完整的元素不相交 0；选定元素包住精确值 | 114,688 个元素全部两层完整，不相交 0；逐签名测试在选定元素上与精确有理数比较通过 | 成立 |
| sm_90 / sm_100 名字覆盖 129/129 | 129/129 | 成立 |
| 逐签名测试三类齐全，违反 0；回归不变 | `2_tool/tests/test_signatures_nvidia_scaled.py` 18 条通过（e4m3 × e4m3 与 4 个 e5m2 × e2m1 流水线变体，各三类；去掉 barrier wait 的变体全部未建立；two_ctas / multicast 变体中止）；全套 2948 通过、0 失败（15 个旧测试文件收集失败同前）；W1 仍为 421/421 | 成立 |
| 设备验证 0（待做） | 0 | 成立（按登记） |

### 3. 说明

- 解码只有一份：`tt.dot_scaled` 的实现抽成 `_scaled_dot`，两个操作共用。
- 边界类用了 E8M0 = 0 / 254 与 e4m3 的 ±448、e5m2 的 57344；前提违反类用 E8M0 = 255，相应的行或列不完整。
- 这些 kernel 的官方测试要求 capability ≥ 9 / 10，本机不能运行，所以只有合成捕获，没有官方测试启动的捕获。

### 4. 没有做或不能说的

- 没有设备证据：结论只是「CPU 参照与 TTIR 一层一致」。
- `two_ctas`、`multicast`（操作数分布在一对 CTA 上）没有出现在这批 kernel 中，没有模型：带这两个属性的 tcgen05 MMA（含增量 11 的
  `tc_gen5_mma`）现在中止并写明原因（实现时发现增量 11 漏了这一检查，一并补上；有逐签名测试）。

## 第十四个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 increment_14.md（登记，提交 `3fb81bf`）。数字是开发证据，**不是盲测成绩**。
成绩只按三种状态写：成立 / 不成立 / 无法判断。

### 1. 复跑（官方主线，sm_86）

官方 `test_core -k atomic`（经典，933 次启动）与官方 Gluon `test_core -k atomic`（227 次启动）在增量 14 的求值器下复跑，按启动逐一
与增量 6（`inc6_combined.jsonl`）、增量 10（`inc10_gluon_test_core.jsonl`）的状态比较（`inc14_compare_*.json`）。

| | 经典 test_core | Gluon test_core |
| --- | --- | --- |
| 原来唯一原因是「整数原子返回值是集合」的启动 | 240 | 12 |
| 其中现为「完整（集合目标）」 | 180 | 12 |
| 其中仍为部分参照 | 60（见下） | 0 |
| 原来完整、现在状态改变 | 0 | 0（另有 4 次仍完整，但新标为集合目标，见下） |
| 复跑总状态 | 完整 529、完整（集合目标）260、部分 84、无写入 60 | 完整 175、完整（集合目标）16、部分 36 |

另有经典测试 10 次启动在增量 6 时中止、现为完整 6 / 部分 4，是增量 7–13 的变化，不是本增量。

**仍为部分参照的 60 次**都是浮点 `test_atomic_rmw` 的 max / min（float32、float64，各 3 种模式，每种 5 次启动）。前端把浮点
atomic max / min 下降为整数位型上的 max / umin 原子操作，再用 `arith.select` 选出结果并 `bitcast` 回浮点，所以增量 6 把它们记在了整数原因下。
现在这些位型返回值是整数集合，`select` 没有集合规则，相应 lane 记未建立。即使给 `select` 和 `bitcast` 加上集合规则，结果也不会变：
测试中地址的初值是 −inf（max）或 +inf（min），返回值集合同时含有这个无穷大和有限值，浮点区间包围表示不了。所以它们仍不建立是对的，
但登记时把它们算进了「会变完整」的范围，登记有误。

**Gluon 中新标为集合目标的 4 次**是浮点 local atomic add（`add_unmasked_float16/32_*_axis1_rhs_cols`）。返回值本来就是各交错下的区间，
增量 10 没有给它们加集合目标标记；本增量改写 local atomic 的返回路径时补上了，值不变。

### 2. 预期对照

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| 原来唯一原因是整数集合的启动全部变为「完整（集合目标）」，只有包围越出整数宽度时仍记未建立；原来完整的启动状态不变 | 经典 180/240、Gluon 12/12；剩下的 60 次不是宽度问题，是浮点 max / min 的位型下降（见上），按登记算未达到；原来完整的启动状态都没有变 | 不成立 |
| Gluon local atomic 中整数返回值是集合的启动变为「完整（集合目标）」 | 12/12 | 成立 |
| 独立模拟器枚举的所有执行中实际返回值都落在参照区间内（违反 0）；各类误用不给出点值或集合声明 | `2_tool/tests/test_signatures_int_sets.py` 46 条：add（int32 / int64 / uint32）、max / min / xchg（int32、uint32）、and / or / xor 对全部执行顺序（或全部子集）枚举，违反 0；回绕、可达值超过 4096、算术、地址、循环边界、存入后被 store / atomic 覆盖、未决分支只有一支写入、下一次启动覆盖，全部不给出点值或集合声明 | 成立 |
| 逐签名测试三类齐全，违反 0；回归不变 | 同上；全套 2994 通过、0 失败（15 个旧测试文件收集失败同前）；W1 仍为 421/421 | 成立 |

### 3. 实现中的说明

- 整数值在求值器里是按 2^宽度 取模保存的，有符号、无符号两种表示都可能出现。集合的包围与点值判定现在都先换成有符号表示再比较。
  原来点值判定直接比较，无符号存储上的有符号 max / min 可能误判为点值。
- 原子返回值与原子读的整数结果原来存在 float64 数组里，int64 的大值会丢精度。现在保持 int64。
- `scf.if` 两支合并时，整数集合的上界也参与「两支是否相同」的判断。原来只比下界，若存在整数区间就会被误当成点值。整数区间是本增量才引入的，以前不会出现。
- 未纳入（按登记）：check / measure 对整数输出的统计不变；后续启动读到这些元素时仍是未建立。

### 4. 没有做或不能说的

- 浮点 atomic max / min 的前端下降（位型 max / umin + select + bitcast）没有集合规则；含 ±inf 初值时本来也表示不了。
- 整数集合用区间包围，不是精确集合（例如 add 的子集和可能有空洞）。rc3 §10 允许这样做，代价是残差界更宽。

## 第十五个增量：结果（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中，未发布、未冻结）。登记见 increment_15.md（登记，提交 `0439b6c`）（第 5 节是实现中、评价运行前的更正：
整数表示缺陷实际在 CAS 路径，load 本来就换成有符号表示）。数字是开发证据，**不是盲测成绩**。成绩只按三种状态写：成立 / 不成立 / 无法判断。

### 1. 评价运行

增量 6 的全部官方测试文件在增量 15 的求值器下复跑（`inc15_broad.jsonl`；审计修复版之后被 `auditfix_broad.jsonl` 替换，原文件在提交 5fe8e56 的 `1_experiments/dsl_v2/captures/`），逐启动与增量 14 之后的状态运行
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

### 2. 预期对照

| 登记的预期 | 结果 | 状态 |
| --- | --- | --- |
| `test_bin_op` 的 8 次 `remf` 部分参照变为完整 | 8/8 | 成立 |
| `test_libdevice_rint[float64]`、`test_clz`、`test_popc` 变为完整 | 3/3 | 成立 |
| int8 `%` 那次仍为部分参照，但写明原因 | 是 | 成立 |
| A 的复跑中没有原来完整的启动变为部分或中止 | 0 次 | 成立 |
| 若有启动的值因 A 改变，逐个列出 | 捕获记录只保存状态，不保存参照值，无法逐个比较值；使用 CAS 的启动状态都没有改变。A 对值的影响由逐签名测试证实（见下） | 无法判断 |
| 逐签名测试违反 0；W1 支持条目增加后仍全部三类齐全并有触发证据 | `2_tool/tests/test_signatures_inc15.py` 40 条（及新 libdevice 条目由已有生成器得到的 6 条）通过；W1 430/430（新增 `math.ctlz / cttz / ctpop` 与 6 个 libdevice 条目）；全套 3040 通过、0 失败 | 成立 |

### 3. 缺陷与测试

- CAS：修复前，int64 的 2^62 + 1 与比较值 2^62 在 float64 中相等，参照会记一次并不存在的交换；uint32 存储中 0xFFFFFFFE 与比较值 −2
  （同一位型）永远不等，参照会漏掉交换；返回的旧值经过 float64，大 int64 丢精度。三条 CAS 测试在修复前都失败，修复后通过。
- 整数未定义行为（除零、INT_MIN / −1、移位越界）原来记未建立但不写原因；现在写明。
- `test_signed_operations_on_unsigned_storage` 在修复前后都通过：它证实 load 路径本来正确，也就是第 5 节更正的依据。

### 4. 没有做或不能说的

- bessel_y0 / y1 的负数输入与 `remf` 除数为 0：按类别 D 的既定规则记未建立。改成 IEEE 的 NaN / −inf 是语义改动，需审阅方决定。
- 浮点 atomic max / min 的位型下降、同址混用原子种类、argmax 带 NaN 的 lane 不一致、同一 program 内两次原子返回值的大小关系、
  原子返回值作为 `cf.cond_br` 条件、asm 内访存：未处理（原因见登记第 1 节）。
