# 外部受控集合评价协议：The Correctness Illusion 公开包（M2，2026-10-06，运行前写定）

本文在接触该集合的任何运行结果之前提交。之后的改动只追加到第 10 节「偏离记录」，写明时间与原因，不改前文。

## 1. 对象

- 来源：`github.com/sarkar-dipankar/gpuemu-corpus`，提交 `2f15310`，许可 MIT OR Apache-2.0；论文 arXiv 2606.20128。
  本地副本 `.cache/src/gpuemu-corpus/`（不入库）；每个被测文件的 SHA256 写入结果清单。
- 26 个条目按实际内容计数：
  - Triton 22 个（控制组 13：attention、elu、flash_attention、gelu、l2norm、leaky_relu、matmul、relu、rmsnorm、
    sigmoid、silu、softmax、tanh；植入变体 9：attention、flash_attention、gelu、l2norm、leaky_relu、matmul、rmsnorm、
    silu、softmax）。走统一入口的模式 B（K_R、e_num、e_sem、e_total）。
  - NumPy 4 个（layernorm、matmul、softmax 控制组与 softmax_llm_buggy）：没有 TTIR，只做黑箱模式（工作项 C：
    只用 f 作参照，测 K − f），结果单列，不计入分解的成绩。
- 精度：本轮只测 float32。float16 不在本轮。

## 2. 条件与单位

- 条件 = （条目，形状）。形状取自各条目 `op_schema` 的候选值的全部组合，保留输出元素 ≤ 4096 且输入元素合计
  ≤ 65,536 的组合：Triton 22 个条目共 699 个条件，NumPy 4 个共 141 个。
- 取值与原 fuzzer 相同：各输入独立 U[−10, 10]，转为 float32。每个种子
  `np.random.default_rng([seed, crc32(条件名)])`，按 schema 的输入顺序生成。单位 = 种子；同一基础输入的变换不作独立单位。
- 第一层（RQ1、RQ2、基线）：全部条件，种子 0–7（开发 0–1，确认 2–7）。
- 第二层（RQ3）：每个条目取保留条件中输出元素最多的一个（并列时取末维更大者），种子 0–95（开发 0–31，确认
  32–95）。由此规则选出的条件见 `scripts/external_corpus.py --list`。
- 预算回退：第一层前 10% 的条件若推算总时长超过 8 小时，全部条目的上限改为 2048 个输出元素，记入第 10 节。

## 3. 规格 f

- f = 作者的 `ref_fp64.py`，在进程内调用（替换其 `read_inputs` / `emit`），输入以 float64 传入，所以输出是 float64
  而不是舍入回输入精度的值；以 `f64_point_spec`（声明误差界 2⁻⁴⁰·max|f|，筛查级）进入比较。f 是作者写的规格，
  不是从候选代码生成的。
- 交叉核验：我们另按算子名与参考脚本中的常数（eps 等）独立写一份 torch float64 公式，在第一层全部输入上与 f 比较，
  相对差 > 10⁻¹² 的条件单独报告，不改 f。

## 4. 标签（RQ2），由植入改动的数学在运行前推出

| 条目 | 改动 | 语义差异存在的条件（阳性） | 不存在的条件（阴性） |
|---|---|---|---|
| attention_triton_buggy | 缺 1/√D | N > 1 | N = 1（一个键的 softmax 恒为 1） |
| flash_attention_triton_buggy | 最大值更新时累加器未重标 | N > BLOCK_N，即 N ∈ {64, 128, 256}（每行在后续分块中最大值增大时触发，概率随数据，每行 ≥ 1/2） | N ∈ {1, 3, 16}（单分块） |
| gelu_triton_buggy | 缺 0.5 | 全部 | — |
| l2norm_triton_buggy | 缺 sqrt | 全部 | — |
| leaky_relu_triton_buggy | α = 0.1 而非 0.01 | 全部（单元素条件下每个种子以 1/2 概率触发，8 个种子） | — |
| matmul_triton_buggy | K 循环内赋值而非累加 | K > 1 | K = 1 |
| rmsnorm_triton_buggy | 缺 sqrt | 全部 | — |
| silu_triton_buggy | sigmoid(2x) | 全部 | — |
| softmax_triton_buggy | 越界填 0 而非 −∞ | H ∈ {3, 1025}（BLOCK > H） | H = 256 |
| 13 个控制组 | 无植入改动 | — | 全部；「正确」只相对作者的规格与容差协议，不是零均值标签 |

- 工具在一个条件上的结论分三种（判定由 `scripts/tool_spec_summary.sem_bin` 给出，不另写）：
  - 可靠地不一致：e_sem 落在「候选」档（相对 RMS ≥ 10⁻⁵，由规则、检测器或区间排除 0 确认）；
  - 未能排除相等：「无」「常数取整级」（< 10⁻⁷）「小」（10⁻⁷–10⁻⁵）；后两档单独计数，不算阳性也不算误报；
  - 未建立：参照不完整（完整比例 < 1）、program 中止、报错或「判不了」。
- 指标：阳性条件上「可靠地不一致」的比例；阴性条件上「可靠地不一致」的比例；未建立的比例（分母含报错）。

## 5. 基线（同条件、同输入、同样本数）

每条基线报告默认配置与事先登记的扫描。扫描的工作点在控制组（与植入变体共用参考的那一个）的开发种子上选定：取
在全部条件开发种子上零告警的最严参数；在确认种子上对全部条件报告。条件的告警 = 任一确认种子告警。

| 基线 | 判定 | 默认 | 扫描 |
|---|---|---|---|
| B1 作者的容差判定 | 逐元素 \|K − round₃₂(f)\| > tol 或出现 NaN / inf | meta.json 中 float32 的 tol | tol × {10⁻³, 10⁻², 10⁻¹, 1, 10} |
| B2 allclose | 逐元素 \|K − f\| > atol + rtol·\|f\| | torch.testing 的 float32 默认（rtol 1.3·10⁻⁶，atol 10⁻⁵） | 两者同乘 {10⁻¹, 1, 10, 10², 10³} |
| B3 TTrace 式阈值 | rel-L2(K, f) > c·max(rel-L2(T₃₂, f), 2⁻²⁴)，T₃₂ 为我们按规格写的 torch float32 eager 实现 | c = 2（我们的取值，不是 TTrace 发表的参数；TTrace 原代码未在这些 kernel 上运行，故称「TTrace 式」） | c ∈ {1, 2, 4, 8, 16, 64} |
| B4 随机算术式 | 本轮不做（需在逐位模拟器中实现随机舍入），记为尚未运行 | — | — |

计时：每种方法记录自己的耗时（B1–B3 计入 f 与 T₃₂ 的计算；工具计入编译、捕获、参照、规格与统计）。

## 6. 消融

- A1 参照来源：K_R 换成同一 Triton kernel 的 float64 重跑 K₆₄，e_num′ = K − K₆₄ 走同一判定层。只对内部不固定
  fp32 且不用 `tl.dot` 的 kernel 做（gelu、leaky_relu、relu、rmsnorm、softmax 及其植入变体，9 个）；其余条目以
  fp64 eager 的 f 代替（即 e_total），单独成表，不合并。
- A2 判定规则：均值检验换成幅度判定——e_num（或 e_total）的任一元素 \|中点\| > B1 的默认 tol 即告警。
- A3 判定层：本集合只有前向输出，没有更新层，不适用；在 M4 的数据上做。

## 7. RQ1：参照的可靠性与成本

- 参照类别分布（分母含报错、中止与未建立）、K_R 区间宽度分布、按阶段的耗时（`timing_seconds`）。
- 独立复算：按每个 kernel 的源码，用不调用工具代码的 NumPy float64 写出其声明语义（实数运算，常数取 kernel 实际
  使用的 float32 值，包括植入的改动），在第一层全部条件的种子 0–1 上检查是否落在
  [K_R 下界 − s, K_R 上界 + s]，s = 2⁻⁴⁰·max\|K_decl\|。违反数按条目报告。
- FPCore 交叉复算：代表性纯片段另行登记（工作项 B），不在本协议内。

## 8. RQ3：系统性作用（第二层）

- 报告 e_num 与 e_total 的规则判定、符号与效应量；按统计校准文档第 4 节的规则处理偏斜（|偏度| ≥ 1 以 bootstrap-t
  伴随区间为准；|偏度| ≥ 2 且确认单位 < 64 记为判不了）。
- 信息差异计数：(a) B1 默认容差下全部确认种子通过、但某条规则检出 e_num 或 e_total 的平均作用；(b) B1 告警、但没有
  规则检出平均作用。只报计数，不称基线的漏报或误报。控制组不作为平均作用的阴性，不报误报率。
- 等价轴按评价计划第 5 节的候选 δ（输出层 2⁻²⁴·q_R）计算，标为「δ 待确认」，不进入任何结论表。

## 9. 产出

`results/external/gpuemu/`：`manifest.json`（提交、文件哈希、条件清单）；`tier1/`、`tier2/`、`blackbox/` 下每个
条件一个工具报告；`baselines.jsonl`；`ablation_a1.jsonl`；`declared_check.json`；`summary.md`（RQ1–RQ3 三张同条件
对照表）。状态写入 `docs/status_ledger_20261006.md`。

## 10. 偏离记录

（运行后追加。）
