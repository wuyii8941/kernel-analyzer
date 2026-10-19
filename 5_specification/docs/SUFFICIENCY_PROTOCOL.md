# 规格是否够用：评价方案（事前登记，2026-10-19）

本方案在任何一项测量之前提交；提交后的改动记为偏差（本文末尾），不覆盖原文。评价代码在
`5_specification/sufficiency/`，结果写到 `5_specification/sufficiency/results/`（每次运行新目录，不覆盖）。

## 0 承诺与不做的事

- **分母包括一切尝试分析的任务**，没有生成报告的也算：编译失败、捕获失败、超预算、绑定被拒，都以原因留在分母里。
- 每个任务的结论只有两种合法形态：在有原理说明的准入范围内有 f；或明确写「没有 f」及原因。没有独立 f 时，
  只要工具自身条件成立仍可报告 K−Gπ，但不升级为语义裁决。
- **不设「够用」阈值**（任务负责人 2026-10-19 决定）：本评价只报告各项比例、分组和失败原因，不下「已经够用」的结论。
  论文只能写「在 X% 的〈计数单位〉上可用」。
- 四项评价各证明一件事，不重复证明同一个覆盖率：

| 评价 | 证明什么 | 不能据此证明什么 |
|---|---|---|
| E1 真实工作负载 | 现有规格能支持多少实际编译调用 | 全部 Triton kernel 都已覆盖 |
| E2 外部算子集 | 相对 PyTorch Core ATen，哪些已支持、哪些缺失、缺在哪一层 | 名称对应就有完整 ground truth |
| E3 冻结后的新实现 | 不补规格、不改分析器时，已有能力能复用多少 | 换随机种子就证明陌生结构泛化 |
| E4 机制区分与正常对照 | 预定机制下错误能否识别，合法行为是否不被误判 | 每类有一个成功案例就覆盖了整类错误 |

## 1 冻结对象

冻结点 = 提交本文件的 commit（记为 F）。冻结对象：

- `5_specification/contracts/`、`adapters/`、`kernel_spec/`、`validated/`、`SOURCES.json`、`sources/`（规格与参照后端）；
- `kernel_spec/registry.py` 的准入规则；
- 自动绑定逻辑：`corpus/prov_model.py`、`corpus/graph_provenance.py`、`corpus/bind_inductor.py`、
  `corpus/dynamic_three_way.py`、`corpus/call_binding.py`、`corpus/region_*.py`、`corpus/check_region.py`；
- 工具：`2_tool/src/kernel_analyzer/`（K−Gπ 核对 `reference_eval/emulate.py:verify` 与程序选择）。

E3 只用冻结点 F 的代码运行（`git worktree` 于 F）。E1 也在 F 上运行。F 之后若修工具，修后的结果单独报告，不并入 E1/E3。

## 2 计数单位

- E1、E3 的编译部分：**编译实例**（一次捕获的 kernel 启动 = 一个单元），称「按编译实例计」。我们没有调用频次或耗时
  记录，因此不称「使用加权」。
- E3 的库部分：**库调用配置**（库自身测试参数网格中的一个组合）。
- E2：**Core ATen overload**（锁定 torch 2.10.0 的 `torch.Tag.core` 条目），另报基本名计数。
- E4：**判定实例**（一个候选输出对一个任务）。

## 3 E1 真实工作负载

工作负载：three_way 的 23 个开发模型 × {bf16, fp32tf32}，与 three_way 相同的训练步骤（2 层、128 token、
compile 设置见 `1_experiments/three_way/capture_model.py`）。旧捕获没有同进程的编译来源记录，因此用
`prov_model.py --capture-dir` 重新捕获（同进程取得来源与第二步的实际启动）。

流水线（全部为冻结代码，评价脚本 `sufficiency/run_compiled.py` 只负责调度与超时）。每个模型配置使用独立的全新
Inductor 缓存目录，静态候选只在该目录内匹配（同一模型配置内的多来源歧义照常计入）。模型权重读自 three_way 使用的
`/data1/tzh/models`（只读）；E3 新模型下载到 `.cache/models/`。

1. 捕获：`prov_model.py MODEL OUT.jsonl --config C --capture-dir D`；
2. K−Gπ 核对：`emulate.verify`，程序选择与 three_way 的 `analyze_units.programs_for` 相同（`sufficiency/verify_units.py`）；
3. 候选：`bind_inductor.py --units --capture-root --code-dir`；
4. 绑定与参照：`dynamic_three_way.py --bindings --full-output-limit 4194304 --capture-root --provenance-root`，
   每个单元一个子进程，**预算 600 秒墙钟**，超时记「超预算」。

三个指标（同时报告，主指标是第二项）：

| 指标 | 定义 |
|---|---|
| 候选覆盖率 | 来源唯一，且图片段中每个运算都有合同名称对应 |
| **完整参照可用率** | 绑定完成（输入、输出、状态全部对应），每个输出在预算内得到全部坐标的参照，参照结果为 value / domain_undefined（不是 not_established / unsupported），且所用合同已通过准入（validated）。GRAPH 合同目前是 drafted，因此另报一列「草案 GRAPH 下的完整参照」，明确标为开发级 |
| bias 查询可用率 | 在上一项基础上进入合同测量模式（`measure.py` contract）并满足其统计条件 |

候选 kernel 算错但参照可靠并检出了差异，仍计为「参照可用」。

抽样：候选覆盖率用全部单元；完整参照与 bias 查询对全部单元按种子 20261019 简单随机抽取 400 个（不足则全部），
报告比例与 Wilson 95% 区间。

分组（事先固定）：按模型；按 kernel 结构（`triton_poi_`、`triton_red_`、`triton_per_`、`triton_tem_`、其他）；
按语义类（图片段含：矩阵乘、归约、softmax / logsumexp、归一化、索引 / scatter、随机、原地状态写回；可多选）。

## 4 E2 外部算子集（PyTorch Core ATen）

脚本 `sufficiency/core_aten.py`（冻结点 F 的版本），三层分开报告：

1. **清单**：锁定 torch 2.10.0 中带 `torch.Tag.core` 的全部 overload 及其 schema。
2. **名称对应**：合同 covers 中有 `aten.<name>`，或经登记的别名（脚本内 `ALIASES`，每条写理由）。
   不做算术的条目不删除，分为「结构／存储语义承担（需核验是否已可靠处理）」与「随机流」。表述固定为：
   「在锁定版本的 Core ATen 清单中，N/192 个 overload 存在合同名称对应；完整签名与适用范围另行核验。」
   不称「可信覆盖率下界」：别名未识别会低估，部分 dtype / 参数 / 反向不支持会高估。
3. **适用范围探测**：用 PyTorch 自己的 OpInfo 样例输入（CPU，dtype ∈ {f32, bf16, f64, i64}，每个 overload 每个 dtype
   至多 50 次调用，按样例顺序），在分派记录下取得实际调用的 Core ATen overload 与参数，原样交给
   `registry.query`（只用已准入合同，取第一个不需要任务选择的 profile）。逐 overload、逐 dtype 统计结果类别；
   value 另与 float64 PyTorch 结果做数值复核（诊断，不计入相容）。

## 5 E3 冻结后的新实现

### 5.1 新模型

选法写在 `sufficiency/select_heldout.py`，结果 `sufficiency/registration/heldout_models.json`（在本文件之前生成，
只看过名单，没有看任何支持情况）：Hugging Face 按下载量排序，text-generation 或 fill-mask，transformers 原生架构，
不 gated，有 safetensors，≤ 2B 参数，架构族不同于 23 个开发模型及已选模型，取前 6 个。结果为：
xlm-roberta-base、distilbert-base-uncased、esm2_t33_650M、ModernBERT-base、camembert-base、albert-base-v2。
**局限**：六个都是编码器（fill-mask）架构；解码器候选多因架构族已用、超过 2B 或缺参数元数据被规则排除。
这一点按原样报告，不改规则补选。

流水线、预算与 E1 相同，单元全部计入（不抽样）。

### 5.2 新库

cut-cross-entropy 25.1.1（已安装于 liger 环境，仓库此前从未使用；PyPI sdist sha256
`5fe5924509248b1aea5c890f8887c6a7759f7c8b1ebc0490e42c247c4f7c1e34`，与安装文件逐字节相同）。调用集合 = 该版本 sdist
自带测试的参数网格：`tests/test_cce_loss_forward.py`（144 个组合）与 `tests/test_cce_loss_backward.py`（288 个组合），
输入按测试中的生成方式（`torch.cuda.manual_seed(0)`）。两种 impl（cce、torch_compile）都计入。

### 5.3 五类结果（按流水线顺序取第一个阻断点）

| 结果 | 含义 | 编译单元的判定规则 |
|---|---|---|
| 原样可用 | 不改规格和工具，按既定方式取得完整参照 | 第 3 节「完整参照可用」成立 |
| 需要人工绑定 | 数学定义已有，但需要新增对应说明 | 图运算都有已准入合同对应，但绑定被拒（来源歧义、来源记录不全、类型 / 形状 / 存储对应未建立、缺占位、元组输出） |
| 需要扩展实现 | 合同已有，但求值或导入能力不够 | 运算有对应但合同未准入（drafted）、参照返回 unsupported / not_established、或工具前提不满足（K−Gπ 未逐位一致导致参照被拒） |
| 需要新增／修改任务合同 | 原有语义确实没有涵盖 | 图中有运算没有任何合同名称对应 |
| 运行或证据不足 | 编译失败、资源不足、来源不明 | 捕获失败、kernel 源缺失、无图片段、超预算、执行错误 |

库调用按同样五类，规则：有已准入合同且有现成绑定计划 → 原样可用；有已准入合同无绑定计划 → 需要人工绑定；
对应合同未准入或参数（如 softcap、shift、filter_eps）不在已准入 profile 内但合同已写 → 需要扩展实现；
合同没有写到的任务语义 → 需要新增／修改任务合同；运行失败 → 运行或证据不足。

参与补规格的样本（外部批次的开发集）不计入 E3 首次遇见的成功率。

## 6 E4 机制区分与正常对照

机制分类（事先固定，不声称是天然全集）：

| 机制 | 错误实例举例 | 正常对照（必须不被判错） |
|---|---|---|
| M1 公式与常数 | 系数、近似式写错 | 正确实现；数学等价的另一写法 |
| M2 归一化分母 | 除以 N−1 / 全长代替有效长度 | 正确分母；不同累加次序 |
| M3 掩码与空行 | 掩码漏用、空行处理错 | 正确掩码；空行的合法输出 |
| M4 并列与选择 | 返回允许集合之外的索引 | 允许集合内的另一种选择 |
| M5 状态写回 | 漏写或错写状态缓冲 | 完整正确写回 |
| M6 布局与 stride | 忽略 stride 当作连续 | 正确的非连续实现 |
| M7 读法歧义 | （不是必然应检出的 bug）任务未选读法时强行裁决 | 规格允许的多种读法都应被接受，或正确拒绝作唯一裁决 |

证据来源分开统计，不合并成同一种泛化证据：
- **S1 已知缺陷回归**：`4_bugs_cases` 及外部批次开发集中发现的缺陷；
- **S2 合成变异**：每类按事先列出的变异（`sufficiency/e4_cases.py`），在已准入合同上用 `registry.check_output` 判定；
- **S3 未参与开发的真实缺陷**：只计验收集中首次发现、且没有据此改规格的缺陷。

变异与对照的具体清单（`sufficiency/e4_cases.py`）在 E4 任何测量之前单独提交，提交号记入第 7 节。

指标：每类、每来源的检出率（判矛盾）、误报率（对照被判矛盾）、未建立率，分别报告；样本数写明。每类至少 10 个错误实例
与 10 个正常对照（S2）；S1、S3 有多少报多少。

## 7 偏差记录

（提交后追加，不改上文。）
