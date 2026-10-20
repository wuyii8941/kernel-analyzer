# 路径 A 结果：FlagGems 5.0.2 在真实训练步上的算子入口覆盖（2026-10-20）

登记协议：`docs/OPERATOR_ENTRY_PROTOCOL.md`（F_A = c09b655），偏差 D-A1…D-A5 见协议 §9。
运行根目录：`.cache/operator_entry/A_20261020/`；本目录保存 `analysis.json`、逐键结果 `keys.jsonl`、
检出与 D3 的测量报告 `reports/`、运行日志 `log.jsonl`。

环境：A6000（sm_86），torch 2.10.0+cu128，Triton 3.6.0，transformers 5.19.0，FlagGems 5.0.2 注册在 CUDA 分派键上。
统一排除 `copy_`、`to_copy`、`index`（§2），另按 D-A3 为各工作负载额外排除运行失败的函数（见第 4 节）。

## 1 执行对象核查

| 核查 | 结果 |
|---|---|
| T1 记录器是否改变启动序列 | 11 个工作负载全部一致；所有 Triton 启动都落在记录到的 ATen 调用内（调用外启动 = 0） |
| T2 重放是否启动同一 kernel 序列 | 进入生成的键中没有不一致 |

## 2 W1：六个模型配置，317 个候选键（908 次调用）

候选键 = 调用期间至少有一次 Triton 启动的（overload，参数结构）。按第一个阻断点归类：

| 类别 | 键数 | 键占比（Wilson 95%） | 调用数 | 调用占比 |
|---|---:|---|---:|---:|
| **原样可用** | 126 | 39.7%（34.5–45.2%） | 526 | 57.9% |
| 运行或证据不足 | 92 | 29.0%（24.3–34.3%） | 165 | 18.2% |
| 需要新增合同 | 65 | 20.5%（16.4–25.3%） | 134 | 14.8% |
| 合同范围外 | 30 | 9.5%（6.7–13.2%） | 60 | 6.6% |
| 需要规格匹配（第二步） | 4 | 1.3%（0.5–3.2%） | 23 | 2.5% |
| 需要扩展生成规则 | 0 | — | 0 | — |

单个训练步的数值，同一键的各单元相同，所以区间只描述「这批键」的抽样不确定性，不对 FlagGems 整体作统计推断。

### 原样可用的 126 个键

| 结论 | 键数 | 合同 |
|---|---:|---|
| 相容（参照完整，没有坐标的 Gπ−f 排除零） | 118 | ARITH 62，REDUCE 19，MATMUL 18，INDEX 6，LAYOUT 4，ACT 4，MATHFN 3，SOFTMAX 2 |
| 检出 | 8 | ACT/torch（`aten.gelu`）4，RANDOM/finite_backward（`aten.native_dropout_backward`）4 |

**8 个检出都是低于输出精度的常数差，不是缺陷：**

| 键 | 排除零的坐标 | `max_gap_relative` | 来源 |
|---|---:|---:|---|
| gelu，bert-tiny fp32/bf16 各 2 个键 | 49152–196413 | 2.3e-9 – 4.5e-9 | 很可能来自 FlagGems 程序里的近似常数 `0.7071067811`（1/√2 的截断，源码 `ops/gelu.py:20`）；未逐位归因 |
| native_dropout_backward，bert-tiny 与 gpt2 各配置 | 43719–263232 | 2.4e-10 – 6.2e-10 | 很可能来自缩放因子 `scale` 作为 Triton 标量参数传入时的表示（`ops/dropout.py:91`）；未逐位归因 |

相对差比 fp32 的半个 ULP（约 6e-8）小一到两个数量级，比 bf16 的小五个数量级以上。这与 E4 中「舍入常数造成小幅检出」的
机制相同：检出说的是「程序声明的计算不是 f」，这在字面上成立；但它不是可观察的错误。按「不设阈值，只报告」的约定，
保留为检出，并在这里写明幅度和来源。

### 其余 191 个键的阻断原因

| 类别 | 原因 | 键数 |
|---|---|---:|
| 运行或证据不足 | 参照求值超出 600 s 预算：MATMUL 54（`mm`、`addmm`、`bmm`），SOFTMAX 12（`_log_softmax` 及其反向），ARITH 3，INDEX 2，ACT 2 | 73 |
| | 捕获大小上限 200 MB | 7 |
| | 记录的参数不可序列化（`ones_like` 的参数含 module 对象，D-A2） | 6 |
| | 绑定：公开输出依赖计划未覆盖的张量状态（`cat` 4，`sum` 2） | 6 |
| 需要新增合同 | `embedding_backward` 12，`pow` 10，`pad` 6，`nll_loss_forward` 6，`nll_loss_backward` 6，`slice_backward` 6，`native_dropout` 4，`arange` 4，SDPA 前向 / 反向 4，`zeros` 3，`scalar_tensor` 2，`ones` 1，`repeat_interleave` 1 | 65 |
| 合同范围外 | ARITH：bf16 调用里的标量不能用 bf16 精确表示 | 10 |
| | MATHFN：`cos`/`sin`/`tanh` 输入幅度超过 100 | 5 |
| | ARITH i64 4，INDEX i64 2，LAYOUT bool 1 | 7 |
| | INDEX：embedding 非默认 `padding_idx` | 2 |
| | ARITH：`where` 的分支含无穷 | 2 |
| | ACT：`gelu_backward` 参照不完整（统计不足，complete_rate 为空） | 4 |
| 规格匹配（第二步） | NORM 合同接口不是 ATen 调用（`native_layer_norm` 及其反向） | 4 |

两个最大的缺口：
1. **求值预算**：73 个键，其中 MATMUL 占 54 个。大矩阵乘的区间参照在 600 s 内算不完。这是求值器的问题，不是合同缺失。
2. **新增合同**：65 个键，集中在 13 个 overload 上。其中 `embedding_backward`、`nll_loss`、`pad`、`slice_backward`、
   `pow` 五族就占 46 个键。

## 3 W2：已知缺陷触发

| ID | 事先的期望 | 结果 |
|---|---|---|
| D20 `argmax` 非连续，dim=None | 检出 | **找到**：REDUCE/real，3 个坐标（每单元 1 个）的 Gπ−f 排除零 |
| D3 `addmm`，input 含 NaN，beta=0 | 检出 | **按登记规则未找到**：归为「合同范围外：参照不完整（complete_rate = 0.992）」。报告本身给出了特殊值判定 `contradiction`，3 个坐标上 f 有限而 K 为 NaN。见下文 D-A5 |
| D5 `F.rms_norm(eps=None)` | 检出，或在调用边界之前已分解 | **未找到**：`aten.rms_norm` 到达了分派边界，但没有已准入合同的 covers 包含 `aten.rms_norm` |
| D18 `interpolate` 非连续 nearest | 检出 | **未找到**：转第二步，INTERP 合同接口不是 ATen 调用 |
| D8 `avg_pool2d` ceil_mode 及其反向 | 拒绝，转第二步（任务选择） | **拒绝，但原因不同**：`avg_pool2d`、`avg_pool2d_backward`、`ones_like` 都没有已准入合同的 covers 包含它们，在「需要新增合同」一步就被拦下，没有走到任务选择 |

D3 的情况（偏差 D-A5，事后读数）：程序声明的计算在 NaN 坐标上本身就是 NaN，所以 Gπ 的参照不完整。登记的结论规则
（`run_e4.verdict`）只看 `complete_rate` 和 Gπ−f，不读同一报告里的 `task_special_value_verdict`，因此给出「未建立」。
测量报告确实记录了矛盾：f 在该坐标有限（MM.C3：beta=0 忽略 input，包括 NaN），K 为 NaN。按登记规则，本结果写「未找到」；
事后读数写「报告中有特殊值矛盾」，两者都列出，不改写登记结论。这同时是一个工具缺口：结论规则应当把特殊值矛盾算作检出。

W2 合计：5 个触发中，登记规则找到 1 个（D20）；D3 的证据在报告里但规则没有采用；D5、D18、D8 被合同覆盖或接口挡住。

## 4 被测实现的运行失败（D-A3，不计入上表分母）

这些 FlagGems 函数在真实负载上无法跑完一步，按工作负载额外排除后才记录：

| 工作负载 | 函数 | 失败 |
|---|---|---|
| gpt2 fp32、gpt2 bf16 | `layer_norm_backward` | 非法内存访问（1×128×768） |
| bert-tiny fp32 | `layer_norm_backward` | 只在带记录器的那一步发生非法访问，定位在 `weight_bias_backward_kernel`；与依赖布局的越界一致，未独立证明 |
| gpt2 bf16 | `layer_norm` | mean/rstd 以 bf16 返回，原生反向报「expected scalar type Float but found BFloat16」 |
| gpt2 bf16、bert-tiny bf16 | `flash_attention_forward` | philox 种子以 int64 返回，原生 SDPA flash 反向报「expected scalar type UInt64 but found Long」 |
| Qwen2.5-0.5B bf16 | `mm` | 需要 131072 B 共享内存，sm_86 上限 101376 B |

它们是真实负载上的运行失败，是否构成上游缺陷，需要另做独立复现（只起草，不提交）。

## 5 读法与限制

- **覆盖率的含义**：在 FlagGems 实际执行的调用键中，约 40%（按调用计约 58%）可以不经人工、直接得到语义结论。
  其余的阻断点都有名字：求值预算、缺少合同、合同 profile 的定义域、接口需要第二步。
- **检出的含义**：本次真实负载上的 8 个检出都低于输出精度；真正的缺陷（D20）在触发集上找到。单个训练步、固定输入，
  不能说明 FlagGems 在这些键上没有缺陷，只能说明在这批输入上没有看到超出常数舍入的差异。
- 排除了 3 个 FlagGems 函数（`copy_`、`to_copy`、`index`）以及第 4 节的运行失败函数，它们不在覆盖范围内。
- 同一键的各单元使用同一组记录值，统计规则多为「无法判断（退化）」，语义比较只在固定样本上进行。
- 生成计划的「已审阅」指的是生成规则 v1 已审阅，不是逐个计划审阅。
- 「合同范围外」中混有 5 个「参照不完整」的键（`gelu_backward` 4，D3 1），它们有唯一 profile，但参照没有算完整；
  协议 §7 的类别表没有单独列出这种情况，这里单独写明。
- 不覆盖 Inductor 融合 kernel（路径 B 尚未实施）；多节点组合依赖的 GRAPH 仍在审阅中。
