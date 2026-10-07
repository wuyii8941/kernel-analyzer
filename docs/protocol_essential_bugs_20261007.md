# 本质错误一轮：独立规格判据相对 eager 差分测试的必要性——执行协议（第一阶段，运行前提交）

日期：2026-10-07。本文在任何候选测量之前提交；之后的改动只追加到第 12 节（偏离记录）。

## 0. 依据与冻结

- **任务**：用户下发的「下一轮任务：从精度问题转向本质错误」（W0–W9、阶段闸门、交付物、验收标准）。
- **规格与主协议**：`specs/phase1/`，即审阅后的独立规格包 v0.4（`independent_specs_phase1_v0.4.tar.gz`，
  sha256 `0b494a62…bd0e`；包内 `SHA256SUMS` 12 个文件全部核对通过）。包内的 `protocol_phase1.md`（v0.4，含 2026-10-07
  审阅决定）是第一阶段的**主协议**；本文是执行方的补充，只写主协议留给执行方的细节（候选、条件、数值契约、分类流程、
  停止规则），与主协议冲突时以主协议为准。
- **执行方不改规格语义**（主协议第 3 节）：规格的疑似错误只上报，不在本轮修补；`specs/phase1/` 的文件在本轮内不改动。
- **检测器与统计层冻结**在标签 `eval-stage-a-20261006`：`git diff --stat eval-stage-a-20261006 HEAD -- src/` 为空（核对于
  2026-10-07）。本轮若改动 `src/`，作为偏离记录。仓库当前为 main@c7b5315（任务写的 9221d91 之后只有非精度一轮的脚本、
  文档与结果，`src/` 未变）。
- **规格自检在本机重跑**：`test_specs.py` 105/105 通过；`independent_check.py 1000` 的结果与包内记录逐项相同
  （CE 9,965 个标量全部在区间内；池化 1,815、index 960 个合法配置逐项相等；0 处不一致、0 处异常泄漏）。
- 任何地方都不把 eager 称为参照真值：eager 是一个实现，E 组比较的是两个实现之间的差。

## 1. 对任务文本的更正与差异（照录，不由执行方决定规格问题）

1. **B016 不是 eager 的错误。** 仓库记录（`bugs/B016_inductor_scatter_into_single_element.md`）与上游 #178871：Inductor 在下标
   可证明为常数时丢失原子加的掩码（以及把消费者融合到原子写之前），eager 正确。被关掉的检查是
   `test/inductor/test_torchinductor_opinfo.py`（v2.10.0）对 `index_reduce.amax/amin` 在 CUDA 上的 `check_gradient: False`——
   编译路径的梯度检查，不是 eager 的。所以 B016 是「编译缺陷 + 计数性质」的召回例子，不是共有错误的例子（与主协议第 7 节
   一致）。
2. **数值差异的统计。** 主协议第 5 节：每个条件 3 个 seed，用于冒烟与复现反例，不用于宣称总体平均作用。任务中「数值差异
   用平均作用统计，δ 取当前已标定的值」在第一阶段不执行：δ = 0.05 是在设置 R 中标定的更新层量，非精度一轮已经显示它不能
   跨设置搬用，更不能直接用于 kernel 输出。第一阶段只记录逐元素的数值差（与 FR 组的 e_num 区间）；是否在第二阶段做平均作用
   统计，在中期报告中提出，由审阅决定。
3. **embedding_bag**（任务 W1 第 3 项）在规格包中没有规格：第一阶段计为「规格未建立」，不测。
4. **规格引用的文档版本。** 规格文件开头引用的是 main（2.16.0a0）文档，契约应以锁定版本 2.10 的文档为准。第一阶段测量之前，
   执行方逐条比对规格所用条款（D1–D6、P-D1–P-D8、I-D1–I-D3、S-D1–S-D2）在 2.10 与 main 文档中的文字；若有不同，上报，
   受影响的条件暂停，不由执行方选择版本。比对结果写入结果目录。

## 2. 第一阶段范围（阶段闸门）

W0（本文）、W7（已知问题召回）、前三个算子的 W1–W4：交叉熵（含融合 linear + CE 与累加窗口归一化）、池化、index/scatter 类。
W5（测试判据审查）与其余算子在中期审阅通过之后进行。完成后提交中期报告并**暂停**。

## 3. 候选（W2）

| 候选 | 版本 | 适用 | FR（有 TTIR，K_R） |
|---|---|---|---|
| eager CPU float64 | torch 2.10.0（`ka_main`） | 全部 | 否（不是 Triton） |
| eager CPU float32 | 同上 | 全部 | 否 |
| eager CUDA float32 | 同上，RTX A6000（sm_86） | 全部 | 否 |
| eager CUDA bfloat16 | 同上 | CE、FLCE | 否 |
| Inductor CUDA float32（`torch.compile`，生成的 Triton） | 同上，Triton 3.6.0 | 全部 | 是 |
| Inductor CUDA bfloat16 | 同上 | CE | 是 |
| Liger `LigerCrossEntropyLoss` float32 / bfloat16 | liger-kernel 0.7.0（`liger` 环境：torch 2.10.0、Triton 3.6.0） | CE | 是 |
| Liger `LigerFusedLinearCrossEntropyLoss` float32 / bfloat16 | 同上 | FLCE | 是（Triton 部分；其中的 matmul 走 cuBLAS，按工具的规则标为混合） |
| Unsloth `fast_cross_entropy_loss` float32 / bfloat16 | unsloth 2026.3.3（`liger` 环境） | CE（只支持 ignore_index = −100、类别下标目标、无 weight 与 label_smoothing） | 是 |
| nightly eager CUDA float32、nightly Inductor float32 | `pt_nightly_cu126`：2.15.0.dev20260907+cu126（Triton 3.8.0）；若在首次测量之前装上更新的 nightly，改用它并记录版本 | 全部 | 否（TTIR 映射只对 Triton 3.6.0 成立，nightly 走黑箱模式，只算 K − f） |

不支持的组合（如 Unsloth 的 weight、Liger 的概率目标、avg_pool1d 的 divisor_override）计入「不支持」，单列，不计对错。
候选报错（含编译期崩溃）计入「报错」，单列。

## 4. 条件与输入（W6）

条件由 `scripts/essential/conditions.py` 生成，随本文提交；每个条件 3 个 seed（0、1、2），输入由（条件编号, seed）的
哈希播种，完全可复现。共 1,724 个条件 × 3 seed：

| 家族 | 条件数 | 内容 |
|---|---|---|
| CE | 274 | 主网格：(N, C) ∈ {(4,5), (7,1000), (3,4099)} × 忽略模式 {无, 部分, 全部, 首尾} × reduction {none, sum, mean} × weight {无, 正整数, 含零且零权重类别出现在目标中} × label_smoothing {0, 0.1}（216）；ignore_index = 0 落在类别范围内（6）；ignore_index = −1（1）；logits 极端值 {×10⁴, 单个 10³⁰, 整行相等}（12）；非连续 logits（转置存储，3）；大小为 1 的维度 (1,1)、(1,5)（12）；概率目标（24） |
| FLCE | 72 | (N, V) ∈ {(4,1000), (3,4099)}，H = 64 × 忽略 {无, 部分, 全部} × reduction × label_smoothing × weight {无, 正整数} |
| 池化 | 776 | avg 1d/2d/3d（384：kernel、stride、padding、ceil_mode、count_include_pad、divisor_override 的组合，含 POOL-A1 的过界窗口，值为小整数或高斯）；max 1d/2d/3d（392：含 dilation、ceil_mode、padding，值为易并列的小整数、高斯或含 NaN/±∞；含一个只采到 padding 的合法窗口） |
| index/scatter | 602 | index_add（56）；index_reduce {prod, mean, amax, amin} × include_self（336）；scatter_reduce {sum, prod, mean, amax, amin} × include_self（210）。形状 {1-D 长 5、1-D 长 1（单元素目标）、2-D dim 0、2-D dim 1}；重复下标 {部分重复, 全部指向同一位置}；值 {小整数, 含零（prod）/易并列（amax/amin）, 高斯} |

累加窗口归一化不按单个算子测，放在 W7 的 HF 召回与配套的构造实验里（第 8 节）。

输入先以 float64 生成，再舍入到候选的 dtype；**规格在候选实际收到的值上求值**（bf16 候选用舍入后的 logits；
label_smoothing 等标量用实际收到的值：ATen 收到 float64，Triton kernel 的标量参数以捕获记录为准），接口舍入单列（CE-A6）。

## 5. 四组比较（W3、W4）

同一输入、同一规格、同一裁决口径，四组同时运行：

- **E**：K − K_eager。K_eager 取与候选**同设备、同 dtype** 的 eager 实现（库实现的测试通常这样比较）；eager 候选本身另与
  eager CPU float64 比较。阈值与 F 相同（第 6 节）。
- **F**：K − f，f 为主解释；同时对每个有文档依据的读法与带标签的错误变体求值，用于分类（主协议第 6 节）。
- **P**（不依赖任何参照实现，对包括 eager 在内的所有候选执行）：
  - CE：平移不变、梯度行和为零、被忽略行梯度为零、类别置换（同步变换 weight、目标与范围内的 ignore_index）；
  - 池化：avg 的线性与伴随、max 的平移、梯度总量（只对非空窗口）、返回下标属于并列集合、不重叠窗口上的集合式次梯度检查；
  - index/scatter：计数、未被写的位置不变、顺序不变、极值的集合式次梯度检查；
  - **反向一致性** ⟨B(x, v), u⟩ = ⟨v, J(x) u⟩：B 是候选的反向（autograd 的 VJP），J(x) u 在规格上精确计算，不用有限差分——
    CE 用规格给出的精确梯度（逐行分块）；avg pool、index_add、scatter/index 的 sum 与 mean 关于输入是线性的，J u 就是规格
    作用在 u 上；prod 用规格的 `prod_grad_factors`；max pool、amax/amin 只在唯一极值处计算（并列处用集合式检查）。
    这是**规格导数核验**。若候选支持前向模式（`torch.func.jvp`），另做**实现的前后向自一致性**，两者分开报告（主协议 6.8）。
- **FR**：有 TTIR 的候选（第 3 节）走 `kernel_analyzer.check` 的模式 B，f 由规格给出的包围区间提供：逐元素 K_R、
  e_num = K − K_R、e_sem = K_R − f。与 F 共用同一个 K − f 契约裁决「候选是否违反任务」；FR 额外报告 e_sem 的区间是否含零
  （与精度无关的语义判定，bf16 候选也适用）、K_R 与各读法 f_r 是否相容，这些单独计分，不进入与 F 共用的检出率。
  FR 的额外目标所用的 e_sem 区间判定与 K 的精度无关，所以低精度候选在 FR 中可以有语义结论，在 F 中没有。

## 6. 数值契约与裁决（W3）

- **float64 候选**：筛查门槛 τ₆₄ = 10⁻⁹·(1 + |f|)（主协议 6.3）。
- **float32 候选**：筛查门槛 τ₃₂ = 2⁻¹²·(1 + |f|)（约 2.4·10⁻⁴，容纳 C ≤ 4099 的归约在 float32 中的舍入）。
- **bfloat16 候选**：F 组只记录数值差，不判语义（主协议 6.3）；FR 组的 e_sem 照常判。
- **精确可表示的输入**（小整数的 index_add / scatter sum / prod，整数输入且除数为 1 或 2 的幂的池化）：中间结果在候选 dtype
  中精确，要求 K = f 严格相等（主协议 6.2）。
- 越过门槛记为「待确认警报」，按以下顺序裁决：(1) 若 K 在全部元素上与某个有依据的读法或带标签的错误变体相容（同一门槛），
  而与主解释不相容，记为该读法 / 变体；(2) 否则保存反例，用不依赖工具代码的独立复算（纯 Python 精确有理数或 mpmath）确认
  后记为「偏离规格」；(3) 规格返回未建立或拒绝输入，计入「规格未建立 / 输入不合法」，不计任何候选的对错。
- 区间含零写「未排除相等」，不写「符合规格」；有限样本无反例不等于全域正确。
- **一个候选在全部输入上必须符合同一读法**：读法按候选汇总；同一候选在不同输入上落在不同读法，本身记为发现。

## 7. 分类与计数

每个（算子, 条件, 候选）归入主协议 6.6 的类别：与规格相容（记录相容的读法）；偏离规格且与 eager 一致（共有错误）；候选
偏离、eager 相容（候选错误）；候选相容、eager 偏离；两者各偏离各的；读法分歧。另列：不支持、报错、规格未建立 / 输入不合法、
无定义情形的实现约定（CE-A3、ACC-A1、POOL-A5 的 NaN 变体）。

结果分三栏（主协议第 6 节）：文档明确条款的违反；声明解释下的差异；未定义情形的实现约定。所有计数附分母
（算子 × 条件 × 候选；前向与反向分开）。

## 8. 已知问题召回（W7，先做）

记录 E、F、P、FR 各自能否发现，不预设 eager 差分必然失败。

1. **B016**（锁定 2.10，CUDA）：`zeros(1).scatter_add(0, zeros(N), f(x))` 一类写法与单元素 `scatter_reduce(mean)` 的反向；
   E（compile 对 eager）、F（规格）、P（计数性质）、FR（K_R 来自编译后 kernel 的 TTIR）。
2. **B014 的 avg_pool 部分**：1 元素序列参数的 `avg_pool2d/3d` 反向在 Inductor 下编译期崩溃；记录它落在 POOL-A1 的哪种读法上，
   或属于规格之外的错误（接口 / 编译崩溃），以及各组能否发现。
3. **HF transformers 梯度累加的 loss 归一化**：先做构造实验（主协议第 7 节：token 数 1 与 3、梯度和 −1/2 与 3/2，两份「均值的
   均值」实现互相比差为 0、与规格比差为 −1/4），构造实验不冒充旧版本实测；再在修复前与修复后的版本上实测（版本号从
   transformers 的发布说明与修复 PR 查实并记录）：随机初始化的小 Llama、可变长序列与 −100 padding，Trainer 的
   `gradient_accumulation_steps` > 1，记录累加后的梯度，与 `spec_accumulation.window_grad`（逐 token 梯度由 float64 autograd
   给出，作为规格的输入）比较。E 组的对应物是「累加对不累加」的差分；P 组是 `prop_split_invariance`（重新划分微批不应改变
   窗口梯度）。修复前后的版本另装在独立环境中（`/data1/tzh/envs/`），不改动现有环境。

## 9. 与轻量基线对照、成本（W8）

对每个检出的偏离，记录：「eager 参照 + 相同裁决」能否检出；「FP64 参照（eager CPU float64）+ 相同裁决」能否检出。成本：
规格行数（规格包）、执行方接入代码行数、书写与接入时间、运行时间，按组记录。

## 10. 上游核对（W9）

任何新错误在写报告之前按四项检索：症状关键词搜 issue、相关测试文件、近期 PR、nightly 复现；已有报告的撤回，只作召回案例
保留。只准备草稿，不提交。每条错误记录附最小复现、不使用工具代码的独立复算、nightly 结果与检索记录。

## 11. 成功标准与停止规则

- 不预先决定哪种结果算成功（主协议第 8 节）：没有找到共有错误，只说明所测范围内没有找到；找到共有错误但 F 与 FR 效果相同，
  说明检出收益来自独立规格，工具的增量另看自动化与诊断成本。
- 停止：第一阶段完成第 2 节的范围后提交中期报告并暂停。发现疑似规格错误（独立复算与规格不符，或与锁定版本文档矛盾）时，
  受影响的切片暂停并上报，不修补规格。候选报错不停止，记录后继续。
- 需要上报、不自行决定：读法分歧中需要人判断的部分；任何上游提交；超出本协议的范围扩展。

## 12. 偏离记录

（运行后追加。）
