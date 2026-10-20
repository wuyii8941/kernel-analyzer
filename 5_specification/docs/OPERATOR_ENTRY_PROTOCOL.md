# 第一步路径 A：标准算子入口的自动绑定与测量（登记方案，2026-10-20）

背景与口径见 `REFERENCE_ENTRIES.md`。本方案在任何工作负载测量之前提交。工具链是在不属于工作负载的合成调用上开发的：
`torch.exp`、`torch.addmm`、转置后的 `argmax`、`sum`，产物在 `.cache/operator_entry_proto/`，不计入结果。
与充分性评价一样，**不设阈值，只报告**。所有尝试都在分母中。

## 0 要回答的问题

对于在真实运行中经过 ATen 分派边界、并由第三方 Triton 实现执行的算子调用，下面三件事各能做到多少：

1. 不写人工绑定，能否自动取得独立于实现的 f；
2. 能否得到完整参照；
3. 能否给出语义结论。

对象是 FlagGems 5.0.2。它用 `torch.library` 在 CUDA 分派键上替换 ATen 算子，位置低于 Python 分派 mode，
所以记录器看到的是调用边界，调用本身仍然分派到 FlagGems。

## 1 冻结对象

登记提交（本文件所在提交，下称 F_A）时的以下内容，测量期间都不改：

- **规格**：合同、适配器、`kernel_spec`。
- **测量工具**：`2_tool`，含 `measure.py`。
- **生成规则 v1**：`sufficiency/operator_entry/instantiate.py`。
- **记录器**：`sufficiency/operator_entry/record.py`。
- **工作负载脚本与本方案。**

外部固定：
- FlagGems 5.0.2，来自 `.cache/pydeps/flag_gems_5_0_2`，即 wheel `flag_gems-5.0.2-py3-none-any.whl`；
- 环境 `/data1/tzh/envs/hf_ga_5190`：torch 2.10.0+cu128、Triton 3.6.0、transformers 5.19.0；
- GPU：RTX A6000。

## 2 被测配置：排除三个函数

被测配置是 `flag_gems.enable(unused=["copy_", "to_copy", "index"])`，测量时的调用是
`use_gems(exclude=[同上])`。排除的是 FlagGems 中实现 `aten.copy_`、`aten._to_copy`、`aten.index.Tensor` 的函数。

原因：测量工具捕获时会在 Triton 启动钩子**内部**调用这三个 ATen 算子，用来复制存储、搬回主机、按窗口索引。
注册它们会导致两个问题：

- **重入**：FlagGems 的启动锁不可重入，开发时实际出现了死锁；
- **污染**：捕获自身的复制会混进被测调用的启动记录。

这正是 `REFERENCE_ENTRIES.md` 第 4 节说的「记录或测量手段改变了被测对象」。所以记录运行与测量运行使用**同一个排除集合**，
两次的执行对象一致。代价是这三个函数的 FlagGems 实现不在本次评价范围内，按限制报告。

## 3 生成规则 v1（自动绑定）

对一个记录到的调用，加上一个候选的（合同，profile），`instantiate.generate` 在临时规格树中写出五样东西：

- 调用文件；
- 输入生成器；
- 绑定计划；
- 保留副本：评价用的手段，只存在于临时树中；
- 测量声明。

具体规则：

- **输入**：
  - `/op` 取 `aten.<基本名>`；
  - 每个位置参数、每个**非默认**关键字参数，都按合同的 ATen 调用格式给出；
  - 张量取记录的调用前状态（`tensor_record`），保持尺寸、stride 和数值；
  - 非张量按记录值作为字面量，依据写作「recorded argument value」。
- **输出**：唯一返回的张量，绑定到 profile 唯一的输出。
- **状态**：每个张量参数声明为不变（unchanged）。
- **拒绝而不猜测**，按阶段记录：
  - 可变或 out 的 overload，记为「状态映射」；
  - 返回值不是单个张量，或 profile 的输出不止一个，记为「输出映射」；
  - 合同格式表达不了的参数（device、layout、复数等），或张量和非张量混合的容器，记为「参数编码」。
- **测量声明**：
  - 输入**全部是记录到的真实值**，每个单元相同；
  - 开发 1 个单元，确认 2 个单元，最多 3 个单元；
  - 每个案例 600 秒；
  - 参照类别只允许 `complete_composed`。

这里的「已审阅」是指生成规则本身随本方案提交、经过审阅，不是逐个计划人工审阅。计划写入
`"generated_by": "operator_entry.instantiate v1"`。

## 4 profile 选择（取代「取第一个不需要任务选择的 profile」）

候选 = 所有 `implementation_status: validated` 且 covers 含 `aten.<基本名>` 的合同中，`task_choice_required: false` 的每个 profile。

- 每个候选都生成并测量一次。某个 profile **适用**，指它的报告中有状态为 `evaluated` 的输出。
- 测量工具只报「sampled call did not retain an admitted specification binding」、没有给出原因时，用同一份绑定计划把
  记录的输入映射后，直接调用 `registry.query`（预算 300 秒）取得参照结果：
  - `unsupported` 或 `invalid_input`：该 profile **不适用**；
  - 其他结果：仍按测量工具的失败处理，记为**未决定**。
- 其他测量失败（绑定拒绝、语义缺失、超预算等）：记为**未决定**。
- 恰好一个适用：采用它。
- 两个或以上适用：拒绝，记「多个 profile 适用」，转第二步。
- 一个适用，但另有未决定的候选：唯一性不成立，记「运行或证据不足」。
- 没有适用的：
  - 如果该算子在某个合同里有需要任务选择的 profile，记「需要任务选择」，转第二步；
  - 否则记「合同范围外」，附原因。
- 候选因超预算而没有结论时，记「运行或证据不足」，不当作不适用。
- 合同的接口不是 ATen 调用格式（NORM、EMB、POOL 别名）：记「需要规格匹配」，转第二步。

## 5 两项执行对象核查

- **T1 记录器是否改变执行**：每个模型配置的步骤如下：
  1. 先热身一步；
  2. 不带记录器跑一步，得到 Triton 启动序列 A；
  3. 带记录器跑一步，得到序列 B，剔除记录器自身复制产生的启动；
  4. 比较 A 与 B 的 kernel 名序列。

  两次之前都 `zero_grad(set_to_none=True)`。不一致的模型配置，其全部调用记为「运行或证据不足：记录改变了执行对象」。
- **T2 重放是否执行同一对象**：生成调用文件后，在同一环境中执行一次，比较其中的 Triton kernel 名序列与记录到的序列。
  不一致的，记「运行或证据不足：重放的执行对象不同」，不测量。

## 6 工作负载

- **W1 模型训练一步**：
  - 模型：`prajjwal1/bert-tiny`、`openai-community/gpt2`、`Qwen/Qwen2.5-0.5B-Instruct`；
  - 配置：fp32、bf16，共 6 个模型配置；
  - 设置：截为 2 层，128 个 token（wikitext-103 验证集，与 E1 相同），`train()` 模式，eager，前向加 loss 加反向；
  - 加载方式与 `corpus/prov_model.py` 相同，但不编译。
- **W2 已知缺陷触发**（S1 回归，开发批次记录于 `corpus/external_batch/README.md`，都在 FlagGems 5.0.2 上发现）。
  每个用公开函数加触发输入调用一次，由记录器取得 ATen 调用：

  | ID | 触发 | 期望 |
  |---|---|---|
  | D3 | `torch.addmm(input 含 NaN, A, B, beta=0)` | 检出（MATMUL MM.C3：beta=0 忽略 input，包括 NaN） |
  | D5 | `F.rms_norm(x, shape, weight, eps=None)` | 检出（NORM：eps=None 取 dtype 的 eps），或在调用边界之前已分解 |
  | D18 | `F.interpolate(非连续 x, scale_factor=2, mode="nearest")` | 检出 |
  | D20 | `torch.argmax(非连续 x)`，dim=None | 检出 |
  | D8 | `F.avg_pool2d(x, 3, 2, ceil_mode=True)` 及其反向 | 拒绝：POOL 的 avg profile 需要除数读法的任务选择，转第二步 |

  这里期望是事先写下的，结果按「找到 / 未找到（原因）/ 无法运行」照实报告。

脚本里另有一个 `SMOKE` 触发，复用开发用的合成调用，只用于检查工具链，不是登记的工作负载，不计入结果。

## 7 计数单位与报告

- **调用键**：每个模型配置内的（overload，参数结构）：张量按 dtype、尺寸、stride 区分，其他参数按值区分。同时报告调用次数。
- **候选集合**：调用期间至少有一次 Triton 启动的键，即确实由 FlagGems 执行。没有启动的键单独报告，包括视图、元数据和原生回退。
- **单个调用键超过 200 MB** 时不保存数值，记「运行或证据不足：捕获大小上限」。

每个候选键按第一个阻断点归入一类：

| 类别 | 含义 |
|---|---|
| 运行或证据不足 | T1、T2 不一致、捕获上限、超预算、执行错误 |
| 需要新增合同 | 没有已准入的合同覆盖该算子 |
| 需要规格匹配（第二步） | 合同接口不是 ATen 调用、多个 profile 适用、需要任务选择 |
| 需要扩展生成规则 | 状态映射、输出映射、参数编码 |
| 合同范围外 | profile 都不适用，且没有任务选择 profile |
| **原样可用** | 唯一 profile，参照完整，给出语义结论 |

原样可用的再按 E4 规则给出结论：检出（某个坐标的 Gπ−f 排除零）、相容（全部完整且没有坐标排除零）、未建立。

报告内容：
- 各类的键数与调用加权数，附 Wilson 95% 区间；
- 按 overload 族、按模型配置分组；
- W2 逐例结果。

## 8 不主张的事

- 不覆盖 Inductor 融合 kernel（路径 B）。
- 不覆盖被排除的三个 FlagGems 函数。
- 数值来自单个训练步：单元相同，不做统计推断，只做固定样本上的语义比较。
- 「已审阅」指的是生成规则，不是逐个计划。

## 9 偏差记录

（提交后追加，不改上文。）
- 登记提交 F_A = c09b655。
- 2026-10-20 记录阶段的结果：5 个触发都成功，6 个模型配置都失败。按原因分为两类：
  - **工具链问题**：
    - bert-tiny 两个配置：transformers 5.19 不接受没有 `model_type` 的 config.json；
    - Qwen2.5-0.5B fp32：保存时有一个参数含不可序列化的 module 对象，整个记录目录没有写完。
  - **被测实现运行失败**：都在热身步发生，此时记录器尚未启用；用 `CUDA_LAUNCH_BLOCKING=1` 定位到具体算子：
    - gpt2 fp32：FlagGems `layer_norm_backward`（`aten.native_layer_norm_backward`，1×128×768 fp32）非法内存访问；
    - gpt2 bf16：FlagGems 的 `_flash_attention_forward` 把 philox 种子返回为 int64，PyTorch 原生的
      `_scaled_dot_product_flash_attention_backward` 期望 uint64，反向报「expected scalar type UInt64 but found Long」；
    - Qwen2.5-0.5B bf16：FlagGems `mm`（128×151936 @ 151936×896 bf16）的配置需要 128 KB 共享内存，A6000（sm_86）上限是 99 KB。
- 偏差 D-A1（工具链）：config.json 没有 `model_type` 时，用 `AutoConfig.for_model("bert", ...)` 读取。bert-tiny 是 BERT 结构。
- 偏差 D-A2（工具链）：单个调用键的参数无法序列化时，不保存该键的数值，键仍在分母中，记「运行或证据不足：记录的参数不可序列化」。
  失败的部分记录移入 `failed_harness_20261020/`。
- 偏差 D-A3（被测配置）：对每个工作负载，把运行失败的 FlagGems 函数逐一定位。每轮在新进程里重跑、加入新排除，直到整步能跑通。
  然后只为该工作负载**额外排除**这些函数，重新记录，使其余调用能被评价。具体做法：
  - 额外排除的清单固定在 `sufficiency/operator_entry/run_failures_20261020.json`；
  - 记录与重放使用同一排除集合（`run.json` 的 `exclude`）；
  - 运行失败本身作为结果单独报告：哪个函数、什么输入、什么错误。它们是被测实现在真实负载上的运行失败；是否构成缺陷，另行用独立复现确认；
  - 这些调用不计入「原样可用」的分母，单独列为「被测实现运行失败」。
