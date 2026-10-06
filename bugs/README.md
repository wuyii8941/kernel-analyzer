# 问题登记

按工作框架分类：① 节点计算误差（舍入、次序、近似指令、TF32 等）、② 输入来源不一致（保存值、重算、读错数据）、
④ 节点算法与规格不符（公式、常数、变体）。状态写清是否已有上游记录；"首次发现"写实际最早检出它的手段，
"工具检出"指用我们的工具（TTIR 自动参照 K_R、区间残差、`assess_units` + 检测器 2.1）跑出的结论。

对规格检查的入口：`scripts/tool_spec_check.py`（e_num = K − K_R，e_sem = K_R − f，用例在
`scripts/tool_spec_cases_*.py`）。结果在 `results/tool_spec/`。

| 编号 | 问题 | 组件 | 类 | 量级 | 上游状态 | 首次发现 | 工具检出 |
|---|---|---|---|---|---|---|---|
| [B001](B001_liger_gpt_oss_rope_regression.md) | main 上 #1451 把 GPT-OSS 的 RoPE 当成部分旋转 | Liger main（未发布） | ④ | q 相对误差 0.97 | 未见报告；Liger 自己的 mini_gpt_oss 测试在 main 上失败 | 普查脚本 | 是：e_sem 检出，相对 RMS 1.24；e_num 未检出；0.8.4 上 e_sem 恒为 0 |
| [B002](B002_liger_phi4mini_partial_rope.md) | 发布版对 phi3 + partial_rotary_factor（Phi-4-mini）整头旋转 | Liger ≤ 0.8.4 | ④ | Phi-4-mini 验证损失 2.70 → 12.93 | main 被 #1451 顺带修好，PR 未提及，未发补丁版本 | 普查脚本 | 是：e_sem 检出，相对 RMS 1.18；main 上 e_sem 恒为 0 |
| B003 | 融合 MoE kernel 的 `tl.dot` 用 Triton 默认 TF32，不受 `allow_tf32=False` 控制 | Liger 0.8.x `LigerExperts` | ① | fp32 下专家梯度相对 2.4e-3 | 未查 | 普查脚本；`TRITON_F32_DEFAULT=ieee` 后消失 | 未跑 |
| B004 | Qwen3-Next 实例补丁的 RMSNorm 用 offset=0（规格 x̂·(1+w)） | Liger 0.7.0 | ④ | 输出整体错误 | 0.8.x 已修 | 普查脚本 | 未跑（已修，作召回） |
| B005 | 纯 bf16 AdamW 的二阶矩只增不减，有效步长逐坐标缩小 | PyTorch `torch.optim.AdamW`（bf16 参数） | ①（状态） | 见 `docs/bf16_adam_beta2_prediction.md` | 机制已发表（Zamirai 2020、Collage 2024），作已知阳性召回；用户计划提 PyTorch issue | 探针脚本（`probe_bf16_adam_state.py`） | 是：`run_detection` 在 torch.compile 后的 AdamW Triton kernel 上自动检出，只在 exp_avg_sq 的写入上（commit 9d44d70） |
| B006 | `tl.dot` 对 fp32 输入默认 TF32，截断造成幅度收缩 −7.0e-4 | Triton | ① | 相对 −7.0e-4 | Triton 文档写明的行为 | 探针脚本（`scripts/probe_tf32_dot.py`） | 未单独跑 |
| B007 | bf16 乘积在张量核上的 FP32 累加随 K 收缩（−1.2e-9·K） | Triton `tl.dot`（cuBLAS 约 1e-7） | ① | 极小 | 未报告 | 探针脚本（`scripts/probe_tensor_core_accumulation.py`） | 未单独跑 |
| B008 | AdamW4bit 的两个矩都缩小步长，差距主要等价于学习率 ×0.7 | torchao 0.16 | ①（状态） | 训练损失差约等于 LR 降 30% | 量化设计取舍 | 训练对照（`lowbit_adam_*.py`） | 未跑 |
| B009 | `_AdamW(bf16_stochastic_round=True)` 只对参数做随机舍入，状态仍是 bf16 就近舍入，二阶矩同 B005 漂移 | torchao 0.16 | ①（状态） | v 比值 1.33（3k 步） | 未查 | 探针脚本 | 未跑 |
| B010 | `torch.compile` 下 `avg_pool2d(ceil_mode=True, count_include_pad=True)` 的反向（上游 issue 也涵盖 1d；我们的 1d 用例窗口未越界，未触发）用整个核的大小作除数，ceil_mode 多出的越界窗口梯度错 | PyTorch 2.10 Inductor | ④ | 梯度相对误差 9.6%（最后一列）；编译后 gradcheck 失败，eager 通过；前向逐位相同 | 已知未修：pytorch/pytorch#198119（2026-09-22） | 工具（Inductor 对 eager 语义筛查） | 是：e_sem 检出，相对 RMS 9.4e-2，前向 e_sem 7.6e-17 |
| B011 | Triton 官方教程 06-fused-attention 的反向不支持非因果：`causal=False` 时不报错，dq/dk/dv 与真实梯度相差约 100%（前向正确） | Triton v3.6.0 `python/tutorials/06-fused-attention.py` | ④ | e_sem 相对 RMS ≈ 1.0（dq 0.998、dk 0.992、dv 0.980）；e_num 3e-4 | 已知：教程测试只跑 `causal=[True]`，注释 `# FIXME: Non-causal tests do not pass at the moment.`；包装函数没有拦截 | 工具（教程 kernel 筛查） | 是 |
| [B012](B012_torch_radam_fp32_rectification.md) | `torch.optim.RAdam` 的 capturable 路径与 `torch.compile(opt.step)` 在 fp32 里算整流项 ρ_t（两个接近 ρ_∞ 的数相减，β₂ 先被舍入到 fp32），β₂ 接近 1 时整流判据 ρ_t > 5 在前几步翻转 | PyTorch 2.10 与 main | ④ | 更新相对误差：β₂=0.999 达 6e-3；0.9995 第 5 步 0.99；0.9999 第 2/4/5 步 0.99、第 10 步 0.24；0.99999 达 18 倍（eager 非 capturable 为 2e-4–2e-3） | 检索未见报告（2026-10-06 经 GitHub API 复查）；CUDA nightly 2.15.0.dev20260907 实测仍在 | 工具筛查（`opt_radam` 报出"小"档 e_sem），随后定向用例 | 是：β₂=0.9995 第 5 步 param 的 e_sem 1.45e-2（≈ 整步更新），e_num 2.4e-7 |
| [B013](B013_vllm_unified_attention_bidir_swa.md) | vLLM TRITON_ATTN `unified_attention` 的 V 掩码按 q 块第一行算窗口右边界，非因果滑窗时块内后面的行丢掉窗口右侧的 key | vLLM main 与 v0.24.0–v0.31.0（#45163 引入） | ④ | 输出相对误差 0.08–0.90（PR #51257 自带的测试配置下 0.15，bf16 噪声 2e-3）；现有模型的默认配置双向段短于窗口，不触发（潜在） | 检索未见报告；#51257 新增的测试会暴露它，但该 PR 认为 main 已正确；相关待核对：#47217（Gemma4 图像双向注意力限制在滑窗内，2026-06-30 合入）、#53147（多模态前缀的滑窗分块裁剪） | 读代码后用工具用例确认 | 是：三个用例 e_sem 相对 RMS 0.905 / 0.115 / 0.216，区间证实（区间不含 0 的坐标 26–87%），e_num 仅 fp16 舍入；对照用例 e_sem 为 0 |
| [B014](B014_inductor_pool_backward_single_element_args.md) | `torch.compile` 下 avg_pool2d / avg_pool3d / max_pool2d 的反向 lowering不规整 1 元素的 kernel_size / stride / padding / dilation（前向用 `pad_listlike` 规整），直接断言失败 | PyTorch 2.10 Inductor 与 main | 编译崩溃 | 训练编译失败（不会给出错误结果）；只影响 1 元素序列写法，如 `nn.AvgPool2d(3, padding=(1,))` | 检索未见报告（2026-10-06 经 GitHub API 复查）；nightly 2.15.0.dev20260907 上 max_pool2d 部分已通过，avg_pool2d / avg_pool3d 仍失败 | OpInfo × Inductor 筛查（PyTorch CI 对这些算子只跑第一个样例），随后静态扫描 lowering 找到同类两处 | 工具运行时暴露（编译错误，不是数值检出） |
| [B015](B015_inductor_avg_pool3d_ceil_backward.md) | Inductor `avg_pool3d` 反向在 `ceil_mode=True`（count_include_pad 默认）时把越界窗口按整个 kernel 体积平均 | PyTorch 2.10 Inductor 与 main，CUDA 与 CPU | ④ | 梯度相对误差 0.16–0.38；前向正确 | pytorch#198119 只覆盖 1d/2d，并明确认为 3d 正确（只在窗口数 > 125 走 fallback 时成立）；未见修复 PR；nightly 2.15.0.dev20260907 实测 CUDA 与 CPU 仍在 | 工具：OpInfo × Inductor 筛查（PyTorch CI 对 avg_pool3d 只跑第一个样例） | 是：`oib_nn_functional_avg_pool3d_6` e_sem 相对 RMS 0.306，20% 坐标区间证实，R2 检出，e_num 8e-8 |
| [B016](B016_inductor_scatter_into_single_element.md) | Inductor 把 scatter_add / index_add / index_put(accumulate) 写进只有一个元素的目标时：原子写丢了掩码（填充通道也累加），读取者被融合到原子写之前 | PyTorch 2.10 Inductor（CUDA 错误结果，CPU 部分写法编译崩溃）；main 相关代码未变 | ④ | N 个贡献算成 ⌈N/XBLOCK⌉·XBLOCK 个（5→8、30→32、5000→5120）；PyG `scatter(reduce='mean')` / `global_mean_pool` 单图时结果 × N/⌈N⌉；单元素 scatter_reduce 反向 mean 系数错、amax/amin 得 inf/NaN | **已知、main 已修**（更正 2026-10-06）：pytorch#178871（2026-03-31），修复 #179833（2026-04-16，2.10 之后）；nightly 2.15.0.dev20260907 上两处机制的例子都正确；此前「未见报告」系检索遗漏，草稿不提交 | 工具：OpInfo × Inductor 筛查（scatter_reduce 0 维样例；PyTorch CI 只跑第一个样例），随后读反向图与生成代码定位 | 是：`oib_scatter_reduce_mean_19` e_sem 0.5（证实）、amax/amin 特殊值类别不一致；专门用例 `sc1_mean_pool_one_graph` e_sem 0.067、100% 坐标证实，e_num 1e-7 |
| [B017](B017_dynamo_shared_code_in_graph_functions.md) | Dynamo 对图内的 Python 函数只按 `__code__` 守卫（`CLOSURE_MATCH`），boolean_dispatch 生成的 `F.max_pool*` / `adaptive_max_pool*` / `fractional_max_pool*`、`torch.unique`/`unique_consecutive`/`lu` 共享代码对象，作为参数、闭包、模块属性传入时第二个函数复用第一个函数的图 | PyTorch 2.10 Dynamo 与 main（nightly 2.15.0.dev20261005 实测仍在；与后端无关） | ④ | 静默错误：`adaptive_max_pool1d(x, 3)` 得到 `max_pool1d(x, 3)` 的数值（形状相同）；双分支模型区域编译后输出错误，整体编译正确 | 检索未见报告；相关的 pytorch#197811（方法描述符，开着）及其 PR #197845 不覆盖（模拟该 PR 后 pooling 仍错） | 工具：OpInfo 前向筛查中 `__rdiv__`/`__rmul__` 的 e_sem 1.38/4.13（同进程用例复用了 `__radd__` 的图），随后读 Dynamo 守卫代码、枚举共享代码对象的图内函数 | 报警来自工具（e_sem 区间证实、kernel 名与算子不符）；定位靠读源码；直接差分在同进程下同样能发现 |
| [B018](B018_inductor_fuse_attention_rewrite.md) | Inductor 把手写注意力改写成 SDPA 时改变语义：A. 2.10 中模式常数是通配符（除以任意常数、任意 `masked_fill` 值都被改写成默认缩放 / -inf 屏蔽）；B. DistilBERT 模式（15/17/20）在 `attn_mask == 0` 处屏蔽，替换却保留 `attn_mask == 1`，非 0/1 掩码与全掩码行结果不同 | PyTorch 2.10 Inductor（A：main 已修，nightly 实测；B：nightly 2.15.0.dev20261005 实测仍在）| ④ | 静默错误：缩放 √d_model 代替 √head_dim 时差 1.19、温度 4.0 差 1.7；段 id 掩码差 1.83；HF 式 `finfo.min` 填充遇整条填充序列差 0.70 | 检索未见 B 的报告（#195320 / #100452 / PR #195907 / #198509 为相邻问题，2026-10-06 经 GitHub API 复查） | 读 `fuse_attention.py` 的匹配与替换后用直接差分探针确认（计划第 6 节的假设） | 否：改写后落在 cuDNN / Flash kernel 上，不是 Triton |
| [B019](B019_averagedmodel_bf16_ema_stagnation.md) | `torch.optim.swa_utils.AveragedModel` 对纯 bf16 模型做 EMA：平均副本继承 bf16，`get_ema_multi_avg_fn` 的增量小于半个 bf16 ulp 被舍掉，EMA 冻结在开始平均时的权重附近；fp32 的平均副本被`update_parameters` 拒绝（按 dtype 分组报错） | PyTorch 2.10 与 nightly 2.15.0.dev20260907 | ①（状态） | 小模型纯 bf16 训练 2,000 步：平均模型验证 loss 9.0 对 fp32 EMA 5.16（种子 0–3，约 270σ）；修正（fp32 副本 + fp32 `avg_fn`）与参照逐位相同 | 检索未见报告（2026-10-07，GitHub API）；文档无 dtype 说明；timm / diffusers 的 EMA 同样继承模型 dtype；草稿待用户决定是否提交 | 工具：阶段 C 影子状态单步测量（bf16 EMA 的单步作用为 δ 的 11.8 倍），随后预注册复现 | 是（更新层单步作用） |

## 观察（不算缺陷，记录在案）

| 编号 | 观察 | 组件 | 严重度 | 证据 |
|---|---|---|---|---|
| O01 | 反向 kernel 的正确性依赖"不带 `other` 的掩码读取在越界通道上为 0"。TTIR/Triton 文档里这个值是未定义的；Triton 3.6 的 NVIDIA 后端会先把目标寄存器置 0（PTX 已核实），所以结果正确。若越界通道是 NaN/Inf，整行梯度会被污染（0·NaN）。一行修法：读取时给 `other=0.0` | flash-attn `layer_norm.py` 反向（`w = tl.load(W + cols, mask=mask)` 之后 `wdy = w * dy` 进入整行求和 c1）、mamba_ssm `layer_norm.py` 反向（同一份代码）、mamba_ssm `layernorm_gated.py` 反向、PyTorch FlexAttention 反向在 Q_LEN 非块倍数时的 dv（越界行的 LSE） | 低（可移植性） | 工具判为参照未建立（`undefined:masked load without other`）；`ptx_zero_fills` 对这些 kernel 的 PTX 全部为真；工具加了经 PTX 核实才启用的"掩码通道补 0"假设后可评估 |
| O02 | FlexAttention 用 fp32 舍入后的 log2(e) 做 exp2，softmax 温度偏 1.3e-8 | PyTorch FlexAttention | 可忽略 | 规格温度乘 (1 − 1.33e-8) 后 e_sem 从 1.76e-8 降到 8.5e-16 |
| O03 | head_dim 96、以及带可学习 bias 的 score_mod，在 A6000（共享内存 101 KB）上编译失败（No valid triton configs / OutOfResources） | PyTorch FlexAttention | 可用性 | `results/tool_spec/flex/flex_*_d96.json`、`*_bias.json` |
| O04 | 融合 kernel 内的门控用内联 PTX（`setp`/`mov`/`mul`/`ex2.approx`/`add`/`lg2.approx` 拼成 softplus）。工具已补上直线型 PTX 子集的语义后可评估：chunk / recurrent 两条路径 e_sem 1.7e-8 / 1.9e-8（常数取整级），无语义问题 | FLA `fla/ops/utils/softplus.py`（`use_gate_in_kernel=True`） | 无 | `results/tool_spec/rerun_v2/fla/fla_gdr_*_gate.json` |
| O05 | `selective_state_update(..., dt_bias=None)` 或 `D=None`（文档里的可选默认值）直接 TypeError：`*(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else 0` 被解析成 `*(... if ... else 0)` | mamba_ssm v2.2.4–v2.3.2 与 main | 崩溃（非数值） | 已知未修：state-spaces/mamba#1028（开着，2026-08-31），#912 关闭未合入 |

## 工具筛查的状态（成熟算子，2026-10-06）

- 普查：FlexAttention、Inductor、FlashAttention / mamba_ssm、FLA、Triton 教程、PyTorch 优化器（eager 与 compiled）、
  vLLM（注意力族、RoPE、KV cache 写入、V2 采样器、拒绝采样、top-k/top-p、LoRA、fused MoE、batch-invariant），
  见 `docs/census_mature_kernels_20261005.md` 与 `results/tool_spec/census/`。
- OpInfo × Inductor：onesample 1440、放宽容差 210、前向全量 773 个用例，见 `docs/opinfo_inductor_screening_20261005.md`；
  同一批用例的直接差分基线与对照见 `docs/baseline_direct_diff_20261006.md`。
- 进行中：OpInfo 的 `dynamic=True` 筛查（符号尺寸路径，先用直接差分预筛）；B016 的修复验证（运行时补掩码后用工具
  重测）。

## 相关但不计入（不是我们发现的）
- OpInfo × Inductor 筛查顺带召回或排除的：`avg_pool1d` 的 ceil_mode 反向（三个样例，相对 0.06–0.16）即 pytorch#198119 已覆盖的 1d；`index_fill` 以需要梯度的 0 维 value 张量、多个不同下标时编译后的反向触发 `aten._unique` 的 size 断言（fake 实现缺失，main 上已补，见 `torch/_subclasses/fake_impls.py` 的 `aten._unique.default`，2.10 仍有）；`normalize` 0 维与单元素 `rms_norm` 的反向候选是规格问题（精确导数为 0 / eps 随 dtype 变），不是缺陷。
- vLLM Triton top-k/top-p（`apply_top_k_top_p_triton`，V2 model runner 在批里有贪心请求、带 seed 的请求或需要 processed logprobs 时整批走它；投机解码的 rejection sampler 也用它）：我们的差分探针（`scripts/vllm/probe_topk_topp*.py`）独立发现三类偏差——少量有限 logits（语法掩码）+ top_k ≥ 有限个数 + top_p < 1 时丢掉 top-p 必须保留的 token（2 个 token 时 20–60% 的行出错，丢失质量达 0.5）；首块统计量估出的 `max_sample` 偏小、`exp` 上溢后整行不截断；bf16 并列时 top-k 只保留恰好 k 个（PyTorch 路径保留全部并列）。前两类已有上游记录：vLLM #59785（2026-10-02）、修复 PR #59804（根因同我们的分析：二分只在 [min_prob, max_prob] 内取 pivot 且用严格 >）、#60030（2026-10-05）。第三类是语义取舍。均不是工具检出。
- vLLM 其余 Triton 算子的差分筛查（`scripts/vllm/probe_gdn_conv.py`）：自带 FLA 的 chunk / recurrent / sigmoid gating（含投机解码的逐 token 状态）、`causal_conv1d_fn` / `causal_conv1d_update`（含投机解码变长）、`merge_attn_states`，全部在 bf16 噪声内，状态逐位一致。观察：`causal_conv1d_fn` 的主循环只处理宽度 2–4（宽度 5 读入 col3 却不用，update 支持到 6）；GDN 两个 decode kernel 不使用传入的 `stride_indices_tok`；KDA 分支对 `a`/`dt_bias` 的读取不带掩码。现有模型都不触发。

- Inductor 的 `pointless_convert`（`fx_passes/joint_graph.py`）把 `x.to(bf16).to(fp32)` 这类往返转换改成一次转换，显式的低精度舍入在编译后消失（与 eager 差半个 bf16 ulp），`emulate_precision_casts=True` 时也一样：已知（pytorch#181568），探针`scripts/probes/` 中的复现见 B018 同批记录。不计入。
- TRL 融合 LM head 绕过 `logits_scaling` / `lm_head_multiplier`：TRL #7439（2026-10-03 合并）。
- TRL GRPO + Liger 丢掉 MoE 辅助损失：TRL #7161。
- `torch._functorch.config.activation_memory_budget < 1` 与 dropout 同用时，编译后的反向重算 dropout 掩码（重新抽随机数），GPT-2 梯度错 33–84%（有限差分已核实）：PyTorch #190758、#190717（开着）。工具不建模随机数，这一条是对照实验发现的，召回已知问题，不计入。
- SDPA 四个后端（MATH / EFFICIENT / FLASH / CUDNN）对 float64 语义的差分筛查（`scripts/screen_sdpa_backends.py`，62 种配置 × 3 种精度，`results/screen/sdpa_backends.json`）：无异常。全掩码行四个后端都输出 0（PyTorch 2.5 起的`_safe_softmax` 约定，不是 NaN）；L=1 因果时规格梯度恰为 0，后端给出 1e-8 级噪声。这些后端是 CUDA/cuDNN，工具不覆盖，只作旁证。

## 目录

- `B00x_*.md`：中文报告（现象、机制、证据、范围、状态）。
- `upstream_drafts/`：英文 issue 草稿，是否提交由用户决定。
- `repro/`：最小复现脚本。
