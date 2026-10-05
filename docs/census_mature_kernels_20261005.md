# 成熟算子的工具筛查（2026-10-05）

目标：用工具本身（TTIR 自动参照 K_R、区间残差、统一判定层与检测器 2.1）在认可度高、影响大的开源算子上
找 ④（算法与规格不符）与 ②（输入来源不一致）类问题。入口 `scripts/tool_spec_check.py`，用例
`scripts/tool_spec_cases_*.py`，统一结果 `results/tool_spec/final/`，汇总表
`results/tool_spec/final/summary.md`（`scripts/tool_spec_summary.py` 生成）。

## 1. 方法

对每个用例（一次 kernel 调用 + 规格 f）在 96 个种子（0–31 开发、32–95 确认；FLA 用 48 个）上：

- K：kernel 实际写出的值（捕获）；
- K_R：同一调用全部 Triton launch 的 TTIR 组合求值，实数语义的有向区间；
- f：被实现或被替换函数的文档语义，在同一输入上以 float64 求值（声明界 2⁻⁴⁰·max|f|；RoPE 用例为严格区间）；
- e_num = K − K_R（数值：舍入、TF32、近似指令）；e_sem = K_R − f（语义：公式、常数、变体）；
- 两者都进 `assess_units`（R1/R2/R3/R5 与检测器）。e_sem 的判读按量级分档：未检出；常数取整级
  （检出但相对 RMS < 1e-7，例如 fp32 舍入后的 log2(e)）；小（1e-7–1e-5）；候选（≥ 1e-5）。

不进入判定、单独报告的情形：输出不是 Triton 写出的（cuBLAS、ATen 归约）；最后一次 Triton 写入之后又被
torch 操作改过；写它的程序中止（没有语义的 libdevice 函数等）；依赖 launch 之间由 torch 算出的中间值
（"混合"：K_R 带上游数值误差）。

## 2. 覆盖

| 组件 | 版本 | 用例 | 内容 |
|---|---|---|---|
| PyTorch Inductor（torch.compile） | torch 2.10.0 | 86 + 77 + 68 | 归一化、softmax 族、激活、损失（含边界项）、池化/插值/填充、batch_norm 训练统计、scatter_reduce、排序与并列、特殊函数、编译后的 14 种优化器更新 |
| PyTorch FlexAttention | torch 2.10.0 | 27 | 因果/滑窗/文档掩码、softcap、ALiBi（含可学习斜率）、GQA、非整块长度、Q≠KV、head_dim 96、Dv≠D、可学习 bias、经 LSE 的梯度；前向与反向 |
| FlashAttention 仓库 Triton 算子 | main | 37 | rotary（交错、部分、偏移、变长、共轭、原地）、layer_norm / rms_norm（残差、prenorm、零中心权重、并行、rowscale）、cross_entropy（smoothing、logit_scale、z-loss） |
| mamba_ssm Triton 算子 | main | 19 | gated RMSNorm（分组、门控前后）、SSD 分块扫描（D、z、dt_bias、softplus、dt_limit、初态/末态、分组、非整块长度、seq_idx）、selective_state_update |
| FLA 门控 delta rule | main（0.6.0） | 24 | chunk 与 fused_recurrent：GVA、kernel 内 l2norm、sigmoid β、负特征值、kernel 内门控（内联 PTX）、初态/末态、V-first 状态、变长、非整块长度；chunk 的反向 |
| Triton 官方教程 | v3.6.0 | 9 | 05-layer-norm、06-fused-attention（前向、反向、因果与非因果） |
| vLLM `unified_attention`（TRITON_ATTN） | main 51eeb0c | 15 | 分页 KV、因果 / 双向 / 按序列 causal、滑窗（含过期块 NaN）、softcap、sinks、ALiBi、GQA、head_dim 80、decode 3D 分段归约 |

## 3. 结论

- **上游未知的新问题（1 条）**：`torch.optim.RAdam` 的 capturable 路径与 `torch.compile(opt.step)` 在 float32 里
  算整流项 ρ_t = ρ_∞ − 2t·β₂ᵗ/(1 − β₂ᵗ)（两个接近 ρ_∞ 的数相减，β₂ 先被舍入到 float32），β₂ 接近 1 时整流
  判据 ρ_t > 5 在前几步翻转（B012）。工具在编译后优化器的筛查里先对 `opt_radam` 报出"小"档 e_sem（默认
  β₂、第 7 步，7.3e-7）；顺着公式定向构造用例后，β₂ = 0.9995 第 5 步 param 的 e_sem 为 1.45e-2（≈ 整步更新），
  e_num 2.4e-7。复现：更新误差在默认 β₂ 下达 6e-3，在 0.9995 / 0.9999 的翻转步达 0.99，在 0.99999 下达
  13–18 倍；eager 的非 capturable 路径保持在 2e-4–2e-3。PyTorch main 代码相同，检索未见报告。
- **上游未知的新问题（第 2 条，vLLM）**：TRITON_ATTN 的 `unified_attention` 在非因果滑窗时，V 的置零掩码按 q 块第一行算窗口右边界，块内后面的行丢掉窗口右侧的 key（B013，vLLM #45163 引入，v0.24.0–v0.31.0）。三个用例 e_sem 相对 RMS 0.905 / 0.115 / 0.216，区间证实；BLOCK_Q = 1 与因果的对照为 0。vLLM PR #51257 新加、尚未合入的测试配置下误差 0.15。现有模型的默认配置双向段短于窗口，属潜在缺陷。
- **检出、但上游已有记录的问题（2 条）**：
  - Inductor `avg_pool2d(ceil_mode=True, count_include_pad=True)` 反向梯度错误（B010，pytorch#198119，未修）；
  - Triton 教程 fused attention 的非因果反向错误（B011，教程测试里的 FIXME）。
  两者都是工具在不知道答案的情况下检出的（e_sem 相对 9.4e-2 与 ≈1.0；e_num 分别为 3.6e-8 与 3e-4，
  说明 kernel 忠实执行了它自己的、错误的算法）。
- **其余全部在常数取整级或干净**：FlexAttention 的系统偏差已核实是 fp32 舍入后的 log2(e)（温度偏 1.3e-8）；
  Inductor 插值的 5e-7 来自 fp32 坐标比例常数，特殊函数的 1e-7 级是 Cephes 近似的设计精度；FLA、mamba、
  FlashAttention 各算子的语义与文档公式一致（FLA 反向的 dg 因依赖 torch 归约出的中间值而记为混合，其
  1.7e-3 是上游 TF32 数值误差，不是语义偏差）。
- **② 类观察**：FlexAttention 反向（Q_LEN 非块倍数时的 dv）、flash-attn 与 mamba 的归一化反向都依赖
  "不带 `other` 的掩码通道为 0"，这在 Triton 文档里是未定义的；Triton 3.6 的 NVIDIA 后端把寄存器置 0
  （PTX 已核实），所以结果正确，属可移植性风险（O01）。
- **规律**：问题都出在默认配置之外——β₂ 接近 1 的 RAdam、ceil_mode 的越界窗口、非因果的教程反向、
  phi3 的部分旋转（B002）。成熟代码在默认配置和常用配置上经得起 fp32 语义层面的检查；工具的价值在于
  能系统地扫这些边角配置，并把"kernel 的算法错"（e_sem）和"kernel 的舍入大"（e_num）分开。
- **已知的误标**：FlexAttention 的 ALiBi 用例把 setup 里构造的斜率常量当成了外来中间值（"混合"），
  其 e_sem 为 1.2e-8，不影响结论。

## 4. 工具本轮的改进（因这次筛查而发现的工具问题）

见 `docs/tool_changes_20261005.md`：一处正确性修正（中止程序后的组合求值）、四处精度修正（整数 0 吸收
未定义、自比较、bool 字节、自定义 scan）、掩码通道补 0 的 PTX 核实假设、内联 PTX 片段解释器、多操作数
scan、混合来源判定。每处都有修正前失败、修正后通过的测试。

## 5. 普查表（`results/tool_spec/census/summary.md`）

正式结果：FlexAttention、FlashAttention/mamba、FLA、Triton 教程、RAdam 定向用例全部用最终版工具；Inductor 三批的
多 launch 用例用最终版工具重跑，单 launch 用例沿用上一轮（后加的规则只影响多 launch 组合）。
`ind_logcumsumexp_bwd` 的正式重跑仍在进行，表中用上一轮结果。

| 组 | 用例 | 输出 | 候选 (≥1e-5) | 小 (1e-7–1e-5) | 常数取整级 | 未检出 | 混合 | 判不了 | 不评 | 报错 |
|---|---|---|---|---|---|---|---|---|---|---|
| Inductor 第一批 | 86 | 89 | 1（B010） | 6 | 22 | 60 | 0 | 0 | 5 | 0 |
| Inductor 第二批 | 77 | 100 | 0 | 2 | 54 | 43 | 0 | 1 | 10 | 2 |
| Inductor 第三批 | 68 | 55 | 1（排序并列，规格歧义） | 1 | 6 | 40 | 3 | 4 | 8 | 0 |
| FlexAttention | 27 | 51 | 0 | 0 | 47 | 0 | 4 | 0 | 8 | 0 |
| FlashAttention + mamba | 56 | 78 | 0 | 0 | 9 | 69 | 0 | 0 | 22 | 0 |
| FLA | 24 | 50 | 0 | 1 | 41 | 3 | 5 | 0 | 0 | 1 |
| Triton 教程 | 9 | 17 | 0 | 0 | 3 | 2 | 6（含 B011） | 6 | 0 | 0 |
| RAdam 定向 | 4 | 12 | 3（B012） | 1 | 6 | 2 | 0 | 0 | 0 | 0 |
| vLLM `unified_attention` | 15 | 15 | 3（B013） | 0 | 1 | 11 | 0 | 0 | 0 | 0 |
| 合计 | 366 | 467 | 8 | 11 | 189 | 230 | 18 | 11 | 53 | 3 |

分档按 `docs/tool_changes_20261005.md` 第 5 节的规则重算：e_sem 由规则 / 检测器确认，或有坐标被区间证实
（区间 K_R − f 不含 0）即算检出。与上一版相比 20 个输出改档：17 个转入常数取整级或"小"档（fp32 常数），
3 个 Triton 教程 / lgamma 反向输出转入"混合"；Inductor 第三批新增的候选是 `ind_sort_values_bwd` 的 dx，
只在约 157 万个坐标中的 2 个上被证实，是一对梯度互换：随机 fp32 输入偶有相等值，`torch.sort` 默认不稳定，
并列元素的次序本身未定义，属规格歧义。

"小"档逐条已解释：插值 6 条（fp32 坐标比例常数）、digamma / i0e / i1e / vector_norm 反向（近似算法或常数）、
默认 β₂ 的 RAdam（即 B012 在默认配置下的弱信号）。"判不了"：Triton 教程 layer norm 反向（自旋锁 + 原子操作，
工具不建立参照）、若干 libdevice 函数（`erfinv`、`hypot` 反向等没有声明语义）。
