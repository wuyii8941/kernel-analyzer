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

## 3. 结论（初稿，待正式重跑数字）

- **检出的实质问题都已在上游有记录**：
  - Inductor `avg_pool2d(ceil_mode=True, count_include_pad=True)` 反向梯度错误（B010，pytorch#198119，未修）；
  - Triton 教程 fused attention 的非因果反向错误（B011，教程测试里的 FIXME）。
  两者都是工具在不知道答案的情况下检出的（e_sem 相对 9.4e-2 与 ≈1.0，e_num 分别为 3.6e-8 与 3e-4，
  说明 kernel 忠实执行了它自己的、错误的算法）。
- **其余全部在常数取整级或干净**：FlexAttention 的系统偏差已核实是 fp32 舍入后的 log2(e)（温度偏 1.3e-8）；
  Inductor 插值的 5e-7 来自 fp32 坐标比例常数，特殊函数的 1e-7 级是 Cephes 近似的设计精度；FLA、mamba、
  FlashAttention 各算子的语义与文档公式一致。
- **② 类观察**：FlexAttention 反向（Q_LEN 非块倍数时的 dv）、flash-attn 与 mamba 的归一化反向都依赖
  "不带 `other` 的掩码通道为 0"，这在 Triton 文档里是未定义的；Triton 3.6 的 NVIDIA 后端把寄存器置 0
  （PTX 已核实），所以结果正确，属可移植性风险（O01）。
- **与 Liger 对比**：Liger 的两条 RoPE 问题（B001、B002）是普查脚本先发现、工具复核检出；成熟算子这一轮
  没有找到上游未知的算法层面错误。这本身是一个结论：这些代码库在 fp32 语义层面经得起检查，问题集中在
  新集成、新功能和测试未覆盖的配置（B002 的 phi3 + 部分旋转、B010 的 ceil_mode 越界窗口、B011 的非因果）。

## 4. 工具本轮的改进（因这次筛查而发现的工具问题）

见 `docs/tool_changes_20261005.md`：一处正确性修正（中止程序后的组合求值）、四处精度修正（整数 0 吸收
未定义、自比较、bool 字节、自定义 scan）、掩码通道补 0 的 PTX 核实假设、内联 PTX 片段解释器、多操作数
scan、混合来源判定。每处都有修正前失败、修正后通过的测试。
