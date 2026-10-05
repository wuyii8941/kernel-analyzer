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

## 正在进行的工具筛查（成熟算子）

- PyTorch FlexAttention（Inductor Triton 模板）：12 种配置 × 前向/反向，见 `results/tool_spec/flex/`。
  - 反向在 Q_LEN 不是块大小整数倍时，dv 依赖一个不带 `other` 的掩码读取（越界行的 LSE）。Triton 3.6 的
    NVIDIA 后端把这个寄存器置 0，所以结果正确；但这依赖实现行为，文档里该值是未定义的。记作低严重度观察。
  - head_dim 96、以及带可学习 bias 的 score_mod，在 A6000（共享内存 101 KB）上编译失败
    （"No valid triton configs. OutOfResources"），属可用性问题，未查上游。
- PyTorch Inductor（torch.compile）：86 个用例（归一化、softmax 族、激活、损失、池化/插值、填充），
  对照 eager 语义。第一轮（工具修正前）结果：除 B010 外，e_sem 都在常数取整量级（≤ 7e-7；插值的坐标比例常数在 fp32 下取整）；修正后重跑中。
- FlashAttention 仓库 Triton 算子（rotary、layer_norm/rms_norm、cross_entropy）与 mamba_ssm（gated RMSNorm、
  SSD、selective_state_update）、FLA 门控 delta rule：进行中。
  初步：e_sem 有 1e-8 量级的系统偏差，已核实来自 fp32 舍入后的 log2(e)（常数取整）：规格的温度乘 (1 − 1.33e-8) 后 e_sem 从 1.76e-8 降到 8.5e-16。量级可忽略。

## 相关但不计入（不是我们发现的）

- TRL 融合 LM head 绕过 `logits_scaling` / `lm_head_multiplier`：TRL #7439（2026-10-03 合并）。
- TRL GRPO + Liger 丢掉 MoE 辅助损失：TRL #7161。

## 目录

- `B00x_*.md`：中文报告（现象、机制、证据、范围、状态）。
- `upstream_drafts/`：英文 issue 草稿，是否提交由用户决定。
- `repro/`：最小复现脚本。
