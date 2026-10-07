# 现有成绩核账（三种状态，2026-10-06）

评价计划（[evaluation_plan_20261006.md](evaluation_plan_20261006.md)）引用的每一项成绩，只用三种状态之一：
**已完成且有报告**（报告与原始数据在仓库）；**已运行，待核验或待入库**；**尚未运行**。新增或变化时改本表。

| # | 成绩 | 问题 | 状态 | 依据（仓库路径） |
|---|---|---|---|---|
| 1 | 盲测两轮的独立复算：16,908,160 个输出元素、974 个投影值，0 处违反 | RQ1 | 已完成且有报告 | `results/reference_eval/blind_test_records/blind_records/v*/..._verification_report.json`；复算 `recount.json`（`scripts/blind_records_recount.py`） |
| 2 | 盲测全部 75 个程序建立完整组合参照，无不支持指令 | RQ1 | 已完成且有报告 | `blind_test_v1/coverage_check.json`、`blind_test_v2/coverage_check.json`；两轮最终计分 |
| 3 | 锁定版本的全部注册指令均映射或显式拒绝（Triton 3.6.0，229 条） | RQ1 | 已完成且有报告 | `results/reference_eval/ttir_op_registry.json`；`docs/auto_reference_results_20261002.md` |
| 4 | 真实训练捕获的 142 个 launch 全部建立参照；逐位模拟全部通过 | RQ1 | 已完成且有报告 | README「核内定位」；`docs/tool_validation.md` |
| 5 | 真实修改的反事实（11 处）与设备输出逐位一致 | RQ1 | 已完成且有报告 | `docs/tool_validation.md` |
| 6 | 语义变体 14 个全部给出 K_R ≠ f 的区间证据；预登记符号 6/6 | RQ2 | 已完成且有报告 | 两轮最终计分；`v1/blind_test_v1_phase2_verification_and_scoring.md` |
| 7 | OpInfo 直接差分基线（2149 个用例），预测 6 条成立 | RQ2、RQ3 | 已完成且有报告 | `docs/baseline_direct_diff_20261006.md`；`results/baseline/compare/` |
| 8 | 已知阳性 16/17 检出且符号正确（去掉事先声明的探针 16/16） | RQ3 | 已完成且有报告 | 两轮最终计分；由冻结判定矩阵复算一致（`recount.json`） |
| 9 | 阴性对照 13/13 与伙伴逐位相同且判定一致（v1 7 个含勘误 5，v2 6 个） | RQ3 | 已完成且有报告 | 两轮最终计分 |
| 10 | 合成数据上误报率与族错误率符合名义水平；默认检测器第 2 版的校准 | RQ3 | 已完成且有报告 | `results/reference_eval/detector_calibration_v2.json`；`docs/tool_validation.md` |
| 11 | 盲测 v2 更新层：理想响应口径 SGD 8/8、AdamW 带历史 8/8 符号与封存推导一致；实际写入口径 11–14 个程序参照未建立 | RQ4 | 已完成且有报告 | `blind_test_v2/phase3/phase3_report.md`；v2 最终计分；更新层答案 |
| 12 | AdamW8bit：状态误差到 8 对独立 run 的训练改善闭合 | RQ4、RQ5 | 已完成且有报告 | `docs/claims.md`；`results/property/result_analysis_v4/iid_training_confirmation/verification.json` |
| 13 | bf16 参数下 AdamW 的问题被无提示地定位到二阶矩写出（B005） | RQ5 | 已完成且有报告 | `bugs/README.md` B005（提交 9d44d70）；`docs/bf16_adam_beta2_prediction.md` |
| 14 | B016、B012 深案例：修复验证、预测对实测、严格包围 | RQ5 | 已完成且有报告 | `bugs/B016_*`、`bugs/B012_*`、`docs/radam_prediction_20261006.md`、`results/tool_spec/final/{scatter1_patched,radam_deep*,strict}` |
| 15 | 真实问题登记 B012–B018（B017、B018 的 B 部分在 main 上仍在；B018 非工具检出） | RQ5 | 已完成且有报告 | `bugs/README.md` 与各报告；nightly 实测输出 |
| 16 | benchmark 修复前后 38 个单位，评分 38/38 | RQ5 | 已完成且有报告 | `docs/benchmark_20261006.md`；`results/benchmark/` |
| 17 | 接口整合回归：15 个用例 0 处判定变化；干净克隆安装 38 项测试通过 | RQ1、RQ5 | 已完成且有报告 | `docs/tool_changes_20261006.md`；`results/regression/` |
| 18 | 外部受控集合（The Correctness Illusion，699 个 Triton 条件 + 141 个黑箱条件）：参照 100% 建立、独立复算 696,816 个元素 0 违反；RQ2 工具 238/238、控制组 0/404、未触发 0/57；RQ3 通过容差而平均作用经 Holm 确认 9 个 | RQ1–RQ3 | 已完成且有报告 | `docs/external_eval_results_20261006.md`；`results/external/gpuemu/`（协议 `docs/external_eval_protocol_20261006.md`，偏离 1 条） |
| 19 | 外部基线 B1–B3（默认与扫描）、消融 A1、A2 | RQ2、RQ3 | 已完成且有报告 | 同上；B4 随机算术式尚未运行（协议写明本轮不做）；A3 不适用于该集合 |
| 20 | 陌生组合子集（Unsloth 14 个入口，bf16）：32 个输出中 31 个进入评价（U11 的 loss 被 torch 后改，不评），参照全部完整；无规格差异；经 Holm 确认的平均作用全部为实现特有，bf16 上的作用全部由中间 bf16 舍入解释（RoPE 逐位核实，其余模拟定量复现） | RQ1、RQ3 | 已完成且有报告 | `docs/unfamiliar_subset_results_20261006.md`；`results/external/unsloth*`（协议偏离 3 条）；softcap loss（fp32）的机制待核实 |
| 21 | 等价轴两项校准；敏感性曲线（正参照与正负混合参照） | RQ3、RQ4 | 已完成且有报告 | `docs/statistics_calibration_20261006.md`；`results/reference_eval/calibration_equivalence.json`、`sensitivity_curves*.json`；δ 取值待确认 |
| 22 | 真实场景的逐层表（编译后的 HF RMSNorm：输出、梯度、SGD / AdamW 理想响应与实际写入） | RQ4 | 已完成且有报告 | `docs/layer_table_results_20261006.md`；`results/reference_eval/layer_table/`（含 L1/L2 的核内定位） |
| 23 | 独立交叉核验（带声明余量）：NumPy 696,816 个元素、FPCore 9,120 个元素，均无超出余量的差异；严格包围验证（无余量，mpmath 区间算术，不经 TTIR 解析器）：9,120/9,120 在 K_R 之内 | RQ1 | 已完成且有报告 | `results/external/gpuemu/declared_check*.jsonl`；`results/fpcore/summary.json`、`results/fpcore/strict_enclosure.json`；`scripts/fpcore_crosscheck.py`、`scripts/strict_enclosure_check.py` |
| 24 | 诊断收益对照（B016、B012 上工具与直接差分） | RQ5 | 已完成且有报告 | `docs/diagnosis_comparison_20261006.md` |
| 25 | 普查统一报告 | RQ5 | 已完成且有报告 | `docs/census_unified_20261006.md`；`results/tool_spec/final/summary.{md,json}` |
| 26 | 成本：分段计时（含捕获开销修正）；同等预算下的比较 | RQ1、RQ5 | 已完成且有报告 | `docs/cost_equal_budget_20261006.md`；`results/external/gpuemu/equal_budget.json` |
| 27 | 外部接入试验（未参与开发的人按指南接入） | RQ5 | 尚未运行（需要外部人员） | `docs/reuse_trial_protocol.md` |
| 28 | 上游提交 B012、B017、B018（B016 经复查为上游已知、main 已修，撤回） | RQ5 | 尚未运行（草稿待用户提交） | `bugs/upstream_drafts/`；`bugs/README.md` |

| 29 | 等价判定的零方差保护（与非零检验共用退化处理）与 CPU 回归；已存 6,359 条等价记录按新规则复算，173 条改为未建立 | RQ3 | 已完成且有报告 | `src/kernel_analyzer/reference_eval/analysis.py`（`_sample_guard`）；`tests/test_reference_eval_equivalence.py`；`results/reference_eval/equivalence_zero_variance_recheck.json` |
| 30 | 逐层表实际写入在 CUDA `foreach` 与 `fused` 路径上的重放（同一批梯度与状态） | RQ4 | 已完成且有报告 | `results/reference_eval/layer_table/cuda_replay.json`；`docs/layer_table_results_20261006.md` |
| 31 | 外部接入试验 | RQ5 | 尚未运行：需要未参与开发的人；找不到时写入局限，不用作者代替 | `docs/reuse_trial_protocol.md` |

| 32 | 重要性标定（阶段 B）：种子波动 σ = 0.0145；剂量反应阈值 d*_γ = d*_b = 0.1；9 个替换的单步作用与 ρ；预测效度——类别 9/9，Spearman 0.45，按事先规则记为「差」（噪声底以下不能排序）；δ_γ = δ_b = 0.05（冻结） | RQ3、RQ4 | 已完成且有报告 | `docs/importance_calibration_results_20261006.md`；`results/importance/`（预测在训练前提交，25a9788） |
| 33 | 第二个规模上的 ρ 稳定性（d = 512、8 层）：稳定 3、不稳定 7、无定义 10——尺子（bf16 autocast 的作用）随规模变化，ρ 与 δ 只在所测设置成立 | RQ4 | 已完成且有报告 | `results/importance/scale_check/`；`docs/importance_calibration_results_20261006.md` 第 8 节 |
| 33b | 更大剂量范围的标定 | RQ4 | 尚未运行 | 阶段 B 局限 |

| 34 | 定向搜索（阶段 C）：9 个候选按冻结尺子分类（两颗种子一致）——低精度优化器状态与 bf16 EMA 超过，bnb 8 位、bf16 随机舍入、Muon bf16 Newton–Schulz、bf16 梯度累加在尺子之内；深案例 C4（AdamW4bit：归因 81%/32%，配对训练 ΔL −0.066，已知设计取舍）与 C9（bf16 EMA：真实接口复现 +3.86，修正逐位恢复，登记 B019） | RQ5 | 已完成且有报告 | `docs/directed_search_results_20261007.md`；`results/directed_search/`；`bugs/B019_*` |
| 35 | B019 的上游提交 | RQ5 | 尚未运行（草稿待用户决定） | `bugs/upstream_drafts/B019_averagedmodel_bf16_ema.md` |

| 36 | 非精度定向搜索：4 个训练程序层面的偏差（N1 梯度累加归一化——已知阳性、N3 fp16 先裁剪后 `unscale_`、N4 衰减作用到 RMSNorm 增益、N5 调度提前一步）影子单步在两颗种子上全部检出（Holm，32 个检验）；N5 与解析值相差 0.05%；配对训练 N3 +0.61（43σ）、N1 +0.014（2.6σ_D，δ 判为之内：δ 不能跨数据设置搬用）、N4 +0.013（0.92σ，比按范数的剂量反应预测大十倍以上：δ 不分方向）、N5 ≈ 0；归因：N3 主要来自 eps（预注册，成立），N1 加重短段落与靠前位置、N4 通过 RMSNorm 增益（探索性） | RQ5 | 已完成且有报告 | `docs/nonprecision_search_results_20261007.md`；`results/nonprecision/` |
| 36b | 按 loss 敏感度加权的更新层作用（⟨∇L_val, u⟩，或按参数组标定剂量反应） | RQ4 | 尚未运行 | 非精度结果第 4 节 |

| 37 | 本质错误一轮第一阶段（独立规格 vs eager 差分）：W0 执行协议（采用审阅后的规格包 v0.4 主协议）、W7 召回（B016 四组都能发现；B014 属规格之外的编译崩溃；HF 梯度累加 4.45.2 / 4.46.0 / 4.57.3 / 5.19.0，其中 4.57.3 的计数问题只有独立规格能看到，已知 #46204）、交叉熵 / 池化 / index-scatter 的 W1–W4（1,652 个条件 × 3 seed，PyTorch eager CPU/CUDA、Inductor、nightly）：H1 成立（B020，248 个共有错误单位，一个根因）；H2 成立（F 50/50，E 与 FP64 参照 0/50）；H3 部分（P 28/50）；交叉熵零偏离 | RQ5 | 已完成且有报告（中期，待审阅） | `docs/essential_bugs_phase1_interim_20261007.md`；`docs/protocol_essential_bugs_20261007.md`；`results/essential/phase1/` |
| 38 | 本质错误一轮 2b：优化器之外的家族（注意力与打包、训练程序层、归一化、嵌入、调度、裁剪、基础算子、MoE；G6 组合、G7 数值流）的 E / P / 状态序列 / 模式 A 的 FR；各家族的 F 与模式 B 的 FR（等规格） | RQ5 | 尚未运行（按协议 v2 第 3 节的顺序进行中） | `docs/protocol_essential_bugs_phase2_20261007.md` 第 3 节；`results/essential/phase2b/coverage_plan.json` |
| 39 | B020、B021、B022 的上游提交；B015 对 #198119 的补充评论；MaxPool1d 文档公式的文档 issue | RQ5 | 尚未运行（草稿待用户决定） | `bugs/upstream_drafts/B020_*.md`、`B021_*.md`、`B015_*.md` |
| 40 | 本质错误一轮 2a：第一阶段补充 S1–S3（事后补充；S3 两处召回 #188344、#197434，index 前向的超 τ₃₂ 元素全部在 float32 求和误差界内）、S4；A1 检测器 2.2（重捕获 24 个条件，第一阶段结论不变）、A2 / A3 重新分类（E / F / P 类别逐条不变，FR 的空集一致改为未建立）、B W5 审查（B020 上 gradcheck 50/50）、C 轻量方法对照（反向错误上 gradcheck 与 F 相同）、D 触发检查、E 措辞修正、F 草稿 | RQ5 | 已完成且有报告 | `docs/essential_bugs_phase2a_record_20261007.md`；`results/essential/phase2a/`；`results/essential/phase1_supplement/` |
| 41 | 本质错误一轮 2b 优化器家族：24 个条件 × 3 seed × 10 个候选（torch for_loop / foreach / fused、编译的 step、bnb、torchao、HF），E 0 超出 τ₃₂，预注册 P 与三条状态序列零违反，模式 A 的 FR 9/10 个设置全部在 K_R 之内（Adafactor 未建立）；F 未运行（规格未交付） | RQ5 | 已完成且有报告（零结果） | `docs/essential_bugs_phase2b_record_20261007.md` 第 1 节；`results/essential/phase2b/optimizers/` |
| 42 | 本质错误一轮 2b 注意力与打包：注意力 24 个条件 × 3 seed × 11 个候选（SDPA 各后端、Inductor、flex、HF、xformers），float32 的 E 0 超出 τ₃₂、预注册 P 零违反、flex 的模式 A FR 全部在 K_R 之内；召回 #164931（flex head_dim 72 编译失败）与 #177842（2.10 cuDNN 布尔掩码用 −65504，整行被掩码时等于不加掩码）；打包 9 个条件 × 3 seed，E 与「打包等于单独运行」零违反；F 未运行（规格未交付） | RQ5 | 已完成且有报告（零结果 + 两处召回） | `docs/essential_bugs_phase2b_record_20261007.md` 第 2、3 节；`results/essential/phase2b/attention/`、`packing/` |
| 43 | 本质错误一轮 2b 训练程序层、归一化、嵌入、调度、裁剪：训练程序层 5 个条件 × 3 seed × 6 个候选（HF 4.45.2 / 4.46.0 / 4.57.3 / 5.19.0、Accelerate 与 PyTorch 文档式循环），F 开放：召回 #34191（4.45.2）、#34242（4.46.0 跨 rank）、#46204（4.57.3 计数，2b 预注册的 padding-free 不变性 9/9 看到），5.19.0 全部相容；归一化 23 个条件，float32 超差全部由条件数解释（平移后相容；模式 A 的 FR 中 K_R 与 float64 eager 一致，偏离全是 e_num）；嵌入、调度、裁剪零结果（调度 steps = 1 的边界矛盾记为约定） | RQ5 | 已完成且有报告（召回 + 零结果） | `docs/essential_bugs_phase2b_record_20261007.md` 第 4–8 节；`results/essential/phase2b/` |

阶段 A 的审计项（`docs/next_phase_plan_20261006.md` 第 1 节）：零方差保护（29）、偏斜失效范围（统计校准文档第 2 节）、RQ2 两栏与核验两栏（外部评价结果文档）、逐层表的设备与路径（30）、口径修正（外部评价、Unsloth、成本文档）已关闭；外部接入试验（31）未做。

注：盲测揭盲后，v1、v2 只作回归集；之后的泛化成绩只来自外部语料与陌生组合。
