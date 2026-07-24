# ForkCert 执行计划 v4（2026-07-13 修订）

## 修订动机

20-step 轨迹分析揭示了一个关键发现：fusion partition 修复在单步尺度有效（recovery ratio 0.45 at step 1, 0.39 at step 5），但到 step 20 完全消失（ratio 1.056）。原因是 `max_fusion_size=2` 不减小 delta 幅度（RMS 反而大 6.5%），只是重排了误差模式——恰好在当前 checkpoint 不触碰任何边界。

这改变了论文主线：**贡献不是"发现并修复 fork"，而是"发现一种结构性训练脆弱性，建立因果归因方法，证明它是持续的且不可通过单点修复消除，因此需要持续检测"。**

---

## 当前证据状态

| RQ | 核心结果 | 状态 | 缺口 |
|---|---|---|---|
| RQ1 存在性 | 5 clipping fork + 29 sampling fork，全过混杂清单 | **完成** | vLLM 跨引擎已完成并终止（composite pair） |
| RQ2 归因 | fusion partition scheduling 因果归因，step-5 和 step-11 独立复现；其他 Inductor 选项（epilogue_fusion/pattern_matcher/aggressive_fusion）全部无效 | **单步完成** | 尚未对全部 5 个 clipping fork 做归因；sampling fork 归因未开始 |
| RQ3 预测 | margin-only AP=0.09，全部 5 正例在 top 1%；仅 5 个正例 | **初步** | 正例太少，无法选最终预测器 |
| RQ4 因果后果 | 单步 54.9% 回归；20-step 修复消失（新发现）；twin 6.49x 跳变比 | **核心发现已到位** | twin 6.49x 有共因混杂，需 matched-step 对照 |
| RQ5 检测效用 | fork signal precision 99.5%/recall 35.9%；3 个粗粒度注入 | **框架就位** | 注入太粗暴，缺对抗性案例，基线未调优 |

---

## 论文主线（修订后）

```
执行路径差异（eager vs compile）
  → 算子级数值扰动（fusion partition scheduling 改变 logprob）
  → 跨越训练离散决策边界（clipping/sampling threshold）
  → 训练语义分叉（gradient 归零 / 采样 token 改变）
  → 单步因果归因有效（修复 fusion → 54.9% 参数回归）
  → 但修复不持续（delta 被重排非消除，新 fork 持续涌现）
  → 结论：fork 是结构性的、持续的训练脆弱性
  → 需要持续检测而非一次性修复
```

核心创新点：
1. **发现训练决策边界作为浮点差异放大器的机制**——不是"浮点有误差"，而是训练算法的离散结构把无害的连续误差变成了语义分叉
2. **构造了可干预的因果归因方法**——通过 Inductor 优化选项消融定位到具体编译优化类别
3. **证明了脆弱性的结构性**——修复不持续不是方法的失败，而是现象本身的特征：系统在每一步都处于 fork 风险中

---

## 修订后的优先队列

### 第 1 位：matched-step 反事实（修补 RQ4 共因混杂）
**目标**：证明 fork step 的梯度跳变不仅仅是"大梯度步恰好也容易 fork"。
**方法**：用已有 twin 训练数据，为每个 fork step 按梯度范数匹配一个非 fork step 作为对照，重算跳变比和 bootstrap CI。
**验收**：匹配后跳变比 CI 仍不含 1 → 因果 claim 成立；CI 含 1 → 降级为相关性，如实报告。
**预计耗时**：纯分析，无需 GPU，用现有 `phase6_twin_stepwise.md` 的 101 条 trajectory 数据。
**价值**：这是核心 claim 的最后一个 confound，不解决则 RQ4 的 6.49x 不可写进论文。

### 第 2 位：补全剩余 clipping fork 的 fusion 归因（强化 RQ2）
**目标**：对全部 5 个自然 clipping fork 完成 fusion partition 归因，得到归因成功率。
**现状**：step-5 t80 和 step-11 t88 已完成（fusion size=2 消除 fork）。step-11 t72 已有 probe 但尚未确认 size=2 是否有效。step-14 的两个 fork 未冻结 checkpoint，不可重放。
**方法**：
  - step-11 t72：运行 `max_fusion_size=2` 探针（复用现有脚本）。
  - step-14：如实报告 2/5 不可重放，归因覆盖率 = 可重放 fork 中的成功率。
**验收**：≥60% 可重放 fork 被 fusion 修复消除 → 归因成立；否则报告交互效应。
**预计耗时**：1 次 GPU 运行，~30 分钟。

### 第 3 位：20-step 非持续性的完整报告（将新发现写成正式产出）
**目标**：把刚做出的轨迹分析从 `trajectory_analysis.md` 升级为正式报告，补充以下分析：
  - 每一步的 batch-level fork count（需对 20 步中的关键步骤运行 batch scan，而非仅 step 5）
  - A/B/C 三条轨迹的 loss/gradient norm 收敛曲线对比
  - 明确写出"修复不持续"的机制解释（delta 重排而非消除）
**验收**：报告完成且数据自洽。
**预计耗时**：已有分析基础，主要是补充几步的 batch scan + 写报告，~2-3 小时含 GPU。

### 第 4 位：RQ5 变异集构造（v3.1 Q1，可与前 3 位并行）
**目标**：构造 12-15 个变异算子，用于 oracle 对比实验。
**关键修订**：基于"修复不持续"的发现，RQ5 的 framing 需要调整——
  - 不再声称 fork signal 是"优于 delta 的通用 oracle"
  - 而是：fork signal 提供**语义级过滤**（precision 99.5%），delta magnitude 提供**数值级扫描**（recall 96%），两者互补
  - 变异集必须包含两类对抗性案例：(a) 大 delta 不翻转决策，(b) 小 delta 恰翻转决策
**验收**：预注册变异清单 git commit；包含上述两类对抗性案例。
**预计耗时**：纯构造，单卡，~1 天。

### 第 5 位：RQ5 正式 oracle 对比
**目标**：在预注册变异集上运行 fork signal vs 调优后 delta 基线。
**方法**：
  - 对 delta 基线做超参搜索（rtol/atol 组合取其最优）
  - 指标：precision/recall、固定 triage 预算下检出数、误报率
  - 明确 framing：不是 fork > delta，而是两者在不同 operating point 上各有优势
**验收**：结果写出并附 framing；若 fork 无优势也如实报告。

### 第 6 位：RQ3 扩充 + 收尾
**目标**：在更多 checkpoint 上跑 margin scan 增加正例数量，做 held-out 校准。
**现实约束**：只有 step-5 和 step-11 有冻结 checkpoint，正例极少。如果无法获得更多 checkpoint，则 RQ3 降级为"feasibility demonstration"。
**收尾**：可主张/不可主张边界清单、全部 certificate 归档、复现脚本。

---

## 从计划中移除或降级的项目

| 项目 | 原优先级 | 处置 | 原因 |
|---|---|---|---|
| vLLM 跨引擎 fork scan | 第 1 位 | **已完成并终止** | HF-vLLM 是 composite pair，delta 过大（mean 0.35），139/512 branch 变化，无法做单算子归因；T4 上 STOP 决策正确 |
| vLLM 上重跑 signed bias | 第 1 位 | **移除** | composite pair 的 signed bias 不可解释为单一来源；同框架 null 结果已够 |
| 长期轨迹修复持续性验证 | 隐含 | **已完成（负结论）** | 20-step 分析已证明修复不持续，这是一个完整的发现 |
| DEEPOPFUZZ 语料相关 | 暂不执行 | **保留为 future work** | 按 v3.1 指示 |
| 第二个模型（Qwen3-1.7B）复测 | should | **降级为 future work** | 当前主线 claim 尚未完全闭合，不应分散资源 |

---

## 关键风险与降级预案

1. **matched-step 对照后 6.49x 不显著** → RQ4 降级为"相关性观察"，论文措辞改为"fork step 与更快参数分歧相关，但共因效应尚未排除"。这仍然是有价值的观察。

2. **step-11 t72 fusion 归因失败** → 归因覆盖率降为 2/3（67%），仍过 60% 验收线。若低于 60%，叙事改为"部分 fork 需要多算子交互解释"。

3. **RQ5 变异集上 fork signal 无优势** → 如实报告 negative finding。论文仍有 RQ1-RQ4 四个完整贡献。RQ5 降级为"互补性展示"而非"优越性证明"。

4. **RQ3 正例不足无法校准** → 降级为 feasibility demonstration，写明"5 个正例不足以选择预测器，但 margin 作为风险排序特征是有效的（all-5-in-top-1%）"。

---

## 论文结构草案（基于修订后的证据链）

1. **Introduction**：训练算法的离散决策边界（clipping/sampling）可以放大本应无害的执行路径数值差异
2. **RQ1**：自然 fork 存在（5 clipping + 29 sampling），发生率与 margin 分布一致
3. **RQ2**：因果归因方法——fusion partition 消融定位到编译优化类别，非 hook、非笼统归因
4. **RQ4**：单步因果修复有效（54.9% 回归）+ **修复不持续**（20-step ratio→1.05）+ matched 反事实（待完成）
5. **RQ3**：margin 是有效的风险排序特征（top-1% 包含全部正例）
6. **RQ5**：fork signal vs delta magnitude 的互补性
7. **Discussion**：fork 是结构性的持续脆弱性 → 需要检测而非一次性修复 → 对 RL 训练可复现性的含义

---

## 立即可执行的命令

```bash
# 第 1 位：matched-step 反事实（无需 GPU）
PYTHONPATH=src:. /data1/tzh/conda-envs/forkcert/bin/python -c "
# 从 twin 数据中提取 fork/no-fork step 的梯度范数，做 matched 对照
# （需要写成独立脚本）
"

# 第 2 位：step-11 t72 fusion 归因
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:. /data1/tzh/conda-envs/forkcert/bin/python scripts/phase8_compile_fusion_probe.py \
  --fork-id clip-step11-grpo_000003_692fbb817526-t72 \
  --intervention max_fusion_size_2
```
