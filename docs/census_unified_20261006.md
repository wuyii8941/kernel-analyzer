# 普查统一报告（评价计划工作项 G / M3，2026-10-06）

把此前分散在 `docs/census_20261005.md`、`docs/census_mature_kernels_20261005.md`、
`docs/opinfo_inductor_screening_20261005.md` 与 `bugs/README.md` 的普查整理成一张表，口径统一为工具的分解
（e_num = K − K_R，e_sem = K_R − f）与 `scripts/tool_spec_summary.py` 的分档。成绩按三种状态写。本报告不新增运行，
只重算汇总（`results/tool_spec/final/summary.{md,json}`）。

## 1. 工具对规格检查（成熟库的 Triton kernel）

| 组件 | 用例 | 输出 | 参照完整 | 无 | 常数取整级 | 小 | 候选 | 混合 | 判不了 | 报错 | 不评 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Inductor 第一批（torch 2.10） | 49 | 52 | 51 | 33 | 15 | 3 | 1 | 0 | 0 | 0 | 5 |
| Inductor 第二批 | 40 | 64 | 63 | 27 | 34 | 2 | 0 | 0 | 1 | 2 | 2 |
| Inductor 第三批 | 25 | 28 | 24 | 18 | 2 | 0 | 1 | 3 | 4 | 0 | 1 |
| FlexAttention | 27 | 51 | 51 | 0 | 47 | 0 | 0 | 4 | 0 | 0 | 8 |
| FlexAttention decode | 10 | 10 | 10 | 0 | 9 | 0 | 0 | 1 | 0 | 0 | 0 |
| FlashAttention 仓库与 mamba_ssm 的 Triton 算子 | 56 | 78 | 78 | 69 | 9 | 0 | 0 | 0 | 0 | 0 | 22 |
| FLA 门控 delta rule | 24 | 50 | 50 | 3 | 41 | 1 | 0 | 5 | 0 | 1 | 0 |
| Triton 官方教程 | 9 | 17 | 11 | 2 | 3 | 0 | 0 | 6 | 6 | 0 | 0 |
| vLLM `unified_attention` | 15 | 15 | 15 | 11 | 1 | 0 | 3 | 0 | 0 | 0 | 0 |
| 编译后的优化器（第二批） | 13 | 12 | 12 | 2 | 4 | 1 | 2 | 3 | 0 | 0 | 26 |
| RAdam 定向用例 | 4 | 12 | 12 | 2 | 6 | 1 | 3 | 0 | 0 | 0 | 0 |
| **合计** | **272** | **389** | **377** | 167 | 171 | 8 | 10 | 22 | 11 | 3 | 64 |

（「不评」：输出不是 Triton 写出，或最后一次 Triton 写入之后被 torch 改过；「混合」：依赖 launch 之间由 torch 算出的
中间值；「判不了」：参照未建立，主要是自旋锁 + 原子的 layer norm 反向与没有声明语义的 libdevice 函数。）

**候选档 10 个输出的处置：**

| 用例 | e_sem 相对 RMS | 处置 | 状态 |
|---|---|---|---|
| `ind_avg_pool2d_ceil_pad_bwd` | 9.4·10⁻² | B010：Inductor avg_pool 反向 ceil_mode 除数错 | 上游已知未修（pytorch#198119） |
| `ind_sort_values_bwd` | 1.4·10⁻³ | 约 157 万坐标中 2 个被证实：随机 fp32 输入中的相等值在不稳定排序下交换梯度，规格本身未定义并列次序 | 规格歧义，不是缺陷 |
| `ua_bidir_sw8_mha`、`ua_bidir_sw24_gqa4`、`ua_perseq_causal_sw8` | 0.905 / 0.115 / 0.216 | B013：vLLM 非因果滑窗的 V 掩码按块首行算右边界 | 未见上游报告 |
| `opt_rprop_etas`（param、prev） | 2.6·10⁻³ / 0.56 | 同一进程中 Dynamo 把 `etaminus` 变为 f64 参数、另一处仍为 f32 常数，`sign == etaminus` 按实数语义不等而执行时相等：舍入决定的分支，总误差为 0 | 不是缺陷（`docs/tool_changes_20261005.md` 第 6 节） |
| `opt_radam_b9995_step5`、`_b9999_step2`、`_b9999_step6` | 1.45·10⁻² / 8.2·10⁻³ / 2.9·10⁻⁵ | B012：RAdam 在 fp32 里算整流项，判据 ρ_t > 5 翻转 | 未见上游报告；草稿待提交 |

「小」档 8 个逐条已解释（插值的 fp32 坐标比例、近似算法的常数、默认 β₂ 下 B012 的弱信号）；常数取整级为 fp32 常数
（log2(e)、√(2/π)、eps 等）。

## 2. OpInfo × Inductor 筛查与直接差分基线

onesample 1440、放宽容差 210、前向全量 773 个用例（`docs/opinfo_inductor_screening_20261005.md`）；同一批用例的
直接差分基线 2149 个（`docs/baseline_direct_diff_20261006.md`）。由此登记 B014（池化反向的单元素参数，编译崩溃）、
B015（avg_pool3d ceil 反向）、B016（单元素目标的 scatter，掩码丢失）、B017（Dynamo 共享代码对象的守卫）。直接差分
在完整用例上同样能发现这些错误，工具的增量在差的拆分与定位（见第 5 节与 M3 诊断对照）。

## 3. 模型集成普查（不用工具，打补丁前后对照）

Liger 0.7.0 / 0.8.4 / main 的模型集成（B001、B002、B003、B004）；transformers 5.18.0 全部 178 个 causal-LM 的重算一致性
（128 个可构建者三种设置下逐位相同）；Unsloth 2026.9.14 的 17 个配置（fp32，全参 17/17、LoRA 16/17 一致）。详见
`docs/census_20261005.md`。

## 4. 本轮新增（M2）

- 外部受控集合（The Correctness Illusion，699 + 141 个条件）：`docs/external_eval_results_20261006.md`。
- 陌生组合子集（Unsloth 的 14 个 Triton 入口，bf16）：`docs/unfamiliar_subset_results_20261006.md`（运行中）。

## 5. 发现与上游状态

| 编号 | 发现手段 | 工具检出 | 上游 |
|---|---|---|---|
| B001、B002 | 普查脚本 | 是（e_sem 1.24 / 1.18） | B002 main 已修（未发版）；B001 未报告 |
| B010 | 工具（Inductor 筛查） | 是 | 已知未修 pytorch#198119 |
| B011 | 工具（教程筛查） | 是 | 已知（教程注释 FIXME） |
| B012 | 工具（编译优化器筛查）→ 定向用例 | 是 | 未报告；草稿待用户提交 |
| B013 | 读代码 → 工具确认 | 是 | 未报告 |
| B014 | OpInfo × Inductor 运行时暴露（编译错误） | 否（非数值） | 未报告 |
| B015 | 工具（OpInfo × Inductor） | 是 | 未报告（nightly 2.15.0.dev20260907 实测仍在） |
| B016 | 工具（OpInfo × Inductor） | 是 | **已知、main 已修**：pytorch#178871 / #179833（2026-04-16）；此前「未见报告」系检索遗漏，草稿撤回 |
| B017 | 工具报警 → 读 Dynamo 源码 | 报警来自工具 | 未报告；草稿待用户提交 |
| B018 | 读代码 → 直接差分探针 | 否（cuDNN / Flash，不是 Triton） | A 部分 main 已修；B 部分未报告，草稿待用户提交 |

## 6. 状态

- 已完成且有报告：第 1–3 节（数据与汇总在仓库）；第 5 节的登记。
- 已运行，待核验或待入库：Unsloth 子集（M2，运行中）。
- 尚未运行：新的全库普查（计划明确不做）；B012、B017、B018 的上游提交（由用户提交；B016 已知已修，不提交）。
