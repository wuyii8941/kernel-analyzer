# 全部已观测 kernel 的自动清单与执行顺序

本页落实“面向全部已观测 kernel、以 Triton 为重点”的框架目标。它不要求把每个输出
都包装成独立问题，也不把清单中的条目自动称为已支持或已发现 bias。

## 三个计数层级

当前保存了 173,736 条输出记录。这些记录来自 27 个目录，但目录中只有 17 份不同的
任务清单；去除相同任务包的副本后共有 146,104 个不同任务位置。原始目录全部保留，
新的执行队列按任务清单摘要与 task ID 去重，不会把复制目录当作新 kernel。

自动分类覆盖全部不同任务位置，并将实际出现的计算归入矩阵乘法、normalization、
softmax、loss、activation、状态递推、RoPE、归约、卷积、embedding、索引计算、布局
变换、逐元素计算和融合计算等组。优先顺序为：

1. 已审核 reference 家族；
2. 精确 AOT 语义端点；
3. 保存的 kernel 结构签名；
4. 无法保守判断时显式标为尚未确定。

这个分类只组织覆盖。它不自动证明数学语义、reference 正确、偏差存在或根因独立。
同一个融合 kernel 的不同输出可以对应不同语义端点；按位置分类不等于把它们宣称为
多个独立 kernel 问题。

## 支持状态

每个任务只处于以下一个最强状态：

- `IDENTIFIED`：实现位置已识别，仍缺 reference 或训练参数绑定；
- `REFERENCE_AVAILABLE`：reference 已存在，仍缺训练参数绑定；
- `READY_FOR_MEASUREMENT`：reference 和目标参数均已绑定，尚无有效测量；
- `VALID_MEASUREMENT_COMPLETED`：原始采集与统一复算已通过。

现有通用能力会被自动连接，而不是等待家族名称手工登记：带实际输入的外部
`mm/bmm/addmm` 使用已有 FP32 重算；具有精确 AOT 端点、reference cut 和目标参数的
位置可使用已有 AOT 重放。后者测量闭合区域的替换作用，不能自动归因为单个 Triton
kernel。真正的单 kernel 机制仍优先使用审核过的相同输入 reference。

## 自动队列

`family_first_execution_queue_v1.json` 不读取历史数值结果。它依次安排：

1. 每个尚待测量家族的一个位置；
2. 每个新的“家族 × 实现方式 × forward/backward × 结构签名”的一个位置；
3. 其余位置。

Triton 排在常规实现之前；已有任一目录副本完成有效测量的任务不会重新排队。通用
采集器可直接处理的任务被转换为可中断续跑的 campaign；家族专用相同输入 reference
仍使用已有 runner，不会静默换成 AOT reference。

这是完整待选清单，不是要求把队列全部运行。当前执行政策只在主要算子族存在方法空缺
时选择有限任务，每个家族先验证一个；总体统计和少数深机制链优先于增加位置数。

## 复算命令

```bash
PYTHONPATH=src:. python scripts/run_training_numerical_analysis.py catalog-observed-kernels \
  --inventory results/property/numerical_coverage_v1/coverage_with_grouped_softmax_embedding_v2.json \
  --measurement-audit results/property/numerical_coverage_v1/family_execution_audit_20260907_all_three_complete.json \
  --measurement-merge-manifest results/property/numerical_coverage_v1/family_measurement_inventory_merge_manifest_v1.json \
  --catalog results/property/numerical_coverage_v1/observed_kernel_catalog_v1.json.gz \
  --summary results/property/numerical_coverage_v1/observed_kernel_catalog_summary_v1.json

PYTHONPATH=src:. python scripts/run_training_numerical_analysis.py queue-by-family \
  --catalog results/property/numerical_coverage_v1/observed_kernel_catalog_v1.json.gz \
  --output results/property/numerical_coverage_v1/family_first_execution_queue_v1.json
```

当前可读表为
[observed_kernel_coverage_v1.md](../results/property/numerical_coverage_v1/observed_kernel_coverage_v1.md)。
清单完成表示覆盖分母与缺口可追踪，不表示全部任务已动态测量，更不表示全部家族已有
数学机制和训练后果。
