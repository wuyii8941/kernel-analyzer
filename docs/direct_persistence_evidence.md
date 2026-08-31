# Direct Persistence 证据表

> 本表是 cold-start 32 步的 direct/feedback/actual 归因表，用来解释短程参数分离由哪一部分组成。它不分配当前长程标签；长程结果见 `results/property/declared_persistent_4096/summary.md`。
> 当前论文顺序和统计定义见 `docs/current_mainline.md` 与 `docs/method.md`。

## Direct、feedback 和 actual 必须分开

记：

- `L_t`：在同一个状态上只改变 candidate/repair 算子得到的有效更新差异；
- `B_t`：两条轨迹已经分开后产生的反馈差异；
- `D_t`：实际参数分离变化。

我们检查：

```text
D_t = L_t + B_t + recurrence_residual_t
```

仅仅看到 feedback 的 `A` 较高，不能说 feedback 贡献较大。当前贡献表固定使用最终实际分离方向做带符号投影，并同时保存累计长度、路径能量、`A` 和 cosine。

## 当前可重算的三条 32 步记录

| 案例 | direct result | feedback result | actual result | direct share | feedback share | 当前分类 |
|---|---:|---:|---:|---:|---:|---|
| Liger | `A=1.720` | `A=3.494` | `A=3.489` | 0.0046 | 0.9954 | feedback dominated |
| Phi | `A=1.029` | `A=5.075` | `A=1.711` | 0.4393 | 0.5607 | feedback dominated |
| Qwen | `A=0.957` | `A=1.191` | `A=1.508` | -0.0862 | 1.0862 | feedback dominated |

这些 share 是带符号的，所以可以小于零或大于一。它们来自已有的累计长度和 cosine；并没有假装拥有未保存的逐步完整向量。

## 数据边界

旧 v3 文件没有保存三条序列的完整逐步向量，因此：

- 三个累计结果的 3×3 内积矩阵可以重算；
- 每一步的完整 cross-Gram 不能重算；
- v4 明确记录这一缺口，并要求新的采集保存逐步向量或可验证的 cross-Gram；
- 缺数据时输出 `ABSTAIN`，不把缺失当成零。

## 统计校正

校正分三组：预先声明的 3 行、结果盲抽样的 12 行、全部 15 行敏感性分析。Holm 是主要结果，BH 只作探索性补充。

回溯结果分为两种视图：

1. 名义短程视图：3 个 positive、12 个 negative；
2. confirmed-only 短程视图：2 个 confirmed positive、12 个已解决 negative，`0543` 单独列为 unresolved candidate。

Holm/BH 的逐行结果和三个 family 的完整成员清单见
`results/property/direct_persistence_v4/multiplicity.json`。`0543` 仍属于
12 行 discovery family，但不会因为校正失败而被改标为 negative。
