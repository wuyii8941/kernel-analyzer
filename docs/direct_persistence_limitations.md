# Direct Persistence Screen 的边界和后续实验

当前结果支持的是：

> 在统一的 cold-start AdamW 设置下，短程有效更新方向性可以作为高召回的后续检查分诊器；它不是通用安全分类器。

## 尚未完成的三类证据

### 1. optimizer 状态

需要在同一 weights、输入和 gradient difference 下比较：

- 已捕获 moments；
- moments reset；
- stateless SGD。

还需要在真实训练早期、中期、后期分别捕获完整 weights、输入、gradients 和 moments。不能把晚期 moments 人工安装到早期 gradient 上并称为真实训练阶段结果。

### 2. 未见实现

先机械冻结 held-out pool，再冻结短筛、严重度指标和 tolerance 基线。所有 eligible rows 都要完成 16 步短筛和 32 步确认，包括短筛没有升级的 rows。

同一个实现换模型属于 `SEEN_IMPL_NEW_OPERANDS`；训练中完全没有出现过的实现才属于 `NEW_IMPL`。

如果 held-out 全部为 negative，只报告升级率、误升级率、abstention 和成本；不报告 recall 或 AUROC。

### 3. catch-and-fix

如果前瞻实验升级了候选，需要完成：32 步确认、定位偏差出现的层、可执行 repair、再次测 persistence、loss/参数后果和运行速度。

现有 Phi 随机舍入结果只是 stateless SGD 的 development demonstration，不能冒充 AdamW `A=1.029` 的修复实验。

## 最终允许的结论

- 跨未见实现仍有效：可以升级为更通用的训练算子 Oracle；
- 只在特定 optimizer 或实现范围有效：明确声明适用域；
- 能解释现象但短筛不能稳定泛化：保留为分析方法，不继续无限寻找 positive。

