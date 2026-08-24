# v4 还剩什么

这份清单只列当前确实没有完成、并且不能用相近结果代替的项目。

## 已完成的部分

- 15 行回溯数据已经统一计算 direct、feedback、actual 三种变化，并做了带符号贡献分解。
- 三组多重比较已经完成：3 行预先声明组、12 行结果盲抽样组、全部 15 行的敏感性分析。
- 四个案例已经做了同一状态下的 AdamW、重置 moment 和无状态 SGD 对照。
- Qwen 已有真实早期、中期、后期状态下的响应测量。
- 新冻结的 Gemma 实现检查已完成 3 行：0 个 direct positive、1 个 feedback control、2 个没有可测载体作用的行。
- 两个保存了完整向量的历史回放已经补算 16/32 步随机符号对照以及可得到的误差幅度指标。

## 仍然是 ABSTAIN 的部分

1. **Phi 0543 的重新播放**：旧的运行包装已经不存在，重新运行的包装顺序也不同，不能把旧摘要当作新原始数据。
2. **完整误差标准对比**：三个新的 Gemma 行已经保存输出、梯度以及同状态 candidate/repair 更新对，并完成了三层 ULP、`rtol/atol` 和保存向量上的 16/32 步方向性摘要；但历史回放仍没有原始位级 operand，因此全池完整对比仍未完成。
   采集器现在支持在同一冻结运行版本中保存 candidate/reference 输出对和更新对；本轮旧运行版本无法通过结构身份检查，因此没有把失败重放当作数据。
3. **前瞻 catch-and-fix**：新的未见实现中没有 direct-persistence positive，因此本轮已明确标记为“不适用”，没有合法的“先发现、再修复、再复测”对象。
4. **通用 recall/AUROC**：前瞻池没有 positive，按协议不计算这两个数。

这些不是被跳过的成功结果，而是当前数据条件下明确的未决项；catch-and-fix 则是由“没有前瞻正例”触发的不适用结果。v4 的有效结论必须限定为：

> 在 cold-start AdamW 约定下，短程 direct-persistence screen 可以作为 fail-closed 分诊；它不能被写成通用安全分类器，也不能据此断言 AdamW 是误差根因。

如果继续补实验，顺序应是：先建立保存原始 operand 和 bit pattern 的新 held-out 运行，再做完整 tolerance 对比；只有出现合格 direct positive，才启动前瞻 catch-and-fix。
