# 模型覆盖审计

这份审计把“测过哪些模型”和“每个模型做到哪一层”分开。模型出现过，不等于它已经完成全算子因果链。

## 数量

按实际结果文件中的独立模型 checkpoint/模型家族计数：**10 个**。

- **4 个系统性普查模型**：Qwen3-1.7B、Phi-4-mini、DeepSeek-R1-Qwen3-8B、Mamba-130M。
- **6 个定向或 held-out 模型**：Qwen3-VL-Reranker-2B、Gemma 3 4B、OLMoE-1B-7B、Llama 3.2 3B、Ministral 3 3B、Gemma 4 E2B。

Liger-Qwen 是 Qwen3-1.7B 的另一种实现，不算新模型。FlashAttention 是外部论文锚点，也不算本项目运行的模型。Granite 目前暂停，DeepSeek-V4 Flash 只有计划，没有实际结果。

## 4 个系统性模型做到了什么

四个模型各有 seq64、seq128、seq256，共 12 个模型–序列长度单元。执行清单、前后向数学来源、候选绑定、typed Triton/non-Triton 参考、same-dtype 对照和 T1 全坐标审计均已通过；T1 全坐标结果为 1562/1562，12/12 单元的声明覆盖门均已关闭。

这仍不等于所有后续 bias 研究都完成：Mamba 的一条独立 wall-time consequence 重跑仍被 AOT 编译阻塞，且不改变已有科学 consequence 结果。另一个边界是：这 12 个单元证明了覆盖和参考链路，不自动等于每个单元都产生 Flash-style 持久 bias。

因此准确说法是：**四个主模型的全算子覆盖门已到位；Mamba 的新增 timing 记录仍未到位，持久性结论仍按实际 case artifact 单独报告。**

## 6 个定向模型做到了什么

- Qwen3-VL：只做了 SiLU 和语义区域等聚焦实验，不是全算子普查。
- Gemma 3：文本和图文筛查及若干视觉控制完成，没有得到新的源持久正例。
- OLMoE：文本筛查和路由/专家控制完成，结果是抵消性控制。
- Llama 3.2、Ministral 3：做了 held-out lm-head dX 和筛查；这是同一实现族在新 operand 分布上的验证，不是新的实现类。
- Gemma 4：完成一个新实现类的定向审计，发现的是 Adam 状态维持的反馈案例，不是 Flash-style 源持久案例。

机器可读版本见 [`model_coverage_audit_v1.json`](../results/coverage/model_coverage_audit_v1.json)。
