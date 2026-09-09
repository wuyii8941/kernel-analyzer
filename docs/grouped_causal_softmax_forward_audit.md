# 前向 grouped causal softmax 接入审核

本记录针对未接入家族，不是 bias 或训练后果结论。

已阅读 `qwen_seq128_r1/trace/model__0_forward_segment0_executed/output_code.py`
中以 `prepare_softmax_online_scalar_tensor_view_where_7` 结尾的完整函数。
该函数的四个输出不能用现有 backward 参考替代：

- `in_out_ptr0`：原 BF16 attention scores 乘 FP32 scale 后加入遮罩，再写回 BF16。
- `out_ptr0`：写回舍入之前的 FP32 行最大值。
- `out_ptr1`：同一 FP32 分数减最大值后，指数的行和。
- `out_ptr2`：归一化后的 BF16 概率。

遮罩同时要求 key 位置不晚于 query，且两者 group ID 相同。该版本使用
BF16 最小有限值作为遮罩加数，而不是负无穷。不能无说明替换这项语义，
也不能从已写回的 BF16 分数重新计算两个 FP32 统计量。

新增 `grouped_causal_softmax_reference.evaluate` 独立计算全部四项，不修改输入。
七项 CPU 测试检查分组与因果遮罩、多个 head、统计量在写回舍入前计算、
表示与布局检查，以及非有限/不可表示 scale 拒绝。参考目前限定有限输入和
有限中间分数；这些域检查不能被当成对原实现域外行为的结论。

后续已经补齐完整源码匹配、运行前输入绑定和 `in_out_ptr0` 的局部替换采集。
冻结计划包含 28 个参数可达位置；首批 4 个和固定顺序队列中的 24 个均完成
32-state local、gradient 与实际参数写入测量，并由共享核验程序复算通过。
三阶段 effect 均为逐坐标零，最大 confirmation update RMS 为 0。这是该固定
集合和该输出边界上的阴性测量，不证明其余三个输出、其他源码版本或随机训练
状态总体等价，也不是训练质量结论。

完整结果见 `qwen128_grouped_causal_softmax_summary_full_v1.json`，覆盖清单合并
见 `coverage_with_grouped_softmax_embedding_v2.json`。第一次完整运行因 BF16
遮罩哨兵使旧 float32 CountSketch 出现非有限值而失败，原记录保留；修正为
float64 摘要后使用新目录重跑，没有覆盖失败数据。

`unbound_kernel_groups_v1.json` 将此同名符号汇总为 880 个位置，跨多个 release。
这是待核对清单，不是 880 种实现；相同符号不保证完整源码相同，其他 release
必须逐一经过源码审核后才能复用参考。选择该项没有读取 bias 数值结果；当前
28 个完成位置不能外推为其余 852 个位置已经支持。
