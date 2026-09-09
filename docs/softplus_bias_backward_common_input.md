# Softplus 反向偏置归约：新家族接入

这是扩展覆盖的实现记录，不是偏差阳性或训练后果结论。此处 bias 指模型的偏置参数；不能仅因名称包含 bias 就说存在数值偏差。

## 数学对象

令 `z[t,c] = x[t,c] + b[c]`。对 beta=1、threshold=20 的 softplus，偏置梯度为

`sum_t (g1[c,t] + g2[c,t]) * where(z[t,c] > 20, 1, sigmoid(z[t,c]))`。

新参考读取实际 kernel 的同一组输入，不重新计算上游状态。它用 FP32 sigmoid 计算导数、求和后写入 BF16。原实现使用 exp 与除法表达导数；两者实数语义相同，浮点执行不同。该推导证明比较对象的语义，不证明差异具有非零均值。

## 自动接入

- `softplus_bias_backward_reference.py` 审核完整计算体、指针类型、寻址、阈值和维度；不按 kernel 名称猜测语义。
- 家族注册到现有 `source_reference_registry.py`；继续复用源码绑定、三阶段采集与统计代码。
- `scan_registered_reference_pool.py` 对冻结清单中的全部源码扫描所有注册家族，检查原源码摘要，不读取数值结果。源码匹配仍不等于完成运行测量。
- 已从 Mamba seq64 的现有源码匹配两种定义，并自动绑定 24 个训练位置。按原清单顺序分为六批，每批四个位置；不按效应量筛选。

首批启动记录位于 `results/property/numerical_coverage_v1/mamba64_softplus_bias_batch000_launch_v1/`。是否完成必须另外检查进程退出、实际实现身份与结果有效性，不能由本文推断。

同一参考进一步自动绑定 seq128 和 seq256，各 23 个位置；三种长度共 70 个位置，不等于 70 个独立实现家族。三种长度都按最多四个位置分批，使用同一采集入口与统计代码。各自的 `*_remaining_launch_v1` 队列仅在对应首批退出成功、`finalize_numerical_family` 核验通过后启动。源码匹配失败或没有训练参数绑定的位置保留在原计划的 unresolved 中，不替换成容易执行的位置。

该版扫描结果为 `results/property/numerical_coverage_v1/registered_reference_pool_v1/coverage.json`：42 个去重源码文件、957 条定义记录，467 条有参考模板，490 条仍缺参考。加入普通行归约后的 `source_coverage_with_row_sum_v1.json` 为 473 条有参考、484 条仍缺参考。这一分母只覆盖已保存 Triton 定义，不包含全部外部库调用，也不表示这些定义已完成训练测量。早期本文的 45 个源码文件表述已按自动去重复算纠正。

## 范围限制

目前审核的存储类型为 BF16 输入和偏置、FP32 上游梯度、BF16 输出；未匹配的实现仍不支持。带两个输出的定义只分析 `out_ptr0` 偏置梯度，不声称覆盖 `out_ptr1`。参数范围、模型执行路径和状态范围以运行协议为准，不把这一家族接入写成全部 kernel 支持。

独立 float64 autograd 测试用于核验数学参考；源码变异测试和输入检查用于防止错误接入。真实 Triton 数值差异、三阶段结果及训练影响需要运行证据，不能由这些单元测试代替。
