# Bias Oracle paired record model v0.2

## Why the record must have two levels

一次 matched-state probe 先产生两个原始结果：reference arm 和 candidate arm。U1、T1a、T1b
不是任何单独 arm 的属性；它们只有在两条 arm 按同一 state、minibatch、RNG coupling 配对后才有
定义。

因此记录分为：

1. `arm_record`：保存每条实现实际执行了什么、原始 update/T loss/NLL/event/post-state 是什么；
2. `paired_effect_record`：链接恰好一条 reference 与一条 candidate，保存 candidate-minus-reference
   effects。

把 paired effect 复制到两条 arm records 会造成两类错误：同一证据被计数两次；以及先估计两边
variance 再相加，丢失 matched execution 中的 covariance。Oracle 的 N 必须来自同一 state 多次
执行得到的 paired effects，而不是两个 arm variance 的简单 pooled sum。

## Identity and validity

一条 paired record 有效至少要求：

- 两条 arm 的 query/trajectory/phase/state/repeat identity 完全相同，只有 `arm` 不同；
- pre model、buffer、optimizer、scheduler、scaler、RNG、minibatch digests 完全相同；
- realization identity 分开保存，compiled graph family 不能冒充 source operator identity；
- T1a/T1b effect 必须能由 candidate arm raw scalar 减 reference arm raw scalar重算；
- U1/U2 必须链接到同一对 update artifacts；
- 每个 arm record 和 paired record 都有 canonical content digest；
- duplicate arm/pair keys、断链、artifact hash mismatch 或 endpoint 偷换都 fail closed。

## Relation to B/H/N/U

population estimator 只读取 `paired_effect_record` 中某一个命名 endpoint：

- 同一 state 的 repeat effects 给 N；
- states/phases/trajectories 间 effect 变化给 H；
- 对目标 trajectory distribution 的等权聚合给 B；
- independent trajectory 数量决定 U。

这只是记录结构，不会自动让 selected states 变成目标总体样本。legacy A/B/C record 即使通过
schema validator，仍必须标记 `population_eligible=false`。
