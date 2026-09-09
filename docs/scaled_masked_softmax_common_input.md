# 缩放与 mask 融合的 softmax backward

这是新增参考的接入准备，不是已经确认的bias案例。

Phi的已保存实现先把分数乘以`s`、加上mask，再用保存的行最大值和分母恢复概率：

\[
p=\exp(s\,scores+mask-maximum)/denominator,
\qquad output=s\,[g p-p\sum(g p)].
\]

mask按源码中的`row % width`选择；输入分数和mask、行最大值、分母都来自
同一次调用前的真实数据，不重新运行另一个forward取得参考状态。
当保存的归一化量与这些分数一致时，上式对应缩放后softmax的梯度；
源码检查本身不证明这些统计量在所有训练状态中一致，更不证明舍入误差均值非零。

`scaled_masked_softmax_reference.py`检查完整函数体、两个缩放位置、mask索引、
归约与输出地址。运行参考要求明确布局、输出不与只读输入别名、FP32保存的
归一化量和正分母。独立autograd检查使用因果mask；改变索引、运算或去掉输出
缩放的源码会被拒绝。当前8项CPU测试通过。

Phi的64、128、256长度各有一个源码定义匹配该模板。这不是三个独立机制，
也不是三个完成的训练实验。扩展扫描复用已有源码发现程序：
`scripts/scan_selected_silu_product.py --family SCALED_MASKED_SOFTMAX`。
本扩展尚未加入运行中的注册表，以免改变Mamba队列已冻结的依赖。
下一步仍需绑定参数、实际执行、三阶段采集和统一复算；不预设数值结果。
