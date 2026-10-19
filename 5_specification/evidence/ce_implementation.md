# CE 实现复核（2026-10-18）

对象为 `adapters/CE.py`、phase1 的 `spec_cross_entropy.py` 及共同区间转换。
范围限于合同的四种 profile、N>=1/C>=2 的有限有理数二维 logits，以及 loss 和完整 VJP。
不自动支持一维/高维输入映射、负类权重、概率目标 smoothing 或旧式归约覆盖参数。

## 来源与任务选择

重新读取保存的 PyTorch 2.10 `sources/PCE.html`：index 目标的 mean 分母是未忽略目标类的权重和；
probability 目标的 mean 分母是样本数 N；ignore_index 只适用于 index 目标。
文档允许一维和高维 logits，但当前合同尚未验证对应映射，返回 unsupported。
原文只用文字描述 label smoothing 与均匀分布混合；有类权重时采用 R_A 是显式任务选择，
不能从程序输出更接近哪种读法来决定真值。R_B 仅保留为诊断入口。
底层历史模块注释仍引用当时的 main 文档；本次按保存的 2.10 公式核对实际调用路径，
不把历史模块名称或旧版本引用当作符合当前合同的证据。

## 公式到实现

令 p_c=exp(x_c)/sum_j exp(x_j)，a_c 为选定任务的目标系数，A=sum_c a_c。
每行 loss=-sum_c a_c*log(p_c)，对 x_c 的导数为 A*p_c-a_c。

- index_unsmoothed：a_y=w_y，其余零；忽略行的 loss、梯度与分母贡献全零。
- index_smoothing_unweighted：a_c=(1-eps)*onehot(y)_c+eps/C。
- index_smoothing_R_A：a_c=w_c*((1-eps)*onehot(y)_c+eps/C)。
- probability_unsmoothed：a_c=w_c*target_c；要求有效概率行，不忽略整行。

index mean 用 D=sum_nonignored w_y，probability mean 用 D=N；none/sum 不归一化。
分母不依赖 logits，所以梯度也除以 D。适配器最后逐行乘显式 upstream，loss 本身不乘 upstream。
每个输入先验证；均值分母为零在超越函数求值之前返回 domain_undefined，
不能因为求值预算不同改变这一数学分类。sum 和 probability mean 的零权重结果仍有定义。

## 数学值的保留与结果比较

底层把有理数向外转换成 Decimal 区间。对一个固定移位 m，计算
log(p_c)=(x_c-m)-log(sum_j exp(x_j-m))；m 即使来自输入区间端点，恒等式仍成立。
四则运算使用有方向的舍入，exp/log 在正确舍入结果外扩一个单位；
exp 下溢时以 [0, 最小正常数] 包围。不能可靠计算的范围明确拒绝。
输出 Decimal 端点精确转为 Fraction，再向外转换；VJP 乘法继续保留包围。
这些说明依赖所用数值库的精确与定向舍入行为，不声称有限测试是其全域形式证明。

完整输出比较器要求 loss 和 gradient 的全部结构一致。精确点在区间外可构成矛盾；
落在非退化区间内只能表示尚不能确定相等。不会因 loss 相容而忽略错误梯度，
也不会把经验容差用于严格认证。

## 独立证据与剩余使用条件

`tests/test_ce_independent.py` 用带严格几何余项界的有理数对数级数验证均匀 logits 的闭式 loss，
VJP 则有精确有理数答案；覆盖四种 profile、全部归约、忽略行、不同类权重和带符号 upstream。
另一组使用 Arb 二分类 logistic 公式独立复算非均匀 logits 的 loss 与梯度，
验证较窄的独立包围完整落在参照区间内。两组均不读取被审 kernel 的结果来构造预期。
`tests/test_enclosed_output_membership.py` 检查完整结构、区间相容不等于相等、
精确零结果及错误梯度。检查器保留来源、预期和实现损坏的注入检查。

据此将 implementation_status 记为 validated，限于上述求值及比较范围。
内容审阅和实际 kernel 绑定不由这些测试升级。R_A 的显式任务选择仍必需；
实际调用还须核对目标类型、权重、ignore_index、归约、upstream、输出布局和完整写回。
不自动更新运行时检查记录，不宣称真实 kernel 覆盖认证已经完成。

## 当前文件独立复核留存（2026-10-18）

Codex 重新读取保存的 PyTorch 2.10 CrossEntropyLoss 页面、当前合同、适配器和底层 Decimal 区间实现，
逐项核对 index/probability loss、两类 mean 分母、ignore_index、none 输出及显式 upstream VJP。
复核中修正了一个范围分类边界：二维 C=0 是 invalid_input；含零长度空间轴的合法高维形状仍属于
当前二维 profile 未支持的输入。该修改不改变四个 profile 的公式。

独立有理对数余项和 Arb 二分类公式测试共16项通过；合同/运行时/任务选择回归79项通过；
合同检查的12个样例全部相容，8个错误变体全部被拒绝。当前 `contracts/CE.yaml` 与
`adapters/CE.py` 因此按本文件声明的范围保留到 `validated/`。这项复核不审阅任何真实调用绑定，
也不把有限测试解释为全部实数输入上的形式证明。
