# SOFTMAX 实现核验（2026-10-18）

对象是 `contracts/SOFTMAX.yaml` 的 `real` profile、`adapters/SOFTMAX.py`，以及
`kernel_spec/aten.py`、`real.py`、`ieee.py`、`aten_adapter.py`、`compare.py` 中实际调用的路径。
这是参照实现相对于该合同的核验；合同仍为 drafted，真实 kernel 绑定仍为 pending。

## 来源及目标

重新对照仓库保存的 PyTorch 2.10 文档 `T_SOFTMAX`、`T_LOG_SOFTMAX`，以及
`TS_DECOMP.py.gz` 的前后向定义、`TS_ATTN_CPP.cpp.gz` 的 `_safe_softmax`、
`TS_INDUCTOR_PRIMS.py.gz` 的 `eager_prepare_softmax`。来源条目见 `SOURCES.json`。
本次没有获取新网页；localhost:27891 代理未能连接，未绕过代理补取来源。

目标是合同声明的实数运算，不是复制原生 kernel 的每次舍入。
显式任务转换单独执行；原生运算内部的计算精度和最终舍入不作为实数目标的一部分。
`_softmax` / `_log_softmax` 的 ATen 调用必须给出 half_to_float，不能继承数学辅助函数的默认值。
true 要求 f16 输入并选择 f32 输出。`_safe_softmax` 的 dtype 按来源要求先转换输入；
全 mask 判断仍使用原输入。反向 input_dtype 必须明确，决定结果类型，不将输出舍入加入梯度公式。
这些约定必须与具体任务选择一致，不能自动用于要求原生逐位结果的任务。

## 公式及完整结果

对一行 x，令 m=max(x)，s=sum(exp(x_j-m))。
有限行的 softmax 为 exp(x_i-m)/s，log-softmax 为 (x_i-m)-log(s)。
共同移位在实数中抵消；先完成精确移位也避免把很大的 m 加入区间后再减去造成精度损失。
在线准备返回 (m,s) 两个张量，归约维保留；不是只返回最大值或概率。
所有逻辑坐标都由同一轴分解遍历，标量按单元素行处理；空输出仍保留完整 shape。
在线准备所依赖的最大值在归约轴为空时拒绝，即使其他轴使输出也为空。

令 y 为传入的前向输出、g 为传入的上游梯度。
softmax 的 Jacobian 为 J_ij=delta_ij*y_i-y_i*y_j，因此 J^T g 的第 j 个坐标为
y_j*(g_j-sum_i g_i*y_i)。log-softmax 的 Jacobian 为 delta_ij-exp(y_j)，
故梯度为 g_j-exp(y_j)*sum_i g_i。实现返回全部坐标，不广播不匹配的 g/y。
把这些式子称为某次前向任务的 VJP，还要求绑定证明 y 确实来自相应前向；
参照不会根据 y 的数值相近就推断来源。

有限行中的 -inf 概率为零、log 概率为 -inf。全 -inf 行及含 +inf 行的普通
softmax/log-softmax 在本实数目标下无定义；safe-softmax 的原始全 mask 行按来源返回零。
NaN 输入被拒绝；不根据设备在无定义处返回零还是 NaN 来裁决语义错误。

## 包围保证与裁决

收到的有限值保持为精确有理数。共同实数模块用精确有理数四则运算，或以有方向的
mpmath 区间运算包围 exp/log 及组合结果；端点转换使用精确二进制值，不用显示字符串。
因区间过宽无法判定的分支返回未建立。显式格式转换采用已有 CAST 路径的精确舍入计算，
区间输入的结果保留上下界；跨越不能可靠表示的溢出边界时拒绝。
包围保证以数值库及共同算术模块的上述行为为前提，不声称有限测试证明了数值库全域正确。

完整输出比较要求每个张量提供 t、v、shape；在线准备必须提供两个完整张量。
dtype、形状及全部坐标都参与比较。精确有限值及声明的无穷值可确认相等；
非退化区间包含候选仅表示未建立，区间外的点可以拒绝。
未决坐标不遮蔽其他坐标的矛盾。空张量不能只凭 `[]` 获得确认，因为它没有保留维度。
这里只比较调用者提供的完整任务结果，不认证设备写回、别名、布局或内存访问。

## 独立证据与状态

- `tests/test_softmax_independent.py` 用 512-bit Arb 的二分类 logistic 公式复算概率、
  log 概率和在线质量，检查其包围完整落入参照区间；包含非均匀有理输入和巨大共同偏移。
  log-softmax 反向由独立构造的完整 Jacobian 复算。
- `tests/test_softmax_backward_domains.py` 用精确有理数完整 Jacobian 验证不同轴和
  输出类型下的全部 softmax 梯度；另检查标量、空形状、输入类型及形状拒绝。
- `tests/test_softmax_domains.py`、`test_reduction_domains.py` 验证类型转换顺序、
  原始 mask、无定义行、NaN、轴及空归约约束。
- `tests/test_softmax_output_membership.py`、`test_enclosed_output_membership.py` 验证
  双输出遗漏、空形状及 dtype 错误、无穷值、微小精确差异和区间相容不等于相等。
- `tests/test_aten_adapter_inputs.py` 区分参数绑定错误与执行器内部错误。
  合同的 12 个样例与 7 个误读要求另由现有检查器执行；它们是补充证据，不是独立数学证明数目。

据此将 implementation_status 记为 validated，限于这里描述的实数求值和完整输出比较。
这不升级合同内容审阅，不授予默认任务选择，不更新旧运行时检查记录，也不新增真实 kernel 覆盖数量。
用于语义裁决之前，仍须完成来源/profile 审阅和具体输入、输出、状态、计算区域的完整绑定。

## 当前文件独立复核留存（2026-10-18）

Codex 逐项读取保存的 PyTorch 2.10 softmax、log-softmax 文档，`decompositions.py` 中的
`_softmax`、`_log_softmax` 及两个反向定义，`attention.cpp` 的 `_safe_softmax`，以及
`inductor_prims.py` 的 `prepare_softmax_online`。当前合同中的公式、类型选择、原始输入上的全
mask 判断、两个在线输出和反向公式均与这些材料一致。设备计算中的逐步舍入仍归入数值差异，
不被写进实数任务目标。

复核另外覆盖标量、零长度维度、正负无穷、safe-softmax 的输入转换、整数在线准备、多个归约轴，
以及 softmax 概率和为一、softmax VJP 沿归约轴和为零这两个独立恒等式。12 个合同样例全部相容，
7 个错误变体全部被拒绝；专项独立公式、输入域、反向和完整输出测试均通过。当前
`contracts/SOFTMAX.yaml` 与 `adapters/SOFTMAX.py` 因此按本文件声明范围保留到 `validated/`。
这项复核不审阅任何真实调用绑定，也不把有限测试解释为全部实数输入上的形式证明。

## 外部公开调用发现的缺口：原始输入 VJP（2026-10-19）

attorch 的公开 `Softmax(dim=-1)` 前向与 autograd 调用需要从原始 logits 定义完整梯度。
已有 `_softmax_backward_data` 合同只定义收到的保存输出上的反向运算；把被测输出作为概率输入，
会让该任务的数学参照依赖被测前向误差。因此新增派生操作 `vjp.aten.softmax(g,x,dim)`，复用
现有 `real` profile 和数值模块，不另建合同。仅接纳同形、同浮点 dtype 的有限 x/g，无 dtype 转换。

重新读取保留的 PyTorch 2.10 `T_SOFTMAX` 公开定义和 `TS_DECOMP` 反向源码。
由 p_i=exp(x_i)/sum exp(x) 求导得到 J_ij=p_i*(delta_ij-p_j)，故 dx_j=sum_i g_i*J_ij。
参照先从 x 独立求 p 的严格包围，再计算 p_j*(g_j-sum_i g_i*p_i)，保留整个梯度张量。
这一组合不读取候选输出；外部调用的物理输入绑定仍需单独核查。

`tests/test_softmax_public_vjp.py` 在修复前有7项因未支持而失败；新增路径后，使用512-bit Arb
独立构造完整 Jacobian，核查不同轴及非均匀上游梯度下的包围，并检查精确均匀 logits、公开
PyTorch autograd、形状、有限性、dtype、标量、空输出和遗漏坐标。合同增加4个样例与3个误读，
包括误把 logits 当概率、遗漏耦合求和、忽略归约轴；16个合同样例相容，10个误读均被拒绝。
来源与导数审核、不同算法复算支持此扩展；并非第三方审核或全域形式证明。
