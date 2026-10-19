# NORM 来源与实现复核（2026-10-18）

复核对象是 `contracts/NORM.yaml`、`adapters/NORM.py` 和实际求值路径
`kernel_spec/c_norm.py`。合同原作者为 Claude；本次由 Codex 重新读取保存的来源、检查公式与输入域，
并以另一种求导方法核对实现。真实 kernel 的调用绑定不在本次准入范围内。

## 任务语义

保存的 PyTorch 2.10 LayerNorm、RMSNorm 和 GroupNorm 文档分别给出前向公式、归约维度、biased
variance 和 affine 参数布局。保存的同版本 `_refs` 源码给出 native LayerNorm/GroupNorm 返回的 mean
与 rstd 及 GroupNorm 的连续通道分组；`_decomp` 源码给出 LayerNorm 和 RMSNorm 的反向公式。
这两份保存源码与当前安装的 PyTorch 2.10.0+cu128 对应文件逐字相同。

Liger 0.7.0 的 LayerNorm、RMSNorm 和 GroupNorm 源码另外支持保存统计量、三种计算精度路径、
RMSNorm 的 `offset + weight` 形式及三个反向公式。Transformers 4.57.3 的 Gemma 和 Qwen3-Next
源码都明确使用 `1 + weight`。因此 offset 形式保留为需要任务明确选择的单独 profile；实现中的
float32 转换仍属于被测程序的数值步骤，不写进实数目标。

`torch` profile 表示带完整 saved statistics 的 native/full 任务；`torch_autograd` 表示公开前向与
反向边界，只返回 y 和 VJP。公开 `torch.nn.functional.group_norm` 还会在调用 native 运算前执行
batch-size 检查。PyTorch 2.10 实测和保存源码都表明 `[N=1,C=1]` 会被公开接口拒绝，而
`aten.native_group_norm` 有定义；合同现已分别处理这两个边界。

文档公式与公开包装器都没有规定 `eps >= 0`。PyTorch 2.10 的 LayerNorm、RMSNorm 和 GroupNorm
也都会在根号内结果仍为正时接受负 eps。合同原先把负 eps 一律列为非法，这会拒绝合法任务；现改为
逐个归约集合要求 variance 或 second moment 加 eps 为正。零或负的根号内结果属于实数任务无定义。

## 参照实现核对

`c_norm.py` 先严格检查张量形状、参数形状和分组，再用 Fraction 计算 mean、biased variance 或
second moment。平方根及其倒数通过共同的严格实数模块求值。LayerNorm、RMSNorm、offset RMSNorm
和 GroupNorm 返回完整 y、声明的 saved statistics，以及对 x、weight、bias 中实际存在参数的完整
VJP。公开 profile 只过滤内部 saved 字段，不丢梯度。

`tests/test_norm_independent.py` 只按四种前向公式实现 Fraction dual-number 自动微分，没有调用
`c_norm.py` 的闭式反向公式。所选输入使平方根为有理数，从前向损失直接得到每个 x、weight 和 bias
坐标的精确导数；四种任务的完整输出、统计量和梯度逐项等于当前参照。GroupNorm 用独立坐标映射核对
连续通道组及逐通道参数。

`tests/test_norm_domains.py` 覆盖负 eps 的合法与无定义两侧，以及 public/native GroupNorm 的输入域
差异。合同检查器当前得到 22/22 个例子相容、14/14 个错误公式被拒绝。严格包围、完整字段比较和
异常分类仍由共同运行时检查；这些测试支持本文件声明的实现范围，不把有限样本称为全域证明。

## 范围限制

RMSNorm 的 `eps=None` 必须由具体调用依据实际输入 dtype 解析成显式值，参照不会猜默认值。
以普通嵌套列表传递、且首个维度为零的张量不能保留后续空维度；调用绑定没有保留完整 shape 时应
拒绝，不能据 `[]` 推断任务形状。Liger 0.6.2 的真实候选与这里保存的 0.7.0 来源版本不同，现有
候选计划继续保留该未解决项。任何真实调用仍须单独核对全部输入、公开输出、梯度、状态和实际
Triton 启动。
