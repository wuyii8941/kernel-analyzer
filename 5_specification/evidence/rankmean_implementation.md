# RANKMEAN 实现复核（2026-10-18）

本次复核对象为 `adapters/RANKMEAN.py`、其调用的
`specification_v2/algorithm_specs/phase2/spec_training_program.py::cross_rank_mean`，
以及 `kernel_spec/num.py` 的有理数、整数和向量输入检查。
结论只涉及显式选择的 token 等权目标，不认证任何实际分布式训练程序。

## 目标与来源

合同 RM.TASK 要求调用者明确选择有效 token 等权。
保存的 `sources/HF.html`（Hugging Face 的 Fixing Gradient Accumulation）支持
“累计 loss sum 除以有效 token 总数”这一训练目标，不能据此否定 sequence 等权目标。
此次读取的是保存的原文，未声称重新联网核对。

若第 r 个参与者持有 token 集合 T_r，且这些集合不重不漏，
S_r = sum_{i in T_r} loss_i，n_r = |T_r|，则有限和的结合律给出
sum_r S_r = sum_i loss_i、sum_r n_r = |T|。
因此 RM.C1 的结果等于逐 token 平均。集合不重不漏及 S_r 的实际来源是绑定前提，
参照不能从几组数值独自证明它们。

## 实现对应

输入检查先要求 S、n、participant_ids 齐全且长度一致非空；每个 S 是有限有理数，
每个 n 是非负整数，参与者不能重复，n_r=0 时必须有 S_r=0。
输入数值保持精确：有限浮点按收到的二进制有理数解释；分数计数不截断。
不支持的参与者标识表示明确返回 unsupported，不作为数学矛盾。
输入中出现 weights、reduction 等额外选项时也返回 unsupported，不能忽略后继续按 token 等权求值；
参照与完整输出比较共用这项检查，回归反例覆盖额外权重、求和归约和额外参与者数量声明。

在 N=sum(n)>0 时，cross_rank_mean 对所有 S 作 Fraction 求和，再除以整数 N；
没有 zip 截断、浮点中转或遗漏参与者。输出仅有合同承诺的 loss。
N=0 在除法前返回 domain_undefined。完整输出比较器要求恰有 loss 字段，
并检查 candidate*N=sum(S)。由于 N>0，该等式与 candidate 等于任务值等价。
这是精确数学值的比较，不是对设备舍入误差的容差判断。

## 独立证据与界限

`tests/test_rankmean_independent.py` 从四个有理数 token 出发，用在线均值递推独立求答案，
遍历其分配到三个参与者的全部方式并改变参与者顺序；包含空参与者。
另检查大整数计数、全空、非法空贡献、重复参与者、缺失字段和不支持的标识表示。
`tests/test_training_output_membership.py` 检查完整输出及 1/3 与近似浮点的区别。
合同开发检查保留了平均局部均值、按参与者数归一化等误读的反例。
测试支持所列情形；上面的逐步对应说明才解释为何实现计算了所声明的有理数公式。

独立留存复核再次逐项对照保存的任务说明、逐 token 在线均值复算、输入拒绝及五个错误实现，
未发现未隔离矛盾。当前合同和适配器的可读原文件因此保留在 `validated/`；后续任一文件变化
都会令准入失效。

implementation_status=validated 及文件留存的范围仅限上述参照求值与完整输出比较。
不把 runtime_binding_status 从 pending 提升。每次实际使用仍需明确选择 token 等权任务，
并绑定参与者、token 来源和完整输出。
