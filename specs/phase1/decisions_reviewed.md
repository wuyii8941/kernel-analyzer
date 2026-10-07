# 第一阶段规格歧义审阅决定

日期：2026-10-07  
审阅对象：`independent_specs_phase1_v0.3.1.tar.gz`  
状态：供本轮实验采用的审阅决定，不替代上游维护者对未明确契约的解释。

## 1. 决定原则

本轮是在评价现有API与候选实现，不是在重新设计一个更优的损失函数或数值API。
- 官方明确规定的条款：按锁定版本的条款。
- 从公开公式和参数说明组合出的解释：可以冻结为本轮主解释，但明确记录依据强度。
- 文档未规定或相互存在张力的行为：采用一个项目profile便于运行，同时保留读法分歧；不能仅凭未满足自选profile计入“上游已确认bug”。
- 实数目标无定义：记录实现约定，不把本轮选择的NaN/零输出作为普遍的数学正确性标准。

## 2. 十二条决定

| 编号 | 本轮主决定 | 评分边界 / 其他读法 |
|---|---|---|
| CE-A1 | 分子采用逐类别权重：R_A，即 `-sum_c w_c*q_nc*log(p_nc)`；忽略的是目标对应的整行，不删除类别列。 | 由概率目标公式与平滑混合定义组合而来。官方未单列所有组合公式，不将本轮解释冒充唯一全域定理；R_B作为解释对照保留。 |
| CE-A2 | 类别下标目标的mean采用D2：`sum_{n not ignored} w[y_n]`。即现有函数的`reading='R_A'`，不要切换成R_C。 | R_C是另一种合理的加权目标规范化，不是D2在数学上唯一的一致延伸。不同于主解释先记规范差异；无进一步契约依据时不自动宣布共有bug。 |
| CE-A3 | 有效分母为0时，有限实数目标无定义；loss及backward输出按约定记录，跳过普通导数性质。 | 全部忽略且reduction=none/sum时仍为零；不能把这两种模式一起列为无定义。平滑时D2为0可能是正数/0，不都是0/0。 |
| CE-A5 | 概率目标mean按D4除以N，不改为权重和。 | 概率输入应满足声明合法域；ignore_index不作为概率目标的行过滤机制。 |
| CE-A6 | 发布标量与实际接收标量两种口径分别报告，主API执行比较使用实际接收值。 | 接口项独立列出；不能把接口正常舍入记成kernel任务错误。 |
| POOL-A1 | 主分析采用R2：窗口和显式padding域的交集大小作为count_include_pad=True的默认除数；False只数实际输入；显式divisor_override优先。 | 固定kernel大小R1保留为尾部读法对照。2.10的AvgPool3d文案仍称默认除数为kernel_size，与“pooling region”的解释存在张力；只因过界分母不同，不自动提交已确认bug。 |
| POOL-A2 | 对有限、有实际最大者的窗口，前向下标必须属于并列最大集合。不强制first或last。 | 严格普通导数评分先限唯一最大者。并列处可检查合法次梯度集合；若任务另外声明“反向使用返回下标”，才逐次检查该选路关系。单成员与平均拆分不互判错误。 |
| POOL-A3 | R_prop作为项目NaN诊断profile；只传播真正参与当前窗口的NaN。 | 文档没有明确规定的NaN行为不进入主bug正确率；R_ignore仍可单列，不因“不传播”自动报错。全NaN与几何空窗口区别记录。 |
| POOL-A5 | 几何采样窗口没有实际输入时，依据负无穷padding取输出-inf、实际输入argmax集合为空。 | 不要求返回索引必须属于空集合；没有实际输入梯度路由。NaN剔除导致的空集合不是同一几何论证，另行声明。 |
| IDX-A3 | R_prop作为项目NaN诊断profile，只作用于实际参与归约的值。 | include_self=False时被更新目标的旧self不参与；未被更新位置保持不变。未明确的NaN策略只记差异，不自动计上游bug。 |
| IDX-A4 | 保留并列路由差异，不按唯一向量裁决；主精确导数评分限唯一极值。 | 不只检查总量：支持集须在极值集合内；上游为1时系数非负且和为1。若API有更具体条款，采用更具体条款。 |
| ACC-A1 | 整个累加窗口无有效token：token平均目标无定义，记录skip/zero/NaN约定，不据此判数学bug。 | 个别microbatch为空、窗口总token数>0时目标仍定义，空microbatch贡献0；全局和/全局计数，不能平均microbatch均值。 |

## 3. CE-A1/A2 为什么不采用 R_C 作为主评分标准

令 `q_nc=(1-eps)*1[c=y_n]+eps/C`，`A=-sum_{n in I,c} w_c*q_nc*log(p_nc)`。
两个分母为：

`D2 = sum_{n in I} w[y_n]`；`DC = sum_{n in I,c} w_c*q_nc`。

`A/D2`和`A/DC`在分母非零时都是明确、可微的目标，没有“前者代数不自洽”的结论。
文档明确按目标输入类型区分mean的分母：类别下标采用目标类别权重和，概率输入采用N。
因此不能仅以“更统一”为由把DC提升为现有API的唯一正确值。

手算：logits=(0,0)，目标0，权重(1,3)，eps=1/2；q=(3/4,1/4)，A=(3/2)*ln2。
D2=1，DC=3/2，所以R_A=(3/2)*ln2，R_C=ln2。这是更换目标，不是提高求值精度。

另一个必须区分的边界：权重(0,1)，同样目标0和eps=1/2，分子是ln2/4>0，D2=0。
所以“总目标权重为0时均为0/0”的注释需改为“分母为0，有限实数目标未定义”。

## 4. POOL-A1 与 POOL-A0

一维记实际输入域I=[0,L)，显式padding域P=[-p,L+p)，窗口采样集合W_o。
主分析除数为`|W_o ∩ P|`（count_include_pad=True），或`|W_o ∩ I|`（False）。
只在完整窗口位于P内时，前者才必然等于kernel_size。

例：输入[1,2,3,4]，k=3，stride=2，padding=0，ceil_mode=True。
第二窗口覆盖索引2,3,4；索引4既不在输入内，也没有声明为padding。
R2输出7/2，R1输出7/3。此例能区分读法；原[1,...,7] -> [2,4,6]例子不能。

AvgPool3d并非没有独立官方文档。2.10文档给出了三维公式、padding、ceil窗口起点及逐维修正，
因此基础三维规则可从该文档直接建立。但是其divisor_override说明写的是默认kernel_size，
不能声称它已经无歧义证明了全部尾部除数解释。保留这个特定冲突即可，不必把全部3d标为假设。

## 5. 并列梯度和NaN

有限输入的最大者集合为A、上游梯度为v。集合式检查可采用
`g_i=v*alpha_i`，其中alpha在A外为0，在A内非负，且总和为1。
这是必要的数学合法性检查，不等于证明实现符合某个更强的API选路约定。
例如上游1时(1,0)与(1/2,1/2)可合法，(2,-1)总和也是1却不合法。
不同API确有不同并列梯度语义，不能从名字“max”统一强制一种。

NaN既不是普通实数，也没有一个支配所有max API的统一语义。
R_prop可以作为主动暴露异常的项目策略，但不能据此宣布所有R_ignore实现错误。
NaN影响应跟随实际参与集合，不能因为输入张量任何位置有NaN就让所有输出为NaN。

### 尚需处理的一个有限边界

本包`scatter_reduce([0],0,[0],[NaN],'amax',include_self=False,nan_reading='R_ignore')`
仍抛出`ValueError: max() iterable argument is empty`；相应max_pool读法返回-inf。
如果保留R_ignore辅助测试，必须记录“全NaN过滤后无数值”策略，或主动返回规格未建立。
这不阻塞首轮有限输入、唯一极值的主测试，但不能把异常记为候选失败。

## 6. 本轮实际做了什么

- 阅读决策表、歧义表、规格实现、协议、独立核对程序。
- 重跑`test_specs.py`：87/87通过。
- 原样重跑`independent_check.py 1000`：CE 1000条件、9965标量全在区间内、44无定义；
  pooling 3000条件、1815合法一致、1185两侧拒绝；index 3000条件、960合法一致、2040拒绝。
- 新增确定性读法例子，见`decision_examples.py`和`results/decision_examples.json`。
- 未读PyTorch算术kernel源码来定义规格；未运行GPU或nightly；未修改原包或仓库。
- 重跑核对仍是原有两种算术实现之间的抽样复现，不是新建了第三方严格区间证明；
  R_A、R_B、R_C都能算对其各自定义，不能靠核对通过决定哪一份定义是API契约。

## 7. 冻结与评分建议

可按这份决定冻结第一轮有限输入核心切片。
CE的R_C、未明确的NaN约定、并列路由、AvgPool3d尾部除数的文档张力各自保留诊断标签。
主结果分列：文档明确条款的违反、基于声明解释的差异、未定义情形的实现约定。
不能将后三者混入“所有实现共有的已确认bug”。

## 公开依据（访问日期2026-10-07）

[D1] PyTorch 2.10 CrossEntropyLoss：
https://docs.pytorch.org/docs/2.10/generated/torch.nn.CrossEntropyLoss.html

[D2] PyTorch main CrossEntropyLoss（补充比对，不能替代锁定版本）：
https://docs.pytorch.org/docs/main/generated/torch.nn.CrossEntropyLoss.html

[D3] PyTorch 2.10 AvgPool2d：
https://docs.pytorch.org/docs/2.10/generated/torch.nn.AvgPool2d.html

[D4] PyTorch 2.10 AvgPool3d：
https://docs.pytorch.org/docs/2.10/generated/torch.nn.AvgPool3d.html

[D5] PyTorch 2.10 MaxPool2d：
https://docs.pytorch.org/docs/2.10/generated/torch.nn.MaxPool2d.html

[D6] PyTorch 2.10 autograd不可微/未定义点约定：
https://docs.pytorch.org/docs/2.10/notes/autograd.html#gradients-for-non-differentiable-functions

[D7] PyTorch main torch.max（明确区分max与amax并列梯度；不是把其条款自动移植给pool/scatter）：
https://docs.pytorch.org/docs/main/generated/torch.max.html

[D8] PyTorch main torch.fmax（NaN不总传播的官方例子）：
https://docs.pytorch.org/docs/main/generated/torch.fmax.html

[D9] PyTorch main scatter_reduce_（参与集合与include_self）：
https://docs.pytorch.org/docs/main/generated/torch.Tensor.scatter_reduce_.html
