# SmoothL1 与 Huber：由外部调用缺口驱动的参照核验

2026-10-19 固定的外部调用开发组实际运行了 attorch 的 SmoothL1Loss(beta=0.5) 与
HuberLoss(delta=0.5)。两者与 PyTorch 2.10 公开模块的前向及 input/target 梯度相符，但当时
没有准入的独立数学参照。HUBER 合同补这个实际缺口，不把旧实现对照结果改记为数学认证。

## 来源和任务选择

attorch 模块明确继承对应 PyTorch 类，文档和测试以相同公开任务比较。参照选择 PyTorch 2.10
模块语义：无权重、相同形状、reduction 为 none/sum/mean。函数式广播、加权 Huber、弃用覆盖
参数不是这个任务。保留的官方安装文档 `T_HUBER_INSTALLED.txt.gz`、已保留的 functional wrapper、
refs 实现和 native schema 共同固定分段公式、参数和调用边界。没有绕过 localhost:27891 获取资料。

SmoothL1 的 functional 简短说明说 beta 必须为正，但模块文档明确说非负、beta=0 等于 L1，
wrapper 和 refs 也有显式零分支。当前选定模块任务按这个明确零分支处理。Huber delta 则必须
严格为正。依据不是被测 attorch 的输出。SmoothL1(beta=0) 的相等点梯度使用已有官方
autograd 最小范数子梯度规则与 sign(0)=0。

对 d=input-target，SmoothL1(beta>0) 为 d²/(2 beta) 或 |d|-beta/2；Huber 为 d²/2
或 delta*(|d|-delta/2)，按 |d| 小于阈值分段。正阈值边界两侧函数值和一阶导数连续。
因此 SmoothL1 元素导数为 d/beta 或 sign(d)，Huber 为 d 或 delta*sign(d)。完整 VJP
乘对应 upstream，mean 再除总元素数，并返回 target 的相反梯度。none 必须接收完整张量 upstream。

## 求值与独立检查

`kernel_spec/regression_loss.py` 复用精确有理数和张量运算；`adapters/HUBER.py` 先验证物理 f32
数据和完整形状。阈值保留收到的 Python binary64 数值；不能把传入的 0.1 重新解释成精确1/10，
也不能默默改成 float32 参数。beta=0 在任何除法之前分支，空输入、特殊值等未核验范围拒绝。

`tests/test_huber_independent.py` 用独立 Fraction 标量循环覆盖两种损失、三种 reduction、
零阈值、正负分支和阈值相等点；同一完整前向和两侧梯度与公开 PyTorch autograd 核对。
另以每个多项式分段内的精确中心差分核验导数，并检查非均匀 upstream、输出遗漏、dtype、形状、
非物理数值、非法阈值及不支持的参数。经验 PyTorch 对照不承担严格包围或数学相等的认证。

开发测试先暴露尚无合同。初版实现又暴露实数包装对象不能用 Python 对象相等判断零、误读实现
调用不存在的辅助函数等问题；均在准入前修复。合同检查拒绝未兑现的零分支及失效变异，没有假通过。
修复后10个合同样例相容、6个错误公式被拒绝；独立测试覆盖完整前向与两侧梯度。

来源、公式、输入域和实现的本次复核由 Codex 完成，不声称独立第三方审核或全域形式证明。
通过后保留当前合同和适配器，允许该范围内的数学求值；实际 Triton 调用的状态、全部启动与输出
对应仍需独立完成。现有外部批次仅作完整公开输出的开发对照，不自动增加正式绑定数量。
