# FlagGems 5.0.2：公开 RMSNorm 默认参数与完整 VJP

本次来源和调用对应复核使用实际安装的 FlagGems 5.0.2、PyTorch 2.10.0+cu128，以及已经归档的
RMSNorm 官方定义。原库源码、默认参数对照和未启用记录器的复现见
`corpus/external_batch/results/norm_rms_defaults_v1.json.gz`；完整开发及另一套实现验收见同目录的
`norm_development_v2.json.gz`、`norm_acceptance_v1.json.gz`。本文件是本地来源与实现审阅，不是第三方认证。

调用固定为公开 `F.rms_norm(x, (7,), weight, eps=None)`，并返回对显式 upstream 的完整 x/weight VJP。
x/upstream 是连续有限 CUDA float32[2,7]，weight 是 float32[7]，绝对值不超过8。三个输入值都不允许
修改。调用边界检查形状、dtype、布局和值域；detach 只建立共享原存储的 autograd 输入。调用内只启用
FlagGems 的 rms_norm 替换；没有把原库最终的 torch.sum 换成另一个实现。

数学任务使用已审阅的 NORM.torch_autograd，完整输出为 y、grad.x、grad.weight。公开任务的 epsilon
来自输入 dtype，因此固定为2^-23；实际 kernel 收到的1e-5属于程序实现。不能把 kernel 的参数写回
数学规格来消除差异。normalized_shape 和 eps 是这个固定公开调用的常量，三个实际收到的张量均逐项
进入规格，未忽略可变化的任务输入。没有公开 bias 或 saved-statistics 输出。

`flaggems_rms_norm/probe.py` 的开发观察完成全部任务对应、只读前后态检查和三次 Triton 启动捕获。
实际输出来源分别为：前向 kernel 写 y；第二个 kernel 写 dx；第三个 kernel 写 weight 梯度的分块
缓冲区，随后原生 `aten.sum.dim_IntList` 和 view 得到完整 dweight。后者不是 Triton 最后写出的存储。
额外两个开发输入的模式A诊断能完整解释 y/dx，明确把 dweight 列为非 Triton 输出。前向返回的
INV_RMS 在后两个 kernel 间的传递已由通用输入与存储检查建立，没有把观测中间值补成规格输入。

这足以审阅完整公开任务的输入/输出对应：K−f 使用三项完整输出的独立数学参照。程序分解的可用性
单独报告：y/dx 可检查 K−Gπ 和 Gπ−f；dweight 的 Gπ 尚未建立，不能只删掉这项梯度来获得完整分解。
工具已有分别报告完整任务比较和不完整程序分解的路径，本次复用它，不降低参照准入条件。

绑定回归拒绝遗漏任一输入、漏掉非 Triton dweight、错误梯度形状、输入变化、启动范围不完整，以及
为固定默认任务额外传入未对应的 epsilon。草案阶段的观察保留为开发证据，不能自动认证；计划和调用
文件在以上来源、实际过程及逐项对应复核后才保留为 reviewed。后续正式测量的统计结果不作为本次
任务选择或审核的依据。

## 固定分布测量与独立前向推导

`flaggems_rms_norm/measurement_v1.json.gz` 保留全部正式报告、已展开声明、计划和调用源码。
测量前固定8个开发输入、16个确认输入，种子144至167；x 各坐标取 uniform[-1/512,1/512]，
weight 取 uniform[0.5,1.5]，upstream 取 uniform[-1,1]。这是针对已发现默认参数问题选择的小输入
分布，不是盲测，也不代表真实训练中的出现频率。草案时期的完整观察另外保留在
`flaggems_rms_norm/development_observation_v1.json.gz`。

24次调用都通过完整任务绑定。y/dx 各336个程序参照坐标完整，dweight 的168个任务坐标全部比较，
但没有程序参照。K−f 的对齐规则在三项输出上均确认非零；固定方向规则没有确认，不能把两种问题
合并成“每个坐标的均值都偏向同一符号”。Gπ−f 在 y/dx 上也被对齐规则检出；dweight 只报告 K−f。
各项 K−f 平均绝对差分别约0.5394、275.955、0.3789；y/dx 的 K−Gπ 平均绝对差仅约1.27e-8、6.50e-6。
这些量受所选输入与梯度单位影响，不是训练影响估计。
本次设备为 RTX A6000（sm_86），Triton 3.6.0；未指定 other 的掩码读取按实际 PTX 已核对的零初始化
解释，相关依据保留在开发诊断中。不能据此承诺其他硬件或编译版本的相同行为。

前向差异可以独立解释。已读取的实际 TTIR 先计算 `m=sum(x*x)/7`，加入实际 f32 参数 c，再取平方根
倒数并乘 x 和 weight。c 的准确值为 `2748779/274877906944`，任务的 e 为 `1/8388608`，因此

`G_y = f_y * sqrt((m+e)/(m+c))`。

因为 c>e，每个非零前向坐标都向零缩小。声明的输入范围保证 `0 <= m <= 1/262144`，所以该比例
严格落在 `[109/1000, 267/500]` 内，即理想程序的前向幅度相对数学任务缩小46.6%至89.1%。
`verify_forward_scale.py` 只用捕获标量、f32 参数编码、声明范围及 Fraction 运算核对这个界，
没有调用 NORM 实现或设备结果；准确结果保存在 `forward_scale_derivation_v1.json`。
公式依据是对这份 TTIR 的逐项读取，脚本不自动证明任意未来 IR 与公式等价。该界只针对前向，
不代替硬件舍入、完整梯度程序分解或其他输入分布的分析。

完整必需规格检查通过，含914项 unittest 和78项外部调用 pytest；工具与新绑定回归71项及6个 subtest
通过。验证日志保留于 `flaggems_rms_norm/validation_v1.json.gz`。

后续报告修复将三种比较同时显示在命令行，并把无程序参照的 dweight 标为未建立；其完整 K−f 仍保留。
冻结声明的另一次运行与旧报告的24份绑定及全部数值、语义、总任务统计逐项相同。新报告、命令行输出
和77项回归记录另存 `flaggems_rms_norm/summary_reporting_v2.json.gz`，不覆盖旧测量。一次旧绑定测试
因近恒定预设差异的分布诊断而失败的记录也保留；该测试现直接核对完整参照及已知0.25/3的差异，
不再要求统计规则必须确认非零。统计判定逻辑没有因此放宽。
