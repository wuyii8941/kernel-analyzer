# FlagGems 5.0.2 MSE：真实调用与规格对应

此处审阅原安装包的 `experimental_ops/mse_loss.py` 前向入口，完整源码保留在
`sources/FLAGGEMS502_EXPERIMENTAL_MSE.py`。包元数据与 Apache-2.0 许可证已随此前 FlagGems
来源保留。调用文件通过安装包位置加载原模块，并与审阅源文件逐字节比较，没有复制重写 kernel。

任务是两个有限、连续、同形 float32 `[7,41]` 张量的无权重 MSE，reduction 显式取 none/sum/mean。
完整返回 loss，两项输入保持不变。该入口没有提供完整公开 autograd；此绑定也不承诺反向、out
变体、广播、空输入或其他 dtype/布局。任务依据为已经审阅的 MSE 全 reduction profile。

绑定逐项消费 x、target 和 reduction。none 对应完整 `[7,41]` 返回值；sum/mean 对应完整标量。
所有张量实参的前后态都被检查，调用范围要求包含全部 Triton 启动。形状和合法参数在调用边界验证。
`tests/test_flaggems_mse_binding.py` 拒绝遗漏 target/reduction、遗漏输出或状态、将 none 换成标量、
未捕获完整启动和修改只读输入；草案通过开发核对仍不能成为正式 bias 证据。

`flaggems_mse/development_observation_v1.json` 是升级前的实际 GPU 观察：三种 reduction 各捕获
一次完整启动，原始 TTIR、输入、完整输出和标准实现结果都保留。none 使用原逐元素 kernel；
sum/mean 使用原归约 kernel，调用内通过 `torch.zeros` 分配初始化标量累加器，再执行 atomic_add。
287个元素只需一个 program；不存在多个 program 的加法顺序选择。连续输入的 contiguous 操作
保留原数据，none 的 reshape 和标量的同 dtype 转换保留返回存储。

这里不能把实际观测到的累加器零值直接当作额外公开输入。正式入口仍须用通用的初始化与 producer
检查证明它是在本调用内建立的常数；无法建立时必须保留拒绝，而不能用这三次数值对照抵消。
mean 的 Python 比例参数进入 f32 kernel 参数，其数值语义由程序参照处理，数学目标仍是精确除以287。

以上是 Codex 对来源、公式、调用边界和映射的审阅，不是独立第三方审核。完整输入/输出对应关系
已完成审阅；各 reduction 的正式求值可用性和统计结果由随后保留的测量报告分别决定。

## 正式测量与独立解释

三个测量声明在观察统计结果前固定：每个8个开发输入、16个确认输入，x/target 独立取
normal(0,2)，种子从120开始。结果为同目录 `flaggems_mse/{none,sum,mean}_measurement_v1.json.gz`。
每种 reduction 的24份绑定均可用于测量；none 的6888个输出坐标、sum/mean 各24个完整标量
均建立调用级参照，完整率和分辨率均为1。sum/mean 的累加器被通用 producer 检查认定为
`const: aten.zeros.default`，并非把观测字节补作额外任务输入。全部输出和两项输入状态均被覆盖。

| reduction | 程序参照 Gπ 与任务 f | 设备到程序的平均绝对差 | 当前统计结论 |
|---|---|---|---|
| none | 所有坐标均未排除零 | 约3.87e-7 | 当前规则未确认稳定方向 |
| sum | 所有坐标均未排除零 | 约7.37e-5 | 确认样本偏斜超过既定阈值，无法判断 |
| mean | 24个坐标全部排除零，平均绝对差约8.45e-8 | 约3.23e-7 | Gπ−f 确认非零；K−Gπ 与 K−f 未确认稳定方向 |

mean 的差异还可独立推导。实际接收的 f32 系数 c 满足

`287*c - 1 = 45/2^32 > 0`，

所以在这个单 program 的精确程序语义下，`Gπ−f = (45/2^32)*f`。除 loss 为0外，这是正向
固定比例差异。`verify_scale.py` 用捕获的实际 Python 参数、f32 编码和完整输入做 Fraction
复算，并与数学规格交叉核对；结果保留为 `scale_derivation_v1.json`。报告中投影方向的正负
不能直接当成原始残差的正负；这里的正向来自上述明确公式。

这说明分开检查程序公式与数学目标，可以解释直接设备对照未确认的微小系统差异。它不证明
真实训练影响，也不证明总体 K−f 无偏或有偏。sum 的“无法判断”没有通过放宽阈值或换种子消除。
本轮只增加这一份完整前向调用绑定，涵盖所述三种 reduction；不扩展到其他形状、版本或反向。

相关绑定回归最终正常退出：83项通过、1项因环境跳过，16个子检查通过。该次运行期间未修改
仓库文件；来源、绑定计划与调用文件的保留副本也已核对一致。
