# torchao 0.16 OptimState4bit 物理状态参考

活动 `OPT4BIT` 合同现在只定义一个能完整核对的任务：PyTorch 2.10 中，torchao 0.16 的
`OptimState4bit.zeros(...).copy_(float32_src)`。旧草案把完整 `AdamW4bit` 一步作为任务，却只返回反量化后的
两个动量；实际 `state_dict` 保留的是 packed `uint8` codes、每块 `float32` scale 和 `float32` qmap。因此旧草案
不能区分物理状态写错、nibble 顺序写反或 scale 写错，已经归档。

新参考按保留的官方源码独立计算：每块 absmax 与 `float32(1e-12)` 取最大值；float32 除法归一化；按照
canonical signed/unsigned qmap 的四步二分选择相邻 code；用 float32 residual 与 midpoint 作 ties-up 决定；最后把
较早元素放进高 nibble。输出要求包含全部 packed codes 和全部 scales。完整 qmap、signedness、逻辑 shape 与
block size 是必须核对且不得改变的前态绑定条件；输入前态与输出合在一起覆盖完整物理状态。

`observed_20261019.json` 在安装的 torchao 0.16.0、PyTorch 2.10.0+cu128 上对 CPU 与 RTX A6000 各跑七类输入，
包括 signed/unsigned、零块、不同块尺度及 4096 元素默认 128 分块。14 次调用中的 8,224 个 code、148 个 scale
和 224 个 qmap 坐标全部逐点相等，源张量与固定状态属性也全部保持正确。

`compiled_triton_observed_20261019.json` 进一步检查真实 `torch.compile` 路径。该调用生成三个 Triton kernel，
分别完成分块 scale、逐元素选码和 nibble 打包。11 次 4096 元素 CUDA 调用中的 22,528 个 packed bytes、352 个
scale 和 176 个 qmap 坐标全部相等，源张量未改变。

调用计划 `torchao.optim_state4bit.copy.f32.signed.4096x128.v1` 又将这三次启动作为一个完整调用正式捕获。
第一次运行暴露了工具不能读取无自身物理存储的 tensor subclass；修复后，第二次运行正确拒绝了未声明的 qmap 读取。
现在 qmap 是显式、只读的输入前态，codes 和 scale 是完整输出。第三次报告覆盖 8 个开发输入和 16 个确认输入：
49,152 个 code 参照坐标和 768 个 scale 参照坐标全部完整，每个坐标的设备值与任务值之差都是 0；两个输出都覆盖完整公开结果，
输入未被改变，三份 TTIR 均完整捕获。报告中的“统计不足”是因为所有残差都精确为零，样本无方差，因此统计规则不声称总体无偏；
它不表示出现了数值或任务不一致。冻结报告为 `.cache/spec_trial_20261019_opt4bit/production_report_20261019_r3.json`。

这些证据只认证上述 signed float32[4096]、block size 128 的 state-copy 调用。完整 AdamW4bit 还包含参数更新、step、两个 packed moment、
4096 size gate 和多参数编排；`bugs/B008` 已撤销旧的完整绑定。unsigned 及其他合法形状目前只有规格参照和官方实现对照，还没有真实调用绑定。
