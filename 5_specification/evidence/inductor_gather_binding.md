# PyTorch Inductor gather 绑定

调用边界是 PyTorch 2.10.0 的公开 `torch.gather`：连续有限 float32 `[32,64]` 输入、显式 int64
`[32,24]` 索引、`dim=1`、`sparse_grad=False`，返回完整 float32 `[32,24]` 结果。值输入与索引输入
均属于任务前态并要求保持不变。

任务规格使用已准入的 INDEX `real` profile。保留的 PyTorch 2.10 文档和 native schema 固定默认
functional gather 的坐标规则、同秩要求、非归约轴形状约束和 `sparse_grad` 选项；独立 NumPy 坐标循环
与 PyTorch 对照覆盖重复索引、不同轴和不同浮点 dtype。ground truth 逐坐标读取收到的 float32 输入，
不采用 eager 或已编译输出。

`compiled_observed_20261019.json` 确认实际编译生成一个 `triton_poi_fused_gather_0`，完整写出
`[32,24]` 结果，值输入和索引均未改变。与 eager 逐位相同只用于核对调用边界，不作为规格。

正式报告 `.cache/spec_trial_20261019_gather/production_report_20261019.json` 使用8个开发输入和16个确认
输入。18,432个程序参照坐标的完整率和分辨率均为1；没有语义缺失、绑定、特殊值、预算、前提或包围
失败，完整公开输出均参与比较，重复执行逐位一致。K−Gπ、Gπ−f 和完整 K−f 在本批输入的每个坐标上
都精确为零。

零残差使每组统计样本完全退化，校准方向也为零，因此报告保留 `statistics insufficient`，不把这批
精确一致结果写成总体无 bias 或等价证明。这是统计层有意拒绝过强结论，不是 ground truth 或绑定失败。

绑定不扩展到其他 shape、dtype、轴、int32 或越界索引、`sparse_grad=True`、特殊值、非连续布局、
gather backward、scatter、out 重载或任意同名 kernel。
