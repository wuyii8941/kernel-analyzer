# ONNX Runtime 对照：结论（按定稿方法重新解读）

ONNX Runtime 不是更权威的参照，用它只因为代码与 PyTorch 无关。只有在 ONNX 与规格定义相同的运算上，结果一致才算证据。

定义相同且一致：softmax、log_softmax、logsumexp、CE（权重、ignore_index、三种归约、全部忽略时为 NaN）、LayerNorm、RMSNorm、GroupNorm、BN 输出与运行均值（动量换算后）、GELU（erf 与 tanh）、bf16 与 f16 转换（逐位）、FP8 饱和转换（逐位）、注意力的默认缩放与左上因果对齐、RoPE 两种布局、scatter 四种归约、gather、cumsum、matmul。

舍入量级的差异来自 ORT 自身的计算精度：GroupNorm 默认 stash_type=float（改为 double 后差 1e-16）、Attention 的 double 路径约 1e-8、GELU tanh 的 double 路径约 6e-9。

定义不同，不构成检验：BN 运行方差（ONNX 用总体方差，动量方向相反）；FP8 在 saturate=0 时的溢出判据（ONNX 表格以舍入前的值与最大值比较）。

ORT 自身的问题：
1. 因果掩码加 softcap：ORT 先加掩码再做 softcap，被掩位置变为 −softcap，概率泄漏到未来位置。ONNX 规范经 onnx#7867 定为先 softcap 再加掩码，ORT 的 CPU 实现此后已修复。
2. FP8 E4M3FN、saturate=0、|x|∈[480,496)：ORT 给 ±448，与 ONNX 文档和就近偶数舍入都不符（候选，未查 ORT issue）。
3. FP8 E5M2、saturate=0、|x|∈[61440,65536)：ORT 给 NaN，文档与就近偶数舍入都应为 Inf（候选，未查 ORT issue）。

本轮未发现规格错误。运行：python3 battery.py（需要 onnxruntime；无需 onnx 包，模型由 onnx_min.py 直接编码）。
