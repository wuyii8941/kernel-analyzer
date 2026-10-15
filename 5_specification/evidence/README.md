# 证据（不决定条款状态，按条款编号引用）

| 目录 | 内容 | 能支持什么 | 不能支持什么 |
|---|---|---|---|
| ort_witness/ | 规格与 ONNX Runtime 1.24.4、SciPy、NumPy 的对照，31,133 次比较 | 定义相同的运算上，规格没有转写错误的证据 | 定义不同的运算（BN 运行方差、FP8 溢出判据）上的任何结论 |
| historical_misreadings/ | 按 v2.4 §3 描述重建的 v1 错误（SGD maximize 次序、BN 计数器），以及由参数含义推出的关系能否抓到它们 | 第二批合同的误读来源 | 其余 6 个已改角色（尚未重建） |
| theory_checks/ | THEORY.md 撤回部分的反例与数值核对 | METHOD.md 中的撤回说明 | — |
| declared_semantics/ | CE 规格与 Liger CE kernel 源码中声明公式的高精度对照（400 组配置） | softcap 可作为前置变换与 CE 规格组合；全部忽略时两者约定不同 | Liger kernel 在 GPU 上的实际行为 |
