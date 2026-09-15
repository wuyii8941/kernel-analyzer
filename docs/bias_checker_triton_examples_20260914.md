# 陌生 Triton kernel 的轻量 bias 检查

本页记录 `check_bias` 入口的一次真实 GPU 端到端验证。示例 kernel 不在
reference registry 中；运行方只显式提供 Triton callable、PyTorch reference 和
输入生成器，分析器不会从 kernel 名称或源码猜测语义。

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/data1/tzh/cache/triton_bias_checker \
PYTHONPATH=src:. \
/data1/tzh/envs/pt_nightly_transformers5/bin/python \
scripts/run_bias_checker_triton_examples.py \
  --device cuda --samples 32 \
  --output results/property/bias_checker_examples_20260914_final.json
```

## 实际结果

| callable | measurement | bias status | total RMS | 解释范围 |
|---|---|---|---:|---|
| Triton elementwise add | `VALID` | `SYSTEMATIC_BIAS_NOT_CONFIRMED` | 0 | 声明输入下与 `torch.add` 逐位一致 |
| Triton row sum（显式 FP16 输入舍入） | `VALID` | `SYSTEMATIC_BIAS_CONFIRMED` | `1.3923e-4` | aligned scaling 区间为 `[-1.0502e-4, -4.1174e-5]`，并且 held-out direction/sign 端点也确认 |
| Triton row sum（FP32 两块重关联） | `VALID` | `SYSTEMATIC_BIAS_NOT_CONFIRMED` | `6.0851e-8` | 同一 FP32 dtype 下只改变 chunk 重关联；本次输入没有确认系统性结构 |
| Triton stable softmax | `VALID` | `SYSTEMATIC_BIAS_NOT_CONFIRMED` | `8.5154e-8` | 方向与 aligned 区间均跨零；不是“证明没有 bias” |
| Triton RMS normalization（native FP32 reference） | `VALID` | `SYSTEMATIC_BIAS_NOT_CONFIRMED` | 0 | 该 callable 的 BF16 输出在声明输入上与 native reference 一致 |
| Triton RMS normalization（reverse FP32 reference） | `VALID` | `SYSTEMATIC_BIAS_NOT_CONFIRMED` | 0 | 同一 FP32 reference 的特征顺序干预未改变 BF16 输出；未形成可见 bias |

六个 case 均完成 32/32 个样本，没有编译、执行、shape 或非有限值错误。结果文件保留
每个 case 的完整报告，包括输入分布范围、总能量、aligned ratio、校准/确认划分和
错误列表。

另外用一个带 `autograd.Function` 包装的陌生 Triton callable 做了 8 个样本的
backward 端到端检查：forward RMS 为 `0`，backward RMS 为
`0.009999990463256836`，输出为 `SYSTEMATIC_BIAS_NOT_CONFIRMED`、backward
为 `SYSTEMATIC_BIAS_CONFIRMED`。这里的 1% backward 缩放是显式正控制，用来验证
接口路径，不是自然训练案例；可复现实验脚本为
`scripts/run_bias_checker_triton_backward_example.py`。

同一组 32-sample 输出检查在第二次 GPU 运行中逐字节一致，结果保存在
`results/property/bias_checker_examples_20260914_repeat.json`。

## 结论边界

这验证的是“新 kernel 已经可以被包装成 callable 后，直接复用统一 bias 检查”。
它不是任意 Triton 源码的零配置分析：调用约定、输出分配、reference 语义和输入
生成仍需由接入方提供；原地写入、多输出、随机状态和 backward 需要相应 wrapper。

`SYSTEMATIC_BIAS_NOT_CONFIRMED` 只表示本次声明输入没有确认所检验的方向或缩放结构，
不表示全空间、其他输入分布或训练状态中不存在 bias。示例中的 row-sum case 是
显式精度变体的正控制，不应冒充自然训练案例或自动根因发现。
