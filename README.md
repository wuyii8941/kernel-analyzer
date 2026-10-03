# Kernel Analyzer

Kernel Analyzer 检验训练中数值实现差异的系统性作用：给定一次真实的 Triton kernel 执行，它从编译产物
自动生成带严格误差界的参照，并用统一流程检验这个实现选择相对参照，是否对参数更新产生非零的平均作用。
研究问题、估计目标与已纠正的说法以 [阶段性总结](docs/stage_summary_20261002.md) 为准；五步证据链的推导见
[讲稿](docs/bias_chain_final.md)。

## 贡献

1. **从 TTIR 自动生成带误差界的参照 K_R。** 参照按 IR 操作的声明语义组合求值，不按算子名查手写公式：
   锁定的 Triton 3.6.0 注册的 229 条操作逐条支持或明确拒绝；float64 区间端点用误差无关变换判定舍入方向，
   初等函数用 MPFR 定向舍入；参照内存、按参照值判定分支、控制依赖、跨 program 冲突与 atomic 次序都有
   规则；每个输出元素归入完整组合参照、条件局部参照或参照未建立。
   见 [自动参照结果](docs/auto_reference_results_20261002.md)。
2. **用统一流程分析实现选择对训练的作用。** 捕获包加一份测量声明（位置、坐标、模式、测量点、比较集合、
   方向规则、开发/确认划分）给出参照、残差、三类与端点保守统计；Liger FP32 dW 累加从重新捕获开始重跑，
   与案例专用脚本的结果逐项相同。
3. **用校准、对照和真实干预检验方法。** 答案已知时判定规则的误报率与检出能力（含高维灵敏度的推导与三种
   补法）；9 个真实 kernel 上 34 处单点改动与 12 个阴性对照；与只看误差大小的方法正面对比；交叉熵近似除法
   从局部一路追到更新；同一份 TTIR 换 Triton 版本编译后的下降差异与跨版本比较。见
   [工具检验](docs/tool_validation.md)。
4. **核内定位。** 按锁定版本的下降规则逐位模拟设备上的 FP32 执行（近似指令由设备对模拟出的操作数执行）。
   基线复现：开发集 100 个、保留集 101 个捕获包上，可模拟元素按存储位模式全部相同（规则与选择在保留集上冻结）。
   模型反事实：一次替换一个节点，定向舍入的 5 处改动定位正确，Liger 交叉熵的偏差落在 ÷N 节点。真实干预：
   11 处「换成正确舍入」的真实 kernel 改动，模拟预测全部逐位等于设备输出。端到端训练（Qwen3-1.7B、bf16、Liger 与
   torchao，300 步）中捕获的 142 个启动：参照全部建立；逐位模拟冻结规则下 103 个通过，补上 IEEE 有符号零规则并改为
   从 PTX/SASS 读取合成选择后 142 个全部通过（未参与修正的三步也通过）。尚未检验的是：不给机制与方向时能否发现
   陌生算子的 bias。

## 使用

```python
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, save_launch

with TritonLaunchRecorder() as rec:          # JIT 与 Inductor kernel 都经过这里
    run_training_step()
for i, launch in enumerate(rec.launches):
    save_launch(launch, f"capture/unit000/candidate/launch{i:03d}")
```

```bash
# 参照（可按单元并行）与统计
python scripts/run_reference_analysis.py --declaration D.json --stage reference --out REF
python scripts/run_reference_analysis.py --declaration D.json --stage statistics --out REF --report R.json
```

声明示例：`results/reference_eval/declarations/liger_fp32_order.json`。单个 kernel 的参照与覆盖：
`KernelReferenceEvaluator(parse_ttir(launch.asm["ttir"])).evaluate(launch)`、`kernel_coverage(module)`。
核内定位：`emulate.verify(launch)`（逐位核对）与 `emulate.localize(launch)`（逐节点贡献）。
环境：主线 `ka_main`，需要 Liger/torchao/transformers 的捕获在 `liger` 环境（见
[环境](docs/environments.md)）；缓存与临时文件都在仓库内 `.cache/`。

## 已检验到什么程度

- 生产求值器的回归测试覆盖审阅中发现的错误（大整数转浮点、负零符号、无舍入模型、控制依赖、中止
  program），参照求值相关测试全部通过。
- 静态覆盖：65 份 TTIR 全部解析并完整处理；实际求值的 kernel 另列，含三类计数与耗时。
- 判定规则在平均为零时误报率接近 5%（给出区间），灵敏度随维度下降，已画出曲线。
- 局限：区间依赖问题会让少量离散判定成为参照未建立；逐 program 求值较慢；kernel 内部中间值不可观测，
  核内定位依赖逐位模拟，张量核点积、扫描与 atomic 处停止，两类由 ptxas/LLVM 调度决定的合成选择要由
  输出确定并报告；结论限于声明的总体、坐标与测量点。

## 早期研究证据

AdamW8bit 保存残差补偿是从根因到独立训练改善闭合的案例：8 对确认训练 OFF−ON 平均验证 loss 差
+0.0283，95% 配对 t 区间 [+0.0157, +0.0408]（固定 Mamba checkpoint 与评估集）。其余案例的根因、
平均作用与训练后果见 [主张账本](docs/claims.md) 与 [全部案例结论](docs/root_cause_closure_current.md)。
旧的 `check_bias(candidate, reference, make_inputs)` 入口仍可用于已有手写参照的场合。

## 目录

`src/kernel_analyzer/reference_eval/` 是自动参照与统一入口；`scripts/` 是捕获、实验与复算；`tests/` 是
验证；`results/` 是协议与机器结果；`docs/` 的入口见 [文档索引](docs/README.md)。
