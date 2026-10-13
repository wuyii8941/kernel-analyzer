# 工具的第一个完整案例：Triton 教程的 flash attention

日期：2026-10-13。对象：Triton 3.6.0（`ka_main`），RTX A6000（sm_86）。

## 目的

用一个真实、常见的算子检验工具能不能被当作审查工具使用：写好一个 Triton 算子、给一次代表性输入（可选地给一个规格），
工具能不能说出有没有系统偏差、来源在哪一行、是什么机制、怎么改，判断不了的地方能不能明确说判断不了。过程中发现的工具缺陷
就地补上，每处先写修复前失败的测试。

## 材料

- `tutorial_fwd.py`：Triton 3.6.0 教程 `06-fused-attention.py` 的前向 kernel，逐字搬来（MIT 许可，SHA-256 e37baf30…；
  下载件放在 `.cache/untrusted/`），只去掉 autotune，用教程自己测试的配置（BLOCK_M 128、BLOCK_N 64、2 stages、4 warps）；
- `make_planted.py` → `planted_fwd.py`：在教程源码上逐行植入一处缺陷的四个变体：

| 变体 | 植入的缺陷 | 性质 |
|---|---|---|
| tf32 | fp32 输入，两个 `tl.dot` 用默认 tf32 | 数值 bias（操作数截断） |
| scale | `qk_scale *= 1.44`（应为 1.44269504） | 语义 bias：声明的程序不是规格 |
| l_bf16 | softmax 分母 `l_i` 在循环里保留为 bf16 | 低精度累加 |
| acc_fp16 | 输出累加器用 fp16 | 张量核 fp16 累加器，不在下降规则内 |

- 用法（`audit_fa.py`）：一次调用 `kernel_analyzer.reference_eval.audit.audit(launch, spec={"desc_o": ...})`，规格是 torch 的
  float64 注意力。输入为标准正态，Z=1、H=2、head_dim 64；N = 1024（l_bf16 为 2048、BLOCK_N 16）；为控制时间只评估前两个
  program（共 16 个）。报告在 `results/audit_*.txt`。

## 结果

| 案例 | 逐位核对 | 事前预测（kernel 未运行） | 实测 K − G 对齐（标准误） | 判断 | 最大来源 | G − S（语义） |
|---|---|---|---|---|---|---|
| 教程 | 全部相同 | +1.2e-6 | +1.0e-6（3.2e-6） | 未发现显著偏差 | P 转 fp16（tutorial_fwd.py:64） | −1.4e-8 |
| 教程 causal | 全部相同 | +6.2e-6 | +6.1e-6（7.3e-6） | 未发现显著偏差 | 同上 | −7.0e-9 |
| tf32 | 全部相同 | −1.38e-3 | −1.44e-3（1.9e-5） | 显著，与预测一致 | 两个 tf32 dot（:45 −7.6e-4，:75 −6.8e-4） | −1.4e-8 |
| scale | 全部相同 | −3.9e-6 | −3.8e-6（3.2e-6） | 数值上不显著 | — | **−2.0e-3**：声明的程序与规格不一致 |
| l_bf16 | 全部相同 | +1.86e-3 | +2.0e-3（1.5e-4） | 显著，与预测一致 | `l_i` 的 bf16 转换（:352），识别为循环携带的窄类型累加 | −1.5e-8 |
| acc_fp16 | 部分（fp16 累加器的 dot 不在规则内） | 无法判断（dot 不在契约内） | −1.5e-5（2.1e-5） | 不显著；预测与逐位核对都如实放弃 | — | −1.4e-8 |

另外，l_bf16 在 N = 8192 时停滞很强（实测对齐 +0.155）：预测器报「无法判断：84% 的元素预测相对误差超过 2⁻⁴」，原因写明；
若照算，它给出 +0.13。

每份报告都给出改法与模型内反事实，例如 tf32：「传 `input_precision="ieee"`，或在 `tl.dot` 前用 `cvt.rna.tf32.f32` 就近取
到 tf32；该位点精确时对齐 −1.44e-3 → −6.8e-4」（两个 dot 各一条）；l_bf16：「把循环里的累加量保持在 fp32；该位点精确时
+2.0e-3 → −2.2e-6」。

## 发现并修复的工具缺陷（都先写修复前失败的测试，`2_tool/tests/`）

1. **对 `tl.dot` 输出的归约不可仿真**（`layouts.py`）。softmax 的行最大值与行和作用在张量核累加器布局（nvidia_mma v2）上，
   原来只认 blocked / slice 布局，逐位核对一个元素都仿真不了，预测器也因「归约次序未知」放弃。现在由寄存器 / lane / warp
   的线性基给出沿轴的组合次序（`linear_bases`、`linear_axis_params`），三种形状逐位通过。
2. **近似函数的下溢被当作「格外」**（`bias_predict.py`）。causal mask 的 logit 加 −1e6 后进 exp2，结果在 f32 里是 0，设备也
   给 0；原来因输入落在契约表范围外而放弃。现在精确结果低于正规数范围的通道按冲零确定。
3. **归因看不到张量核 dot**（`emulate.rounding_nodes`）。候选节点只列了早期的几种下降方式，tf32 flash attention 的报告里
   −1.44e-3 的来源完全缺失。现在列入所有舍入的下降方式（张量核 dot、扫描次序、设备组合的归约与扫描、整数转浮点、
   libdevice 精确求值、FTZ）。
4. **不在规则内的 dot 的提示有误导**（`emulate.py`）。fp16 累加器的 dot 被报成「tensor-core dot (tf32)」；现在写出元素类型。
5. **fp32 fma 链与张量核契约的状态传播**（`bias_predict.py`，外部审阅指出）。原来用精确前缀推进累加器，被舍入吞掉的增量
   在下一步又被算了一次（审阅反例 1 + 3×2⁻²⁴：预测 −1×2⁻²⁴，真实 −3×2⁻²⁴）。现在用设备值估计（精确前缀加已预测的均值误差）。
6. **没有审查入口**：新增 `reference_eval/audit.py`，一次调用完成事前预测、逐位核对、精确参照、逐节点归因、规格对比；
   报告中：
   - 每个输出给出显著性（统计显著且超过输出格式的单位舍入才算问题；统计显著但低于格式分辨率的单独标出）；
   - 工具自检：事前预测与实测是否一致；
   - 能量加权的相对误差（原来的不加权 RMS 被接近零的输出放大）；
   - 机制识别，包括循环携带的窄类型累加、贡献相消。

## 能说什么

- 对教程 flash attention 及植入的三类缺陷（数值截断、语义常数、低精度累加），工具在一次调用里给出了正确的判断、来源行、
  机制和改法，事前预测与实测一致；超出模型的部分（fp16 累加器、强停滞）如实报「无法判断」并写明原因。
- 改法后面的数字是**模型内反事实**（该位点精确时），不是运行修改后的 kernel；TF32 matmul 上同类修改已有事前预测并经设备
  证实的先例（`../composition_study/README.md` 第 10 节）。

## 局限

- 只评估了 16 个 program 中的 2 个、一种输入分布（标准正态）；前向，没有做反向（反向用原子累加 dQ 时会报「无法判断」）。
- 用时：N = 1024 时预测约 35 秒、归因约 40 秒（两个 program）；大规模输入需要抽样。
- 改法是按机制的规则给出的，不会自动改写源码；「贡献」由逐点替换得到，不同位点之间有相互作用，相加只是近似。
- 理论层（契约定义、组合定理与余项界）仍未写出，显著性判断与容差是经验的（见 `../composition_study/README.md` 10.4 节与外部
  审阅）。
- 只对 Triton 3.6.0 / sm_86 / 记录的编译选项成立。

## 复现

```bash
export HOME=/data1/tzh/kernel-analyzer/.cache XDG_CACHE_HOME=/data1/tzh/kernel-analyzer/.cache \
       TRITON_CACHE_DIR=/data1/tzh/kernel-analyzer/.cache/triton TMPDIR=/data1/tzh/kernel-analyzer/.cache/tmp \
       PYTHONPATH=/data1/tzh/kernel-analyzer/2_tool/src KA_CACHE_DIR=/data1/tzh/kernel-analyzer/.cache/composition_study/ka
PY=/data1/tzh/envs/ka_main/bin/python; cd 1_experiments/flash_attention_case
$PY make_planted.py
for c in tutorial_fp16 tutorial_fp16_causal planted_tf32 planted_scale planted_l_bf16 planted_acc_fp16; do
  $PY audit_fa.py --case $c; done
$PY fa_case.py --case planted_l_bf16 --N 8192 --programs 0 --suffix _N8192     # 强停滞：预测器放弃
```
