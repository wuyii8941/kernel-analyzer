# 接入指南：用 `kernel-analyzer check` 检查一个陌生 kernel

适用于 Triton kernel（直接调用的 `@triton.jit` 函数，或 `torch.compile` 生成的 Inductor kernel）。锁定环境：
Triton 3.6.0、sm_86；其他版本先按 `CLAUDE.md` 重跑注册表枚举。

## 1. 写一个绑定文件

只写调用与数据，不写参照、不指定检查什么。模板：`2_tool/examples/bindings/TEMPLATE.py`。

```python
NAME = "my_kernel"                     # 报告名
IMPLEMENTATION = "what runs"           # 一句话：哪个库、哪个 kernel、什么配置

def setup():                           # 可选：编译、预热（不计入测量）
    ...

def make_inputs(seed):                 # 一次独立抽样：同一个 seed 必须给出同样的输入
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(64, 1000, generator=g).cuda()}

def run(inputs):                       # 调用被测实现，返回 {输出名: 张量}
    y = my_kernel(inputs["x"])
    return {"y": y}

def spec(inputs):                      # 可选：外部数值参照；给出后进入兼容的模式 B
    return {"y": f64_point_spec(reference_math(inputs["x"].double()).cpu().numpy())}
```

- **模式 A（不给 `spec`）**：只测 e_num = K − K_R（实现相对它自己按实数解释的语义），报告写"任务语义未检验"。这是陌生
  算子的默认用法，不需要任何参照。
- **模式 B（给 `spec`）**：另算 K_R 与外部数值参照之差，以及 K 与该参照之差。报告保留旧字段名 `semantic` 以兼容已有记录，但明确标成 `reference_difference`，`semantic_verdict` 始终为 `not_established`。这条接口没有核验参照来源、参照实现或它与 kernel 的任务对应，不能据此报告语义 bias。
- **合同模式**：统一入口的声明使用 `compare.mode = "contract"`，并给出 `binding_plan` 与 `specification_root`，不再提供 `spec()`。工具只接受已审阅并独立保留的绑定计划、调用文件、合同和适配器；每个抽样调用都重新核对输入前态、输出后态、全部 Triton 启动和合同参照。绑定计划里的 `measure_output` 固定合同结果与报告输出的对应关系。
- 合同模式还会沿已捕获的 Triton 启动追踪到达公开输出的所有张量来源，包括整数索引和布尔 mask。闭包、全局变量或模块状态中的张量若未写入绑定计划，当前样本直接按绑定失败拒绝；能够由已声明输入的拷贝或精确常量解释的中间张量可以继续使用。这样不会把遗漏的任务输入造成的差异误报成 kernel bias。
- `f64_point_spec(value)` 只是 float64 点值周围的经验筛查带，不是严格误差包围，也不是 ground truth。合同模式只把 `exact` 或 `rigorous_enclosure` 的数值输出送入 bias 统计；允许集合、离散结果和声明近似仍单独判断。
- K−Gπ 分解要求输出坐标由记录到的 Triton 启动写出（可以是视图）；中途由 torch / cuBLAS 算出的坐标不进入该分解。合同模式的 K−f 仍覆盖整个公开输出，并把覆盖范围单独写入报告。

合同模式的最小声明片段：

```json
{
  "call": "bindings/my_call.py:call",
  "compare": {
    "mode": "contract",
    "measure": ["loss", "gradient"],
    "binding_plan": "../5_specification/corpus/call_bindings/my_call.json",
    "specification_root": "../5_specification"
  }
}
```

绑定计划的 `subject.measure_call` 必须与 `call` 完全相同，`subject.runtime_package` 必须写明实际安装的发行包和版本。每个实际输入都必须整体进入某个合同输入，不能只映射张量元素或参数对象中的一个字段；确实与任务无关的输入要在 `unused_inputs` 中逐项写明依据，嵌套在容器里的张量会被拒绝。合同承诺的复合输出必须保留全部字段和位置，不能只挑其中一项测量。当前三十个 profile 已通过规格参照准入；INDEXREDUCE 只覆盖物理有限同 dtype 浮点张量、int64 index 和可表示 alpha 的函数式 index_add 前向值，原地更新和其他 indexed reduction 继续拒绝；RANDOM 只覆盖无需随机流的 dropout 前向边界值和显式 Boolean mask 的 native dropout backward，随机训练前向继续拒绝；CLIP 只覆盖一个物理有限梯度缓冲区的 clip_grad_value_ 完整写回，范数裁剪和未完整对应的多缓冲区调用继续拒绝；SGD 只覆盖一个稠密实数参数的一步完整参数与 momentum-buffer 状态转移，内部 fused、AMP overflow、可微和多参数任务继续拒绝；BN 只覆盖带显式 running statistics 的物理有限 BatchNorm 推理前向值，INDEX 只覆盖物理有限浮点数据与 int64 索引上的默认只读 lookup，POOL 只覆盖零 padding、floor mode、values-only 的物理有限二维 max_pool2d 前向值，LOGIC 只覆盖已选的物理有限比较、逻辑、布尔位运算、truth reduction 和浮点 `isnan`，ARITH 只覆盖有限、同 dtype、物理可表示的 f16/bf16/f32/f64 常见逐元素操作，CAST 只覆盖有限 f16/bf16/f32/f64 的显式 RTNE 转换，CONV 只覆盖批量非转置 1-D/2-D 有限浮点默认操作，LAYOUT 只覆盖有限浮点张量的函数式逻辑值和形状变换，MATMUL 只覆盖有限 f16/bf16/f32/f64 的默认函数重载，SORT 只覆盖物理有限浮点、每个选定行无并列值的精确 sort/argsort/已排序 top-k，RANKMEAN 仍要求当前任务明确选择 token 等权。二十三份真实调用计划中，Inductor 末维 softmax、融合 mean CrossEntropy、Inductor SiLU、直接 Triton float32 matmul、grouped Conv2d、行求和、逐元素 exp、显式 bf16 转换、逐元素乘加、矩阵转置、大于零比较、行 index-select、无并列值的行排序、max_pool2d、BatchNorm 推理、dropout backward、index_add、梯度值裁剪与 warm-buffer SGD 十九份已审核并保留，另外四份仍是草案。严格布尔输出会另外报告完整误判、假阳性和假阴性计数。开发测试不能代替独立的调用绑定复核，也不能把这十九份绑定扩展为同名算子的普遍认证。

## 2. 运行

```bash
kernel-analyzer check my_binding.py --out pre-reorg-20261009:results/my_check --seeds 96   # 前三分之一为开发种子，其余为确认种子
```

每个用例一个 JSON。`--seeds` 小于 6 时统计无法成立。

## 3. 读报告

每个输出：

| 字段 | 含义 |
|---|---|
| `reference_classes` | 参照建立的比例（完整组合参照 / 有限值），未建立的不进入统计，也不能算作阴性 |
| `numerical` | e_num 的判定（规则 R1–R5 与默认检测器）与大小（`scale.relative_rms`） |
| `semantic`、`total` | 模式 B：K_R 与外部参照、K 与外部参照的差，不能作语义裁决。合同模式：前者只在调用级 Gπ 已建立时表示 Gπ−f，后者表示完整公开输出上的 K−f |
| `special_values` | NaN / inf 的类别不一致（模式 A 为 K 对 K_R） |
| `depends_on_non_triton_intermediates` | 参照用到了非 Triton 算出的中间量，e_sem 混有上游的数值误差 |
| `launches[].float_atomics` | 浮点原子操作：K 依赖运行次序，e_num 在两次运行之间会变 |

合同绑定默认用 `numerical_difference` 程序参照，即把运算舍入留在 K−Gπ。若公开任务本身就是一个明确舍入的
转换，经过审阅的绑定计划可用带依据的 `program_reference.mode=rounding_check`，让该转换舍入同时进入 Gπ
与 f；报告在 `program_reference_mode` 原样记录选择。这个选择只改变 K−Gπ 与 Gπ−f 的归因，不改变完整
公开输出上的 K−f。

分档（`pre-reorg-20261009:scripts/tool_spec_summary.py`）：未检出 / 常数取整级（< 1e-7）/ 小（1e-7–1e-5）/ 候选（≥ 1e-5）/ 补偿（e_sem 大而
总差小，舍入决定了离散分支）/ 混合 / 判不了。"未检出"只说明在这些输入与测量点上没有检出，不是无偏的证明。

## 4. 常见问题

- Inductor kernel：引擎已关闭静态启动器、每个用例前重置 Dynamo；验证运行时补丁时用 `KA_PRELOAD`，它会关掉 Inductor 缓存。
- 一个调用里有多个 kernel：参照沿全部记录的 launch 组合；中途被 torch 改过的缓冲区会被标出来。
- 指针表（`tt.int_to_ptr`）：用 `TritonLaunchRecorder.register_implicit(*tensors)` 登记表里指向的张量。
- 程序太多、太慢：缩小输入；参照按 program 求值，耗时与元素数成正比。
- 支持不了的操作会让 program 中止并写明原因（`aborted_programs_seed0`、`not_established_reasons_seed0`）——请把原因
  记下来，这正是接入试验要统计的。
