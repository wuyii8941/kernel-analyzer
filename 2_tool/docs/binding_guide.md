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

def spec(inputs):                      # 可选：任务规格 f，给了就是模式 B
    return {"y": f64_point_spec(reference_math(inputs["x"].double()).cpu().numpy())}
```

- **模式 A（不给 `spec`）**：只测 e_num = K − K_R（实现相对它自己按实数解释的语义），报告写"任务语义未检验"。这是陌生
  算子的默认用法，不需要任何参照。
- **模式 B（给 `spec`）**：另报 e_sem = K_R − f 与总差 K − f。`f64_point_spec(value)` 把 float64 计算的结果当作带声明
  误差界（2⁻⁴⁰·max|f|）的规格，只作筛查；登记问题前要换成严格包围或交叉核验（见计划第 2 节）。
- `run` 返回的张量必须是被测 kernel 写出的（可以是视图）；中途由 torch / cuBLAS 算的输出会被报告为"非 Triton 写出"，
  不评。

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
| `semantic`、`total` | 仅模式 B：e_sem 与 K − f |
| `special_values` | NaN / inf 的类别不一致（模式 A 为 K 对 K_R） |
| `depends_on_non_triton_intermediates` | 参照用到了非 Triton 算出的中间量，e_sem 混有上游的数值误差 |
| `launches[].float_atomics` | 浮点原子操作：K 依赖运行次序，e_num 在两次运行之间会变 |

分档（`pre-reorg-20261009:scripts/tool_spec_summary.py`）：未检出 / 常数取整级（< 1e-7）/ 小（1e-7–1e-5）/ 候选（≥ 1e-5）/ 补偿（e_sem 大而
总差小，舍入决定了离散分支）/ 混合 / 判不了。"未检出"只说明在这些输入与测量点上没有检出，不是无偏的证明。

## 4. 常见问题

- Inductor kernel：引擎已关闭静态启动器、每个用例前重置 Dynamo；验证运行时补丁时用 `KA_PRELOAD`，它会关掉 Inductor 缓存。
- 一个调用里有多个 kernel：参照沿全部记录的 launch 组合；中途被 torch 改过的缓冲区会被标出来。
- 指针表（`tt.int_to_ptr`）：用 `TritonLaunchRecorder.register_implicit(*tensors)` 登记表里指向的张量。
- 程序太多、太慢：缩小输入；参照按 program 求值，耗时与元素数成正比。
- 支持不了的操作会让 program 中止并写明原因（`aborted_programs_seed0`、`not_established_reasons_seed0`）——请把原因
  记下来，这正是接入试验要统计的。
