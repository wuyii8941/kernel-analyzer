# 路径 A 补齐计划：让每个候选调用键都得到三态结论（2026-10-20）

目标（用户 2026-10-20）：「能诊断就行，不需要能找到所有缺陷；补充到路径 A 全部能实现。」

「能诊断」的含义：对路径 A 记录到的每个候选调用键，工具给出 成立 / 不成立 / 无法判断 三态中的一个**语义结论**，
即 K−f 已经比较过，或者在第二步由人确认任务选择后比较过。被测实现在调用中崩溃或类型错误，结论记为运行失败（不成立），
它本身就是诊断。不要求在固定输入上触发所有缺陷。

基线：`sufficiency/results/PATH_A_20261020/`，317 个候选键中 126 个原样可用，191 个被阻断。

## 1 缺口与工作项

| 编号 | 缺口 | 键数 | 工作项 |
|---|---|---:|---|
| G1 | 参照求值超出预算（MATMUL 54、log_softmax 12、大张量逐元素 5、embedding 2） | 73 | 批量的严格参照后端：矩阵乘、softmax 族和逐元素运算按张量求值，不逐元素构造 Python 实数对象；与现有精确参照做差分测试 |
| G2 | 捕获上限 200 MB | 7 | 提高上限（受磁盘约束，按键记录实际大小）；依赖 G1 |
| G3 | 记录的参数不可序列化（`ones_like` 的 memory_format 等） | 6 | 记录器把 dtype、layout、device、memory_format 编码为可序列化的值 |
| G4 | 公开输出依赖计划未覆盖的张量状态（`cat` 4，`sum` 2） | 6 | 查明未追踪的内部读取来源，补绑定或补追踪 |
| G5 | 程序参照缺 libdevice 语义（`gelu_backward` 的 `__nv_powif`） | 4 | 工具补 `__nv_powif` 语义；同时允许 Gπ 不可得时只报告 K−f |
| G6 | 已准入合同缺 ATen 接口或 covers 名称 | 约 30 + D5、D8、D18 | NORM（`native_layer_norm`、`rms_norm` 及反向），CE（`nll_loss_forward/backward`），EMB（`embedding_backward`），ATTN（SDPA flash/efficient 前向与反向），INTERP（`upsample_nearest2d`），POOL（`avg_pool2d` 及反向） |
| G7 | 已准入合同的 profile 定义域 | 约 26 | ARITH：`pow.Tensor_Scalar`、`where` 分支含无穷、i64、bf16 调用中不能精确表示的标量（按「值输入按实际接收的精确有理值解释」）；MATHFN：幅度超过 100；INDEX：i64 数据、embedding 的 `padding_idx`；LAYOUT：`pad.default`、`slice_backward`、`repeat_interleave`、bool `tril` |
| G8 | 没有合同的算子 | 约 20 | 新合同 CREATE（`arange`、`zeros`、`ones`、`ones_like`、`scalar_tensor`）；RANDOM 增加「给定返回 mask」的 `native_dropout` 前向 profile |
| G9 | 生成规则 v1 只接受单一张量返回 | G6–G8 的多输出算子 | 生成规则 v2：多输出（tuple 返回）逐项映射，不由合同定义的输出（如 philox 种子）显式标为不比较并写明理由 |
| G10 | 结论规则不读特殊值判定（D-A5）；参照不完整没有单独类别 | D3 + 5 | 结论规则 v2：特殊值矛盾算检出；类别表增加「参照不完整」 |
| G11 | 被排除的 `copy_`、`to_copy`、`index` | 不在候选集 | 测量捕获的复制改为不经过 ATen 分派（直接设备内存复制），取消排除 |
| G12 | 运行失败的函数（D-A3） | 不在候选集 | 用原生运行记录的输入单独重放这些函数，给出运行失败或语义结论 |

## 2 规则

- 合同改动按 `CONTRACT_AUTHORING.md`：先写草案，再由独立复核（另起上下文，独立代码复算）确认后更新 `validated/`。
  不为让 FlagGems 通过而选读法；需要选择的读法分成 profile，由第二步选择。
- 共享数值代码（G1）是新增后端，不改动现有精确参照的语义；每个新后端都要有与精确参照的差分测试，并在 `status.md` 记录。
- 修复先写修复前失败的测试。
- 结果在**同一批记录**上重新绑定与评价（A2），与基线逐键对照；新的协议为 `OPERATOR_ENTRY_PROTOCOL_A2.md`，基线结果不覆盖。

## 3 顺序

1. G1（最大缺口，也决定 G2 能否成立）与 G10（规则）。
2. G9（生成规则 v2），因为 G6–G8 的多输出算子都依赖它。
3. G6、G7、G8：按键数从多到少。
4. G3、G4、G5、G11、G12。
5. A2 重跑与报告。
