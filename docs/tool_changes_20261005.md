# 工具改动（2026-10-05）：对规格检查入口与参照求值器的四处精度修正

## 1. 对规格检查入口 `scripts/tool_spec_check.py`

把 blind_test_v2 阶段 2 的做法（e_sem = K_R − f 进入统一判定层）做成通用入口：

- 一个用例 = kernel 调用（在 `TritonLaunchRecorder` 下运行）+ 规格 f（被实现或被替换函数的文档语义，
  在同一输入上求值）；
- K_R 由 `evaluate_sequence` 对该调用的全部 launch 组合求值；e_num = K − K_R 与 e_sem = K_R − f
  都以有向区间进入 `analysis.assess_units`（R1/R2/R3/R5、端点保守推断、检测器 2.1）；种子默认
  0–31 开发、32–95 确认；
- 规格两种写法：严格的 float64 区间包围（RoPE 用例），或 float64 求值加声明界 2⁻⁴⁰·max|f|（筛查用；
  确认一个发现时再换严格包围）；
- 报告里写明：TTIR 覆盖、参照完整比例、未建立原因、中止的程序及原因、非 Triton 写出的输出、
  在最后一次 Triton 写入之后又被非 Triton 操作改过的输出（例如 FLA 反向的 `dk.add_(dk2)`），
  后两类不参与判断。

用例分组：`tool_spec_cases_{liger,flex,inductor,tridao,fla}.py`。

## 2. 参照求值器 `reference_eval/ttir_eval.py` 的四处修正

每处都有对应测试：修正前失败、修正后通过（`tests/test_reference_eval_ttir.py`，52 项全过）。

| 修正 | 起因 | 规则 | 测试 |
|---|---|---|---|
| 已建立的精确 0 吸收未定义整数（`muli`、`andi`） | FlexAttention 的块稀疏遍历：越界预取的 next_block（掩码读取、无 `other`）只以 `jump * needs_jump` 进入指针跳转，needs_jump = 0 时与它无关；原规则把未定义一路传到全部输出 | 任一操作数是已建立的 0 时结果为已建立的 0（整数 x·0 = x&0 = 0 对任何 x 成立；浮点不适用，0·NaN ≠ 0） | `masked_prefetch_jump` |
| 同一个 SSA 值的自比较只由 NaN 决定 | Inductor 的 `maximum` / `clamp_min` 写成 `where((a > b) \| (a != a), a, b)`；原规则把 `a != a` 的两侧当成区间里的两个独立数，判不出，路径合并使参照宽到 0.2（`F.normalize`） | 两侧是同一个值时：非 NaN 则 eq/ge/le 为真、ne/gt/lt 为假；NaN 时按有序/无序谓词 | `nan_propagating_clamp_div` |
| bool 存储可经 int8 指针读写 | Inductor 把 bool 输出转成 int8 写（交叉熵反向），原规则拒绝这种重解释 | i8 与 i1（torch.bool）互访；写入存 0/1，读出未决的 bool 记为未建立 | （由 Inductor 交叉熵反向用例覆盖） |
| 自定义组合函数的 scan | Inductor 的 logcumsumexp 用带组合区域的 `tt.scan`，原来只支持求和 | 按扫描顺序依次折叠组合区域；Triton 要求组合函数可结合，实数下与树形顺序结果相同，每个前缀的区间包住其实数值 | `log_cumsum_exp` |

另有两处小修正：

- `arith.constant true/false` 在 MLIR 里不打印类型，解析器补成 i1（原来导致 IndexError）；
- `tt.dot` 不打印 `inputPrecision` 时默认是 IEEE（与 `emulate.py` 一致）；原来把它标成 tf32，只影响
  原因标签，不影响参照值。

## 3. 已知局限

- 掩码读取不给 `other` 的值在 TTIR 语义里是未定义的。FlexAttention 反向在 Q_LEN 不是块大小整数倍时，
  dv 依赖这样一个值（越界行的 LSE 进入 exp2 后与 dO 相乘），工具判为参照未建立。PTX 显示 Triton 3.6
  的 NVIDIA 后端在这些读取前把寄存器置 0，所以这台机器上结果正确；这是对实现行为的依赖，不是工具错误。
- deepseek_v4 这类含离散选择的模型，一 ulp 扰动的条件数底本身约为 1，普查判不了。
