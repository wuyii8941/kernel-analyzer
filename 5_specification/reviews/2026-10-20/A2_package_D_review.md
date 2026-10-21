# 路径 A 收尾 包 D 独立复核（ARITH / MATHFN / INDEX `real` 域扩展，2026-10-21）

复核人：独立复核方（未参与起草）。对象文件（sha256）：

| 文件 | sha256 |
|---|---|
| contracts/draft/ARITH_path_a_A2_20261020.yaml | 658c77cc2048031e61c69903ff8d507b44efff11b1174b5fcea2e370bdb1dc6b |
| contracts/draft/ARITH_adapter_path_a_A2_20261020.py | e07515f666e3738a8c14736749c2a995a825308de4baeba664d0159f3c79fddf |
| contracts/draft/MATHFN_path_a_A2_20261020.yaml | d4d8863e10b2b5e66994a0057d0443cb46de8763cd36879a46e79f22556d7ccb |
| contracts/draft/MATHFN_adapter_path_a_A2_20261020.py | 9fa06af809e5bf5428ca32643ca70b147a004eef968d22f18288f8dd9f5c50c3 |
| contracts/draft/INDEX_path_a_A2_20261020.yaml | c1616d3a831399415029d69666406c6489daeb5e47cbe3179a1017c4cd434dbf |
| contracts/draft/INDEX_adapter_path_a_A2_20261020.py | 991b0db9498afe1c91377ca1e999b5ebfe7b43c9e74cec3b724a31babaae8fec |
| kernel_spec/arith_mathfn_index_arb.py | 527e32b1565a0b4386774ed72116ae7bf5cf4b2401b56413fae658c20d868c21 |

## 结论

| 合同 | 结论 |
|---|---|
| ARITH | **ISSUES**（A1、A2 需改合同与适配器；A3、A4 只改文字） |
| MATHFN | **CONFIRMED**：上表两份 MATHFN 文件连同 `kernel_spec/arith_mathfn_index_arb.py` 可以晋升，并可放入 `validated/`（观察 O1 不阻塞） |
| INDEX | **CONFIRMED**：上表两份 INDEX 文件可以晋升，并可放入 `validated/` |

自动画像（autograd profile）未变：ARITH 四个 autograd profile 与顶层已验证适配器在 120 组随机输入上的结果类别和全部字段逐一相同；INDEX 浮点路径在随机输入上与已验证适配器一致。

## 来源核对

- T_POW 快照写明 exponent 是 “a single float number”，`out_i = x_i ^ exponent`。按整数值取整数幂，在实数上成立（x^3.0 = x^3）。
- IEEE754.POW0 对应 §9.2.1，`pow(x, ±0) = 1`（pown 同样）。AR.ZERO_POW 的依据成立。
- IEEE754.INF_ARITH 与 INVALID 对应 §6.1、§7.2，与 AR.EXTENDED_REAL 列出的无定义点一致。x/0 记为无定义，与有限 profile 保持一致。
- T_EMBEDDING_F 快照：padding_idx 只通过梯度定义；文中例子之所以返回零行，是事先执行了 `weights[0,:].zero_()`；正文还写明 nn.Embedding 在构造时把该行置零。TS_EMBEDDING_CPP.FORWARD（index_select）与 TS_FUNCTIONAL_EMB.EMB_PADDING（[-V,V)，负值加 V，None→-1）与 IX.PADDING_FORWARD 相符。
- T_GATHER、T_INDEX_SELECT 不涉及值运算，IX.INTEGER_DATA 成立。快照中 embedding 权重写作 “floating point type”，所以整数权重仍为 unsupported，这个处理正确。
- T_TENSOR_ATTR 快照第 97–99 行写着 “A floating point scalar operand has dtype torch.get_default_dtype() and an integral non-boolean scalar operand has dtype torch.int64”。AR.SCALAR_VALUE 没有讨论这句（见 A3）。
- T_DIV 快照第 24–25 行写着 floor “Equivalent to floor division in Python (the // operator) and NumPy's np.floor_divide”；T_FLOOR_DIVIDE 给出的是 `floor(input/other)`。两者遇到无穷操作数时结论不同（见 A1）。

## ARITH 问题

**A1：floor 除法遇到无穷操作数，草案没有写决定。**
- 位置：ARITH yaml:145-154（AR.EXTENDED_REAL）与 yaml:204-209（AR.DIVISION）都没有提到这种情形；适配器 :252-257 `_div`、:372 `_floor_divide` 直接按扩展实数取 floor。
- 现状：草案给出 `div(inf,2,floor)=inf`、`div(-1,inf,floor)=0`、`floor_divide(inf,2)=inf`。
- 文档冲突：T_DIV.FLOOR 指向的 Python `//` 与 np.floor_divide 给出 `inf//2 = nan`、`-1//inf = -1`；PyTorch 2.10 eager 在 CPU 和 CUDA 上同样得到 `[nan,-1,0,nan]`。T_FLOOR_DIVIDE 的公式 floor(x/y) 则支持草案的读法。
- 要求：文档在这里有真实分歧，必须写成决定，列出备选与取舍；或者把 floor 除法、floor_divide 遇到无穷操作数的情形定为 unsupported。trunc 模式与 eager 一致，没有问题。
- 复核测试暂不断言这些组合。

**A2：合法的 ATen Scalar 被判成 invalid_input。**
- 位置：适配器 :139-157 `_received`；yaml:91、:134-141、:622（invalid_input 列表中的 “integer scalar outside int64”）。
- 现状与证据：
  - PyTorch 2.10 接受 [2^63, 2^64) 内的 Python int。TorchDispatchMode 记录到 `aten.add.Tensor(f64[1], 9223372036854775808)`，返回 9.223372036854776e18（uint64 Scalar）。只有 ≥2^64 或 <−2^63 时才抛 OverflowError。
  - bool 标量同样合法：`tensor([1.]) + True == 2`。
  - 草案把这两类都判为 invalid_input。按 CONTRACT_PROTOCOL §5，invalid_input 只用于违反文档写明的条件。
- 要求：改为 unsupported，或者按收到的精确 uint64 / bool 值处理。domain 与决定文字一并改。
- 复核测试 `test_issue_A2_legal_scalars_are_not_invalid_input` 以 expectedFailure 记录这个问题，修复后应转为通过。

**A3：AR.SCALAR_VALUE 的备选列得不完整，后果写得不够直白（只改文字）。**
- 位置：yaml:127-144。
- 现状：决定写 “None of these pages says whether the scalar's value is first converted”，但没有引用 T_TENSOR_ATTR 中 “floating point scalar operand has dtype torch.get_default_dtype()” 这句。这句可以读成第四种备选 (d)：标量取默认 dtype（float32）的值。对 bf16/f16/f32 张量，(d) 与 (c) 结果相同；对 f64 张量则不同。
- 要求：
  - 把这句写进依据条目（T_TENSOR_ATTR.PROMOTION 的 items 需要用 `--items-only` 补上），并说明为什么不选 (d)：这句话只用于决定结果的 dtype 类别。
  - 在决定里直接写明后果：eager 实际采用读法 (c)，所以对 f32 张量，eager 自身相对 RN(f) 会出现最多约 1 ulp 的偏差（证据 §3 中 0.044715、0.7978…、0.3535… 三项）。这部分偏差算作 K−G。
- 取 (a) 本身是诚实的：CONTRACT_PROTOCOL §4.1 规定按实际收到的值解释，备选也已作为诊断读法保留。

**A4：证据文字过时（只改文字）。**
- 位置：evidence/arith_mathfn_index_domains.md:37。
- 现状：仍写 “An integer scalar that cannot be converted to the dtype is invalid”。现行 AR.INTEGER / AR.INT_OVERFLOW 的规定是：只有结果或被选中的值越界时才是 domain_undefined（例子 E97、E103、E104）。
- 要求：改成与现行规定一致。

其余各项复核成立：
- AR.EXTENDED_REAL：inf−inf、0·inf（包括经由 alpha 产生的情形，例如 alpha=0 乘 inf）、inf/inf、x/0 判为 domain_undefined；fma 也一样。其余组合用复核方自己的扩展实数模型逐格比对（7×7 值网格 × 9 种运算，另加 alpha 网格），全部一致。
- AR.POW / AR.ZERO_POW：整数指数、负底数、±inf 底数、0^0=1、|n|≤64 的边界、非整数指数 unsupported、0 的负幂 undefined，全部一致。
- AR.INTEGER / AR.INT_OVERFLOW：五种整数 dtype 的上下界，以及 add/sub/rsub/mul/neg/abs/sign/clamp/maximum 的越界判定，全部一致。仅在被选中时越界的标量，其判定与决定相符。
- 选用 domain_undefined 而不是取模回绕，理由充分：文档没有规定越界行为，C++ 中有符号溢出是未定义行为。

## MATHFN

- 只对 sin/cos/tanh 解除幅度上限，理由成立：三者在全体实数上有定义且值有界；exp、expm1、sigmoid 仍保留上限，理由也成立。
- Arb 求值经复核方独立验证。验证方法：自写 Arb 1200 位球，要求它完全落在适配器区间内，并要求适配器区间的相对宽度小于 2^-200；另用 mpmath 3000 位点值作数值旁证。
- 覆盖的点：
  - 记录的 RoPE 角度（f32 [1,4,4]，按 inv_freq 生成，最大 127）；
  - tanh 在 50、−269、1e30 处；
  - f32 最大值、f64 最大值、2^1000+2^948、355、103993；
  - 2^-149、2^-1074、2^-300 等极小值；
  - exp/expm1/sigmoid/log1p/tanh 在 2^-55…2^-200 处（mpmath iv.exp 曾在这一带失效）。
- 宽度：sin(f64max) 区间宽约 7e-80。tanh(50) 的上端点小于 1，sin(355) 的符号正确。
- **O1（观察，不阻塞，属于既有缺口）：MATHFN 适配器不检查输入能否用其 dtype 表示。** 顶层已验证版本也接受 `{t:f32,v:['1/3']}`。上限解除后，像 2^1000+1 这样不可表示的巨大输入，经 320 位 Arb 转换只能得到 [-1,1]：结果仍然严格成立，但没有信息量。建议今后补上与 ARITH/INDEX 相同的可表示性校验（invalid_input）。

## INDEX

- 整数 gather / index_select 原样搬移各 dtype 的上下界值；记录的 bert int64 缓冲区（值 >2^53）结果精确。
- padding_idx 的处理：
  - 原生 −1 或 0..V−1：前向不变；
  - 原生其他值：unsupported；
  - 公开接口 None 或 [−V,V)：负值规范化，前向不变；
  - 公开接口越界：invalid_input；
  - max_norm、freq、sparse、整数权重：unsupported。
- 以上全部与复核方用 NumPy 构造的期望一致。

## 复核测试与命令

```
cd /data1/tzh/kernel-analyzer/5_specification   # 先 export HOME/TMPDIR/... 与 PYTHONPATH
/data1/tzh/envs/hf_ga_5190/bin/python -m pytest -q -p no:cacheprovider tests/test_arith_mathfn_index_a2_review.py
  # 33 passed, 1 xfailed (A2), 1287 subtests passed
/data1/tzh/envs/hf_ga_5190/bin/python -m pytest -q -p no:cacheprovider tests/test_arith_mathfn_index_domains.py
  # 19 passed, 153 subtests passed（作者测试）
/data1/tzh/envs/liger/bin/python tools/check_contracts.py --contracts /data1/tzh/kernel-analyzer/.cache/spec_a2_review/D/contracts \
  --adapters /data1/tzh/kernel-analyzer/.cache/spec_a2_review/D/adapters --only ARITH MATHFN INDEX
  # ARITH 104/104 例子、10/10 读法、38/38 变异；INDEX 36/36、8/8；MATHFN 30/30、10/10；exit 0
```

## 晋升时需要更新期望的既有测试

做法：把三对草案替换到 `.cache/spec_a2_review/D/sim` 中的规格副本里，状态改为已审，复制到 validated，再运行 21 个相关测试文件。结果 8 项失败，均来自下表；其余全部通过。`test_bulk_reference::test_cast` 的失败是因为复制副本时 CAST.yaml 正被他人编辑，与本包无关，在当前工作树中可以通过。

| 测试 | 原期望 | 新期望 |
|---|---|---|
| test_arith_profile_domains.py `test_unreviewed_types_nonfinite_values_and_mixed_dtypes_are_refused` | `mul({i64,[1]},2)`、`mul({f32,[inf]},2)` 被拒 | 分别为 value `[2]`、value `[+inf]`；b8、f8e4m3、complex64 仍被拒 |
| test_arith_profile_domains.py:93 `test_selected_functional_schemas_reject_other_overloads` | `pow({f32,[1]},2)` unsupported | value `[1]`；可以改用 `pow(x,2.5)` 来保留 unsupported 的断言 |
| test_index_read_only_profile.py:43 `test_source_values_must_be_finite_physical_values` | `index_select({i64,[1]},0,[0])` unsupported | value `[1]`；b8、complex64、f32 inf 仍为 unsupported |
| test_index_read_only_profile.py:64 `test_indices_options_shapes_and_ranges_are_strict` | `embedding([[1,2],[3,4]],[0],padding_idx=0)` unsupported | value `[[1,2]]` |
| test_embedding_public_call.py:47-51 `test_mutation_and_nondefault_public_tasks_are_refused` | 公开接口 padding_idx −1、0 unsupported | 两者都是 value `[[3,4],[5,6]]`；max_norm、norm_type=1、scale_grad_by_freq、sparse 仍为 unsupported |

A1、A2 修复后，复核测试中 expectedFailure 的那一项要改成普通测试，A1 被跳过的那些组合按新决定补上断言。

## 补充：ARITH 定向复核（2026-10-21，作者修订 A1–A4 后）

本节对象文件：

| 文件 | sha256 |
|---|---|
| ARITH yaml | 4c16bd34f1ec820ba8fe355c24f2c0a9c42122c2cbdaaabfab799bad6a7e0283 |
| ARITH adapter | 796fa9fcac5fd51e1304c026c7639aa6d64700f29eb910e3b32e4bb4c8b5b3aa |

MATHFN、INDEX 已按前文确认的哈希晋升。

- **A1 已关闭。** 新增决定 AR.FLOOR_INFINITE：
  - 正确引用 T_DIV 快照第 24–25 行（Python `//`、np.floor_divide）与 T_FLOOR_DIVIDE 的 floor(x/y)；
  - 列出三种备选，选择 (c) unsupported；
  - 零除数先判，仍为 domain_undefined；trunc 模式与真除法不受影响。

  适配器 `_no_infinite_floor_operand`（:256-266）在 `_div` 的 floor 模式和 `_floor_divide` 中调用，与决定一致。复核测试的 7×7 网格现已断言：floor 模式和 floor_divide 遇到无穷操作数一律 unsupported，零除数优先判 domain_undefined。例子 E105–E107 的期望正确。
- **A2 已关闭。** `_received`（:139-170）的分类：
  - [−2^63, 2^64) 内的整数按精确值接受，uint64 段与 eager 的分派记录一致；
  - bool 标量与该范围之外的整数为 unsupported；
  - 非 binary64 的有理数（如 1/3）仍为 invalid_input。

  整数张量配 uint64 标量时，只在结果越界时判 domain_undefined（例如 i64 + 2^63 越界，−2^63 + 2^63 = 0 正常）。pow 指数或 clamp 边界取 True 时为 unsupported。E108、E109 正确。

  残余文字小瑕疵（不阻塞）：适配器 invalid 报错文字仍写 “nor an int64 integer”，与决定措辞不同，可在下次修改时顺手改。
- **A3 已关闭。** 新来源条目 T_TENSOR_ATTR.SCALAR_DTYPE 与快照第 97–99 行原文相符。决定已补上：备选 (d) 及否决理由（该句只用于决定类别与结果 dtype）、“eager 采用读法 (c)、在记录的 f32 调用上出现最多约 1 ulp 偏差、计入 K−G” 这一后果，以及 bool 与超范围整数为 unsupported 的规定。
- **A4 已关闭。** 证据第 37 行已与 AR.INT_OVERFLOW 一致。
- 复核测试已更新：A1 网格改为断言 unsupported；新增 `test_recheck_floor_infinite_and_scalar_classes`；作者已去掉 A2 的 expectedFailure，该测试现为普通测试并通过。

**ARITH 结论：CONFIRMED。** 上表两份文件可以晋升，并可放入 `validated/`。

命令与结果（新暂存目录 `.cache/spec_a2_review/D/recheck`：ARITH 用草案，MATHFN/INDEX 用已晋升的顶层文件）：
- `check_contracts --only ARITH MATHFN INDEX` 退出码 0；ARITH 109/109 例子、10/10 读法、38/38 变异；
- `tests/test_arith_mathfn_index_a2_review.py`：35 passed，1342 subtests；
- 作者测试 `tests/test_arith_mathfn_index_domains.py`：20 passed，157 subtests。

ARITH 晋升时需要更新的既有测试：以当前工作树为底，在模拟晋升环境里运行 11 个引用 ARITH 的测试文件，另加 alpha、where 广播、bulk 测试。失败的只有下表两个测试（共 3 处断言，其中 2 处是第一个测试的子测试），其余 122 项测试通过。

| 测试 | 新期望 |
|---|---|
| tests/test_arith_profile_domains.py `test_unreviewed_types_nonfinite_values_and_mixed_dtypes_are_refused`（:45-54） | `mul({i64,[1]},2)` 为 value `[2]`；`mul({f32,[inf]},2)` 为 value `[+inf]`；b8、f8e4m3、complex64 仍为 Invalid/Unsupported |
| tests/test_arith_profile_domains.py `test_selected_functional_schemas_reject_other_overloads`（:93-94） | `pow({f32,[1]},2)` 为 value `[1]`；若要保留 unsupported 断言，可改用 `pow(x,2.5)` 或张量指数 |
