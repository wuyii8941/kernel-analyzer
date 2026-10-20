# 路径 A 收口 F 包（CAST 整数/布尔来源）独立复核

复核日期：2026-10-20/21。复核人：独立复核方（未参与起草）。

## 对象（sha256）

| 文件 | sha256 |
|---|---|
| `contracts/draft/CAST_path_a_A2_20261020.yaml` | `b32c80e69c1f71dfbc24fab06d4b4f30663c5a7b324894a430b1ee8a76aadff4` |
| `contracts/draft/CAST_adapter_path_a_A2_20261020.py` | `50020058e225e188e7ed30a710611703225831f21f88b5ce753cec21674a21e6` |
| 对照：`contracts/CAST.yaml` / `adapters/CAST.py`（已验证，与 `validated/` 相同） | `dbd2fe58…300b` / `b6c56852…e01f` |
| 复核测试 `tests/test_cast_integer_a2_review.py` | `0643a8944dc4c967ca9b067dfd2a91662d7017d87ed5d2948c9c0f83dbe8c0f7` |

## 结论：ISSUES（只涉及 YAML 文字；数值语义与适配器成立）

适配器（`50020058…`）按原样可以晋升。YAML 要先改下面 1–3 条文字，晋升时再把状态字段改为 reviewed / validated，所以晋升文件的 sha 本来就会变；修改后只需复核这几处文字。

## 问题

1. **`CAST_path_a_A2_20261020.yaml:146-147` 有残留空键。** CA.C4 下有 `nonfinite values: null` 和
   `target overflow and unreviewed dtypes or options are refused before semantic comparison: null`。原因是已验证 YAML 用了 flow
   映射，语句里的逗号把一句话切成了几个键（原文件里就有这个问题）。草稿重新序列化时把这两个键显式写成了 null，而 statement 本身已经补全。
   修法：删除这两行。不影响检查器，但已审阅的合同里不应留下这种垃圾键。
2. **`:53-54` CA.PROFILE 对 T_TO 的转述超出了原文。** "it refers to C++ type conversion rules" 写得像 Tensor.to 页对所有转换都引用了 C++ 规则。
   快照 `sources/T_TO.txt.gz` 只在一条 Note 里说过 "According to C++ type conversion rules, converting floating point value to integer
   type will truncate"，只讲浮点到整数。把 [conv.fpint]/[conv.double] 用到整数→浮点、浮点→浮点，是本合同自己的解释。应写明这一点，
   并且所引用的 T_TO.DEF 条目并不包含这句话。结论不受影响，因为 profile 本来就是任务选择。
3. **`:89-90` CA.INTSRC 对溢出边界的说法不准确。** 原文写"C++ calls an out-of-range conversion undefined and this profile makes no claim
   there"，意思是 C++ 的未定义区间正好对应 profile 拒绝的区间。实际上 f16 能表示的范围是 [-65504, 65504]，所以 65505..65519 在 C++ 下已经是
   "outside the range … undefined"。本 profile 对这些值仍然给出 65504（由测试和例子 CA.E22 确认），依据是 IEEE 754 roundTiesToEven：舍入后再判溢出（§4.3.1、§7.4）。
   这个读法是正确的，与浮点来源一致，但依据只来自 IEEE754/任务选择，不来自 C++。应改成："65505..65519 由 IEEE 754 舍入定义（C++ 对此未定义）；
   |n| ≥ 65520 舍入后溢出，判为 unsupported"，并在 basis 中加入 `IEEE754.OVERFLOW`。
4. （次要）**"u8 values are 0..255"（`:86`）没有来源条目。** `sources/T_TENSOR_ATTR.txt.gz` 写有 "8-bit integer (unsigned)"，但 SOURCES 中
   T_TENSOR_ATTR 只有 PROMOTION 一个条目。建议用 `fetch_sources.py --items-only` 增加 dtype 表条目后引用。不阻塞。

观察（不算本包问题）：拒绝类别依赖元素顺序。例如 i32→f16 的 `[65520, 2^31]` 得到 unsupported，`[2^31, 65520]` 得到 invalid_input。
已验证的浮点路径也是这样（`[70000,'1/3']` 对 `['1/3',70000]`）。建议以后统一为"先完成全部记录校验，再判 profile 范围"。

## 1 来源核对

- `CPP23_CONV_FPINT`（N4950 7.3.11）的 TRUNC/EXACT/INEXACT/BOOL 四个条目与快照逐句一致，没有添加原文以外的内容。
- `CPP23_CONV_DOUBLE`（7.3.10）的 EXACT/BETWEEN 与快照一致。
- 两个新来源缺少 `snapshot_sha256`/`raw_html_sha256` 字段，而 T_TO 有。这是工具版本造成的差异，不影响内容。
- CA.SPLIT、CA.F2I 的依据成立；CA.F2I 引用 T_TO.TRUNC 正确。CA.PROFILE 和 CA.INTSRC 的问题见第 2、3 条。

## 2 语义

- 整数按数学值处理，u8 无符号，布尔 0/1，表示得了就精确返回，否则直接一次 RNE 到目标，不经中间格式：都正确，并有独立测试。
- 无效与不支持的划分：越界整数、非布尔 b8、整数 dtype 里的布尔、非整数值、NaN/inf 判 invalid_input；整数/布尔/f8/complex 目标、f8/complex 来源、
  溢出、_to_copy 选项判 unsupported。与 CONTRACT_AUTHORING §4 一致。
- **保留 task_choice_required: true 是对的。** Tensor.to 页没有给出舍入方向，非精确的情形确实需要选择。CA.SPLIT 已如实写明：所记录的
  Qwen position-ids 调用的值是 0..127，在 f32 中精确，结果与舍入选择无关。需要提醒协调方：`task_choices_A2.json` 里的 CAST 选择仍写着
  "pending the user's confirmation"。
- bulk 快速路径（`kernel_spec/bulk.py::_cast`）对非浮点来源会 `_Decline`，整数来源总是走适配器，不会出现 float64 双重舍入。

## 3 独立实现

自写 IEEE 754 RNE 编码器：Fraction 通用版，外加一个"枚举两个相邻值"的整数版。两者对 4000 个随机整数互相一致；f64 结果另与 CPython `float(int)` 对照，CPython 的整数转浮点是正确舍入的。
覆盖内容：
- u8、i8 到四种目标全部穷举；i16 到 f16/bf16 穷举；
- 各目标精度边界 2^(p-2)..2^(p+3) 附近 ±9 的整数和精确中点；
- 2^24、2^53、2^63 附近的点名例，以及 bf16 的双重舍入见证 2^31+2^23+1；
- 随机 i64/i32 到 bf16/f32/f64；
- f16 的 65000..65519（以及负值）返回值，65520 及以上判 unsupported；
- 布尔、记录的调用（三种 dtype 写法）、空形状；
- 全部拒绝类别；
- 浮点→浮点：16 种组合各 150 个随机值（含次正规数），草稿与已验证适配器逐坐标相同，也与编码器相同；随机拒绝输入的类别也相同。
没有发现任何坐标不一致。

## 4 test_cast_profile_domains.py 晋升后的期望

作者说晋升时需要更新的两个子例确实需要改，正确的新期望是：
- `{"t":"i64","v":[1]}`→f32 期望 **value [1]**；
- `{"t":"b8","v":[1]}`→f32 期望 **invalid_input**，不是 value，因为 1 不是布尔值；如果要测接受，应改用 `[True]`→value [1]。
其余断言（f8/complex 来源与各类目标、inf、70000→f16、NaN）在草稿下保持不变，已验证。

## 命令与结果

```
S=/data1/tzh/kernel-analyzer/.cache/spec_a2_review/F   # 草稿对拷贝为 $S/contracts/CAST.yaml、$S/adapters/CAST.py
cd 5_specification && /data1/tzh/envs/liger/bin/python tools/check_contracts.py --contracts $S/contracts --adapters $S/adapters --only CAST
#  CAST | drafted | 28/28 | 0/0 | 8/8 | consistent，exit 0
/data1/tzh/envs/hf_ga_5190/bin/python -m pytest -q 5_specification/tests/test_cast_integer_a2_review.py
#  19 passed, 556 subtests passed
/data1/tzh/envs/hf_ga_5190/bin/python -m pytest -q 5_specification/tests/test_cast_integer_sources.py 5_specification/tests/test_cast_profile_domains.py
#  19 passed, 104 subtests passed
```

## 复查（2026-10-21）：结论 CONFIRMED

作者只改了 YAML：
- 新 YAML sha256：`fbdc5ccdc79d01ea8e2719d8b09f0b1017a34a5797bca10017be76b7494602cd`；
- 适配器未变：`50020058e225e188e7ed30a710611703225831f21f88b5ce753cec21674a21e6`。

逐条核对：
1. CA.C4 的两个空键已删除，YAML 中不再有 `: null`；statement 完整。
2. CA.PROFILE 现在写明：Tensor.to 页只在浮点→整数截断处引用 C++ 规则；把 [conv.double]/[conv.fpint] 用到浮点目标是本合同自己的读法。
   这与 T_TO 快照一致，basis 也补了 T_TO.TRUNC。
3. CA.INTSRC 的范围判定改为以 IEEE 754 为准：先按 §4.3.1 做 RNE，舍入后再按 §7.4 判溢出。文中写明 65505..65519 在 C++ 覆盖的范围之外，
   但会舍入为 65504 并被接纳；|n| ≥ 65520 判 unsupported；不对 IEEE 754 给出的无穷作主张。basis 增加了 IEEE754.OVERFLOW。
   这些与独立测试一致。
4. 新条目 T_TENSOR_ATTR.INT_DTYPES 与快照 `sources/T_TENSOR_ATTR.txt.gz` 的 Integer dtypes 表一致（uint8 无符号；int8/16/32/64 有符号，别名 short/int/long；bool 为 Boolean），
   没有添加原文以外的内容。CA.INTSRC 已引用它。

晋升时把 review/activation 状态改为 reviewed / validated、更新 scope 里"not yet reviewed"一句，除此之外不改语义；
`test_cast_profile_domains.py` 按上文第 4 节更新。

复查命令（新建 stage `.cache/spec_a2_review/F`）：
- check_contracts：CAST drafted 28/28，8/8，consistent，exit 0；
- `test_cast_integer_a2_review.py`：19 passed，556 subtests passed；
- `test_cast_integer_sources.py` + `test_cast_profile_domains.py`：19 passed，104 subtests passed。
