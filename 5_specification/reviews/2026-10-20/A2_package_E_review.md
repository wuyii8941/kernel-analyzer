# 路径 A 收口 A2 包 E 独立审阅：CREATE（新合同）与 RANDOM（新 profile native_forward_given_mask）

审阅日期 2026-10-20/21。审阅人没有参与起草。审阅对象（sha256）：

| 文件 | sha256 |
|---|---|
| contracts/draft/CREATE_path_a_A2_20261020.yaml | c0720fe7a27d873fd35a3dfa3157d5f58a6c34f7241e0076ffa48cfd46c09675 |
| contracts/draft/CREATE_adapter_path_a_A2_20261020.py | ceac49c44894cda42bd08df4a180807c1726d0c4f51c609534e2dad5482f0efa |
| contracts/draft/RANDOM_path_a_A2_20261020.yaml | a7a6e771ec917227b159f2176dfaaee5df0e3747c97d0deefa98757b7809b2f7 |
| contracts/draft/RANDOM_adapter_path_a_A2_20261020.py | c8817f01714a33bb1e5a0364586f95bd6bfad1bfeced689be0e1ac27399301ab |

## 结论

- **CREATE：ISSUES**（CR-1 至 CR-3，都是小问题；语义主体成立）。
- **RANDOM：ISSUES**（RN-1 至 RN-3，都是小问题；新 profile 的语义成立，原有两个 profile 语义未变）。

两份合同都不能按当前哈希晋升，也不能进入 validated/。修完以后只需要定向复核：xfail 测试翻转成通过，文字修正逐条核对。

## 来源核对（逐条读过快照）

T_ONES_LIKE.DEF/DTYPE、T_ZEROS_LIKE.DEF/DTYPE、T_SET_DEFAULT_DTYPE.INITIAL（初始默认 float32、可改）、
T_EMPTY.UNINIT、TS_NATIVE_FUNCTIONS_CR.CREATION_SCHEMAS（第 793–818、2840–2856、4538–4553、4745、6747–6771 行）与
NATIVE_DROPOUT_SCHEMA（第 286 行，`bool? train`）、TS_RANGE_UTILS_H.CHECKS/SIZE（int64 用整数算式；其他 dtype 一律用 double
计算 ceil((end-start)/step)）、TS_TENSOR_FACTORIES_CPP.SCALAR_TENSOR、TS_REFS.SCALAR_TENSOR（按 Python 类型推断 dtype）、
TS_DROPOUT_CPP.NATIVE_CPU（train 无值或为真时 scale=1/(1-p)，1-p==0 时 scale 取 0，mask=Bernoulli(1-p) 布尔张量）、
TS_DROPOUT_CU.NATIVE_CUDA（train 为假时直接返回；p==1 返回零张量和全假 mask；train 无值按训练处理）、
TS_DECOMP.DROPOUT（`train and p != 0`，因此 train=None 按评估处理）。条目都与原文相符，没有添加原文没有的内容。
T_ARANGE 页面确有“非整数 step 与 end 比较时受舍入影响”的警告，CR.ARANGE_LENGTH 的依据成立。

## 重点问题的判断

1. **布尔结果用 1/0 编码（CR.BOOL_ENCODING、RN3.MASK_POSITION）：可以接受。** 理由：编码是一一对应的精确值，没有丢信息；
   决定列出了全部备选及取舍；输出声明写明了编码。不过有两点限制：
   (a) check_output 检查了 dtype 必须为 b8，但没有检查每个元素恰好是 True/False（见 CR-1、RN-1）；
   (b) 路径 A 实际走的测量路径（`2_tool/src/kernel_analyzer/contract_binding.py` 把参照值转成 binary64 数组，
   `run_path_a.py`/`instantiate.py` 都不调用 check_output）不比较输出 dtype。所以在路径 A 中，“观察结果必须是 bool”
   这一点不会被执行：被测算子如果返回 float 1.0，同样会通过。这是 harness 的共性缺口，不是合同本身的问题；
   但报告路径 A 结果时不能说已经核对了 dtype。建议 harness 把观察到的 dtype 与记录的返回 dtype 对比，转交协调方。
2. **RN3.CONDITIONED / SCALING：成立。** 把返回的 mask 当作参照输入，只主张“输出与 mask、缩放一致”，nonclaims 里写明不认证
   mask 的随机性。p 取收到的 double 的精确值，scale 1/(1-p) 也精确（0.1 的例子我独立复算过）。train=False、p=0 返回原输入和
   全真 mask；p=1 返回零张量和全假 mask。这与分解程序、CUDA 源码一致；CPU 源码在 p=1 时 scale=0、Bernoulli(0)，结果相同。
   train=None 时分解程序按评估处理，两个 kernel 按训练处理，判为 unsupported 有依据。
   确定性情形下提供的 observed_mask 会被忽略，参照返回文档规定的 mask，这一点正确。
3. **CR.ARANGE_LENGTH 的拒绝带：理由成立，实现正确。** 原因：参数都是 double 时，RangeUtils 的 double 求值在相对误差约
   2^-52 的范围内，2^-30 的带宽足够保守。我在 400 组随机 double 参数上验证过：带外的精确长度都等于 double 求值的长度；
   带内的参数都被拒绝。“整数且绝对值小于 2^53 时不设拒绝带”同样成立：|e-s| 小于 2^53 时，double 商不会越过整数。
   带宽保守，所以 arange(0, 0.3, 0.1) 这类情形也会被拒绝（eager 与精确值其实一致），这是选择的代价，已经写明。
4. **CR.DEFAULT_DTYPE / default_float32：成立。** explicit 拒绝依赖默认 dtype 的调用。default_float32 的标注是
   `task_choice_required: true`，决定状态是 needs_binding，三个备选都已列出。记录到的调用都不依赖默认 dtype。

## ISSUES

**CR-1** `CREATE_adapter_path_a_A2_20261020.py:283`：布尔比较写作 `1 if bool(v) else 0`，所以 b8 记录里的 2、0.5
都会被当作 true，与 CR.BOOL_ENCODING（yaml:127–130）“逐元素比较 1/0 编码”不符。
证据见 `test_bool_check_rejects_non_boolean_entries`（strict xfail）。修法：要求元素的类型是 bool，或数值恰好为 0/1。

**CR-2** `CREATE_adapter_path_a_A2_20261020.py:156–157`：step 为 ±inf 时判为 invalid_input。但 CR.ARANGE_LENGTH
（yaml:98–99）与 error_handling 只说 start/end 非有限时无效；RangeUtils.h 也只检查 start/end。PyTorch 2.10 eager 对
`arange(0,5,inf)` 返回空张量；精确公式取极限得 ceil(0+)=1，同样有歧义。现在的分类没有依据。修法：改成 unsupported，
或者在决定里写明备选和理由。证据见 `test_arange_infinite_step_class_recorded` 和 `test_eager_factories`。

**CR-3** `CREATE_adapter_path_a_A2_20261020.py:138`：报错文字引用了不存在的 profile `default_float32_numeric`，
实际名称是 default_float32。属于表述问题。

**RN-1** `RANDOM_adapter_path_a_A2_20261020.py:349`：与 CR-1 相同，`_check_native` 用 bool() 强转观察到的 mask，
b8 记录里的 2 会被接受，与 RN3.MASK_POSITION（yaml:119–120）不符。证据见 `test_native_check_rejects_non_boolean_mask_entries`（strict xfail）。

**RN-2** `RANDOM_path_a_A2_20261020.yaml:69–71`：RN2.SCHEMA 仍写着“Native dropout forward ... is a different task”，
yaml:157 的 RN2.C5 写的是“Both profiles”。现在同一合同已经覆盖 aten.native_dropout，也有三个 profile，应当注明这两处只适用于
deterministic_forward/finite_backward。yaml:245 的 nonclaim 缺一个句号（“...diagnostic only) In the conditioned case”）。
这些不影响语义，属于表述一致性问题。

**RN-3** `evidence/create_random_contracts.md:76`：“p = 1.5 raises (invalid_input)”只对 CPU 训练成立。复核结果：
CPU/CUDA 评估模式下 p=1.5 原样返回；CUDA 训练时 p=1.5 返回 -0 和全假 mask；CUDA p=-0.5 按 2/3 缩放并返回全真 mask。
把 p∉[0,1] 判为 invalid_input，可以靠文档“p 是概率”成立，但证据说明应当照实记录参照程序不一致，并在 RN3.SCALING
（yaml:110）中注明依据是“p 为概率”（T_DROPOUT），不是 kernel 检查。

## 原有 profile 是否未变

适配器的 diff 只增加代码，旧函数一行没改，只把 COVERS 改为读取同目录的合同。yaml 中旧条款、例子和变异原样保留。
我在 150 组随机输入上比较了 staged 草稿与顶层已验证的 `adapters/RANDOM.py`（deterministic_forward、finite_backward，
覆盖各种 dtype、秩、p、train、inplace、scale，包括无效值和越界值），结果类别和数值全部一致，check_output 也一致。

## 复核命令与结果

```bash
export HOME=/data1/tzh/kernel-analyzer/.cache XDG_CACHE_HOME=/data1/tzh/kernel-analyzer/.cache TMPDIR=/data1/tzh/kernel-analyzer/.cache/tmp \
       TRITON_CACHE_DIR=/data1/tzh/kernel-analyzer/.cache/triton PYTHONPATH=/data1/tzh/kernel-analyzer/2_tool/src
/data1/tzh/envs/hf_ga_5190/bin/python -m pytest -q -p no:cacheprovider 5_specification/tests/test_create_random_a2_review.py   # 92 passed, 2 xfailed (CR-1, RN-1)
/data1/tzh/envs/hf_ga_5190/bin/python -m pytest -q -p no:cacheprovider 5_specification/tests/test_create_random_contracts.py   # 20 passed
cd 5_specification && /data1/tzh/envs/liger/bin/python tools/check_contracts.py --contracts ../.cache/spec_a2_review/E/contracts \
       --adapters ../.cache/spec_a2_review/E/adapters --only CREATE RANDOM   # exit 0; CREATE 43/43, 7/7; RANDOM 32/32, 1/1, 13/13
```

审阅测试 `tests/test_create_random_a2_review.py`（sha256 531be10cf5283e2d4a4083cb32d48e01d97b8795bc9bf39b5bc80a93add76ffa）
的期望值全部用 Fraction/整数/struct 独立算出，没有调用适配器或 kernel_spec.aten 的辅助函数。内容覆盖：记录到的全部调用结构
（ones 按 128×128 全尺寸）、空维与 0 维、各种 dtype、拒绝类别、arange 整数与浮点随机性质、拒绝带与 double 求值一致性、
native_dropout 条件化随机测试（f32/bf16/f64、0–5 维）、确定性情形、check_output 正反例，以及 eager 旁证（[1,8,128] f32/bf16）。

## 复核（2026-10-21，作者修正之后）

新文件哈希：CREATE yaml 539fb3a757485288c641f14ca1b9ed296b3481d2417c5521ff1666fb6de4bce7、CREATE adapter
c084a03c711f25541f3be9e03d29e29352e9944ac6c24bc3fea0b77569b4d4f1、RANDOM yaml
4a330febfe670d4e736394f0816b7a75d0a41b8e638d4d6f51c64c143e1c0c0b、RANDOM adapter
9d7f16f2de81b6c624b7fafb0743b3df9e6ac17b259618cfa4e2874afca340ef、evidence ddaff0b47a76588249d5692fe6f4fc3df498282a69491f7ddad646aca763e6f2。
我逐行比对了修正前后的 diff，没有发现所列修正之外的改动。

- CR-1 / RN-1 已解决：观察记录中的 b8 元素必须是 bool 或 np.bool_，比较也不再用 bool() 强转。整数 1/0、2、0.5 都会被拒绝；
  np.bool_ 和 0 维布尔记录仍然被接受。新增测试 `test_boolean_fixes_accept_real_booleans` 防止 `v is True` 这一写法日后退化。
- CR-2 已解决：CR.ARANGE_LENGTH 增补了 (i) invalid_input、(ii) 长度 0、(iii) unsupported 三个备选，选择 (iii)，
  理由与 RangeUtils.h 快照、eager 行为一致；适配器改为 Unsupported，同时新增例子 CR.E44 和 domain 条目。
- CR-3 已解决：报错文字改成 default_float32。
- RN-2 已解决：RN2.SCHEMA 限定只适用于 deterministic_forward/finite_backward，指明 native forward 只由
  native_forward_given_mask 覆盖，状态退回 drafted（整份合同待审，这样处理合适）；RN2.C5 点名两个 profile；nonclaim 补上句号，
  第一条改为“不主张独立于返回 mask 的真值”，与 RN3.CONDITIONED 一致。
- RN-3 已解决：RN3.SCALING 写明依据是文档中的“p 为概率”，并照实写出 kernel 的不一致行为（CPU 训练在 bernoulli_ 处报错、
  评估模式直接返回、CUDA 训练照常计算）。这些都与 Dropout.cpp/Dropout.cu 快照和我的 eager 复核一致。evidence 第 80 行也同步改了。
- 作者对审阅测试的改动正确：去掉两个 strict xfail 标记（缺陷修好后这两个测试本该通过）；
  `test_arange_infinite_step_class_recorded` 的期望改为 unsupported，与 CR.E44 一致。

复核结果（新建 stage `.cache/spec_a2_review/E/`）：check_contracts 退出码 0，CREATE 44/44、7/7，RANDOM 32/32、1/1、13/13，
结论 consistent；`test_create_random_a2_review.py` 95 passed（sha256 edbb92082da8a119494f7a54b25f26d70898842942b1271ebad7292fdecba072）；
`test_create_random_contracts.py` 20 passed。

**最终结论**
- **CREATE：CONFIRMED**，按 yaml 539fb3a757485288c641f14ca1b9ed296b3481d2417c5521ff1666fb6de4bce7 与 adapter
  c084a03c711f25541f3be9e03d29e29352e9944ac6c24bc3fea0b77569b4d4f1，可以晋升并进入 validated/。
- **RANDOM：CONFIRMED**，按 yaml 4a330febfe670d4e736394f0816b7a75d0a41b8e638d4d6f51c64c143e1c0c0b 与 adapter
  9d7f16f2de81b6c624b7fafb0743b3df9e6ac17b259618cfa4e2874afca340ef，可以晋升并进入 validated/。
- 仍然有效的 harness 提示（不属于合同问题）：路径 A 的测量路径不比较输出 dtype，也不调用 check_output。因此报告路径 A 结果时，
  不能说“已经核对 bool dtype”。
