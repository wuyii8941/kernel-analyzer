# 路径 A2 协议：补齐之后的算子入口重跑（2026-10-21 登记）

目标：用户要求补齐路径 A，「能诊断就行，不需要能找到所有缺陷」。见 `PATH_A_CLOSURE_PLAN.md`。

基线：路径 A（`OPERATOR_ENTRY_PROTOCOL.md`，F_A = c09b655，偏差 D-A1…D-A5），结果在
`sufficiency/results/PATH_A_20261020/`。基线不覆盖、不重算。

## 1 不变的对象

- 被测实现：FlagGems 5.0.2，注册在 CUDA 分派键上。
- 环境：torch 2.10.0+cu128，Triton 3.6.0，RTX A6000（sm_86），环境 `hf_ga_5190`。
- 工作负载：与 A 相同。
  - W1：bert-tiny / gpt2 / Qwen2.5-0.5B，fp32 和 bf16 各一，截为 2 层、128 token、一步训练；
  - W2：触发 D3、D5、D18、D20、D8。
- D-A3 的运行失败清单（`run_failures_20261020.json`）不变。

## 2 与 A 的差别

| 项 | A | A2 |
|---|---|---|
| 基础排除 | `copy_`、`to_copy`、`index` | 无。记录器快照、保存，以及测量捕获的设备复制都用 cudaMemcpy，不经过 ATen 分派（`reference_eval/capture.py`） |
| 记录 | `.cache/operator_entry/A_20261020/rec` | 重新记录，`.cache/operator_entry/A2_20261020/rec`；11 个工作负载 T1 全部一致 |
| 捕获上限 | 200 MB | 1.5 GB；memory_format 以标签保存 |
| 合同 | 2026-10-20 时的已准入合同 | 加上经独立复核后晋升的扩展（第 3 节） |
| 参照求值 | 合同适配器 | 大张量走 `kernel_spec/bulk.py`（与适配器差分测试）；`kernel_spec/real.py` 的 exp/log/tanh/sigmoid 改用 Arb |
| 程序参照 | — | libdevice `__nv_powif`/`__nv_powi` 有了语义 |
| 生成规则 | v1：单一张量返回 | v2：tuple 逐位置映射（`ATEN_RETURNS`）；写回 self 的原地算子；非语义参数（device、layout、pin_memory、memory_format、non_blocking）重放但不交给合同；条件参照（`conditioned_on_returns`）；整数返回不测量；转换类任务用 rounding_check 程序参照 |
| 预算 | 600 s | 每键 max(1800, 4e-6 × 标量运算数) s，上限 8 h |
| 第二步 | 只报告「需要规格匹配」 | 若 `task_choices_A2.json` 记录了选择，就按该选择求值，类别为「按记录的任务选择可用（第二步）」，并带出选择原文。两条选择（CAST 舍入、POOL 除数）由执行方提出，**待用户确认** |
| 结论规则 | E4 规则 | v2：E4 规则，另加两条记为「检出」——K 与 f 的特殊值类别矛盾（D-A5），返回张量形状与规格不同 |
| 类别 | 6 类 | 另加：「参照不完整」，「按记录的任务选择可用（第二步）」，「被测实现运行失败（重放）」 |
| 运行失败 | 排除后不评价 | G12：对排除的函数，在其记录的调用上单独重放（子进程）：崩溃记为运行失败；能运行就照常求值 |

## 3 合同版本

全部经过独立复核，复核记录在 `reviews/2026-10-20/A2_package_*_review.md`；晋升时由
`sufficiency/operator_entry/promote_draft.py` 核对复核时的 sha256。

| 合同 | 复核记录 | 晋升提交 |
|---|---|---|
| CAST（整数、布尔源） | package F | f16ed75 |
| NORM aten_native，NLL（新） | package A | d6ae106 |
| CREATE（新），RANDOM native_forward_given_mask | package E | 67f5995 |
| POOL torch210_aten_avg2d，INTERP torch210_aten_upsample_nearest2d | package C | efa0db0 |
| EMB aten_dense_backward | package A | f03a6ba |
| MATHFN，INDEX，LAYOUT，ATTN | packages D、C、B | 803e082 |
| ARITH | package D | 见第 7 节 |

## 4 已知且登记的不能诊断项

- SDPA efficient 反向在 dropout_p > 0 时（bert-tiny、gpt2 各 1 键）：掩码由 philox 随机流在核内重新生成，文档没有定义这个流，
  也没有返回掩码供条件参照（ATTN AT.ATEN_RNG）。这两个键记为「合同范围外」，原因写明。
- 规格侧的覆盖限制按各合同原样报告，例如 INTERP 无 scale 时比例不能在 binary32 中精确表示，以及 CREATE 的 arange 拒绝带。

## 5 报告

报告内容同 A：类别键数与调用加权数、Wilson 区间、按算子与按工作负载分组、W2 逐例。另加：

- **A→A2 逐键对照**：每个 A 的键在 A2 中归入哪一类；按缺口编号 G1…G12 汇总；
- **检出逐项列出**：Gπ−f 的最大相对差，以及来源判断（舍入常数、标量读法、真实缺陷）；
- **按任务选择得到的结论**：单独列出，等用户确认选择后才算最终结论；
- **限制**：
  - 测量层只比较值，不核对返回 dtype（package E 复核指出），布尔结果按 1/0 比较；
  - 单步固定输入，单元相同；
  - 生成计划的「已审阅」指规则已审阅。

## 6 不主张

- 不主张发现全部缺陷，只主张每个候选键有三态结论，或者有登记的不能诊断原因。
- 不覆盖 Inductor 融合 kernel（路径 B）。

## 7 偏差

（提交后追加，不改上文。）
- 登记提交 F_A2 = 25873b1（绑定开始前）。
- 偏差 D-A2-1（运行中发现的两处工具问题，2026-10-21）：
  1. 生成规则 v2 把重放用的 memory_format 解码成 torch 对象后再保存，torch.save 失败（`ones_like` 键）；
     改为保存记录时的标签形式。
  2. 元组输出中出现形状不同时，绑定只报「omits or adds promised fields」。结论规则 v2 本来就要把形状不同记为检出，
     但识别不了这条笼统的报告。现在 `call_binding` 给出两边的形状，结论规则 v2 据此识别。
     测试：`tests/test_call_binding_conditioned.py::ShapeDifference`。
  修复后的 `call_binding.py` 同步进本次运行的临时规格树，`instantiate.py` 和 `bind_key.py` 由此后启动的任务直接读取。
  修复前已完成的受影响键移到 `failed_harness_A2/`，在本次运行结束后重绑。
  另：bert-tiny bf16 key 53（`native_layer_norm_backward`）在并行运行中报告输入 a3「被改变」，
  单独重测和直接调用都没有复现。按「观察到但未复现」报告，不当作已证实的缺陷。
- 偏差 D-A2-2（重放改变了参数所在设备）：
  - 记录器把数值复制到主机保存，没有记下原设备；生成的调用一律用 `map_location="cuda"` 读取。
  - 后果：memory-efficient attention 反向的 philox seed/offset（原本在 CPU）被搬到 CUDA，FlagGems 把设备指针当主机内存读，导致段错误（bert-tiny fp32 key 73）。这是工具造成的，不是 FlagGems 的缺陷。
  - 处理：
    - `scan_devices.py` 无数值地重跑每个工作负载，从结构键取得各参数的设备，写出 `rec/<w>/devices.jsonl`（键编号与算子都与记录一致才写）。11 个工作负载中只有 2 个键含 CPU 张量：bert-tiny fp32 和 gpt2 fp32 的 efficient attention 反向。
    - 生成规则 v2 按记录的设备放置张量。
    - 今后的记录在 keys.jsonl 中直接写设备。
- 偏差 D-A2-3（运行中发现的工具与生成问题）：
  1. `check._within_input_mean_total` 调用了不存在的 `iv.sum_bounds`，只有重复启动结果不同时才会走到（FlagGems 用原子加的 embedding_backward）。修复前测试失败：`2_tool/tests/test_within_input_mean_total.py`。
  2. `contract_binding` 要求每个输出映射都被测量，没有表示「声明不测量的位置」的写法，导致元组返回无法评价（flash 前向）。现在 `literal: null` 加非空 basis 的条目可以不测量。修复前测试失败：`test_contract_binding.py::test_declared_unmeasured_position_*`。
  3. 没有张量输入的工厂调用（arange、zeros 等），测量要求至少一个声明输入：改为给调用一个不使用的输入，在 `unused_inputs` 中写明 basis。
  4. 适用性探针在条件输入（来自返回值）存在时无法求值：探针改为跳过这些输入，profile 是否覆盖该算子仍由算子本身判定。
  5. A2 下，第一次执行（含规则 v2 探测返回位置是否为 None 的那次）都放在独立进程中，崩溃记为运行失败，不会使绑定进程退出。
  修复前已完成的受影响键，在本次两遍运行结束后移到 `failed_harness_A2/` 并重绑。
- 偏差 D-A2-4（任务选择由用户决定，2026-10-21）：用户决定，文档没有说清的读法以 PyTorch 官方源码为准。核对 v2.10.0 源码后，两条原先「待确认」的选择都与源码一致。
  - CAST：转 bfloat16 用 `round_to_nearest_even`，转 float16 用 IEEE 转换 `fp16_ieee_from_fp32_value`，都是 roundTiesToEven。
  - POOL：CUDA 与 CPU 核的除数都是先把窗口截到 padding 后的范围、再截到输入之前算出的 `pool_size`，即 PA.DIVISOR。
  - 源码作为参照程序保留：TS210_BFLOAT16_H、TS210_HALF_H、TS210_AVGPOOL2D_CU、TS210_AVGPOOL_CPU。
  - `task_choices_A2.json` 改为「用户已决定」。按这两条选择得到的结论，含义是「与 PyTorch 官方实现的读法一致」。
  - 此前用「待确认」措辞绑定的键，在收尾时按新措辞重绑。
