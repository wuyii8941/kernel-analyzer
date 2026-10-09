# B017：torch.compile 把共享同一代码对象的不同 torch 函数当成同一个（max_pool / adaptive_max_pool 等），静默复用错误的图

日期：2026-10-06。对象：PyTorch 2.10.0（`ka_main`）Dynamo；PyTorch main（2026-10-06 的 `torch/_dynamo/variables/torch.py`、
`guards.py`、`trace_rules.py`）相关代码未变。类：④（编译后的语义与 eager 不符），静默错误，与后端无关（`backend="eager"`
同样出错）。

## 现象

```python
@torch.compile
def apply(pool, x, k):
    return pool(x, k)

x = torch.randn(2, 3, 10, device="cuda")
apply(F.max_pool1d, x, 3)            # 正确
apply(F.adaptive_max_pool1d, x, 3)   # 形状 (2, 3, 3) 与 eager 相同，数值是 max_pool1d 的结果
```

`torch.nn.functional` 的 `max_pool1d/2d/3d`、`adaptive_max_pool1d/2d/3d`、`fractional_max_pool2d/3d`，以及 `torch.unique`、
`torch.unique_consecutive`、`torch.lu` 之间，只要通过参数、闭包、默认参数、模块属性或列表元素传进编译区域，
第二个函数就复用第一个函数的图。实测（`4_bugs_cases/bugs/repro/B017_*`，CUDA，Inductor）：

| 写法 | 结果 |
|---|---|
| 编译函数的参数 `apply(pool, x, 3)` | adaptive_max_pool1d 得到 max_pool1d 的数值（形状相同） |
| 模块属性：同一个 `nn.Module` 类的两个实例，构造时分别传入 pooling 函数 | 第二个实例编译后用第一个的 pooling |
| 闭包工厂 `make(pool)` | 同上 |
| 区域编译：双分支模型（同一输入的 max 与 adaptive-max 特征拼接），`m.a.compile(); m.b.compile()` | 输出错误（整体 `torch.compile(model)` 正确） |
| 对照：`torch.Tensor.add` / `mul`（方法描述符） | 同样错误，这是已报告的 pytorch#197811 |

## 机制

`trace_rules` 把这些函数列为图内函数（`TorchInGraphFunctionVariable`，`torch_non_c_binding_in_graph_functions` 里有
`"torch.nn.functional.max_pool2d"` 等条目）。它们从参数等来源进入时，`BaseTorchVariable.create_with_source` 对
Python 函数（`inspect.isfunction`）装 `CLOSURE_MATCH`，而 `CLOSURE_MATCH` 对 `types.FunctionType` 只检查 `__code__`
（设计目的是让同一函数的不同闭包实例共用一张图）。这些函数都由 `torch._jit_internal.boolean_dispatch` 生成，是同一个
内部函数 `fn` 的不同闭包（`if_true`、`if_false` 不同），`__code__` 相同，所以守卫放行，图内调用目标仍是第一次编译时的
函数对象。用户自己的闭包不受影响：用户函数被内联，闭包单元格的内容另有守卫；图内函数不内联，只靠这一条守卫。

守卫转储（`TORCH_LOGS=guards`）中对 `L['op']` 只有一条 `ID_MATCH: ___check_obj_id(L['op'].__code__, …)`，
指向 `_jit_internal.py:609` 的 `fn`。

在锁定版本里按 `trace_rules` 的全部 2566 个条目枚举图内的 Python 函数（584 个代码对象），不同函数共享同一代码对象的
组有 5 个（`pre-reorg-20261009:scripts/probes/probe_dynamo_guard_pairs.py` 与本节的枚举）：

- boolean_dispatch 的 `fn`：上面列出的 11 个公开函数，加 `torch.functional` 的 4 个内部变体；
- `torch.nn.modules.utils._pair/_single/_triple/_quadruple`（`nn.grad` 里也导出）：实测 `_pair` 之后传 `_triple` 同样复用图；
- `contextlib.contextmanager` 生成的 13 个（`torch.backends.cuda.sdp_kernel`、`cudnn.flags`、`cuda.nvtx.range`、profiler 等）；
- `typing_extensions.deprecated` 包装的 6 个（`torch.cuda.amp.custom_fwd/custom_bwd`、几个 memory 查询）；
- `torch.autograd.function._iter_*` 的 4 个内部函数。

后三组几乎不会作为值传入编译区域。对照：`functools.partial` 的不同关键字、`operator.methodcaller/itemgetter`、Enum 成员、
类、不同对象的绑定方法、numpy ufunc、dtype、staticmethod 都被正确守卫。

## 与已有记录的关系

- pytorch#197811（2026-09-20，开着）：闭包捕获不同的 `torch.Tensor` 方法描述符时复用图，原因是描述符分支不装守卫
  （`# Don't need to guard on wrappers`）。PR #197845（开着）给 `MethodDescriptorType` / `WrapperDescriptorType` 装
  `ID_MATCH`。**这一修复不覆盖本问题**：复现脚本 `--pr197845` 模拟该 PR 后，描述符对照恢复正确，pooling 的四种写法
  仍然错误。
- pytorch#197860（开着）：内联的绑定方法没有代码对象守卫，是另一条路径。
- 检索未见关于 boolean_dispatch 函数或图内 Python 函数只按 `__code__` 守卫的报告。

修法（`--fixed` 实测，五种写法全部恢复）：`create_with_source` 对图内的 Python 函数用身份守卫（`ID_MATCH` /
`FUNCTION_MATCH`）而不是 `CLOSURE_MATCH`——图内函数是模块级单例，身份稳定；`CLOSURE_MATCH` 的"同代码即同图"只对会被
内联、闭包另有守卫的用户函数成立。

## 发现过程（工具）

OpInfo × Inductor 前向全量筛查（773 个用例）里，`oi__rdiv__5`、`oi__rmul__5` 的 e_sem 相对 RMS 为 1.38 与 4.13（区间证实），
记录的 kernel 名是 `triton_poi_fused_add_0`。单独运行时 K = f，说明不是 lowering 问题，而是同一进程里前一个用例
（`__radd__`）的图被复用：我们的筛查每个进程跑多个用例、没有在用例之间 `torch._dynamo.reset()`，而 OpInfoCase 用同一个
闭包 `f` 包装每个算子。由此读到 Dynamo 的守卫代码，描述符一支是已报告的 #197811；顺着同一函数里的 `CLOSURE_MATCH`
分支，枚举所有共享代码对象的图内函数，得到 boolean_dispatch 这一组，再用上表的写法确认。

工具在这里的作用是报警（e_sem 远超舍入、kernel 名与算子不符）；之后的定位靠读 Dynamo 源码与静态枚举。同一进程里的
直接差分（compiled 对 eager）同样能发现这种错误；PyTorch 自己的 OpInfo 测试在每个测试之间重置 Dynamo，所以看不到。

**对筛查结果的影响**：同一缓存复用也污染了筛查本身。前向筛查的 2 个候选与 10 个报错用例在每个用例前重置 Dynamo 后全部
正常（`pre-reorg-20261009:results/tool_spec/opinfo_screen_rerun/`）。`pre-reorg-20261009:scripts/tool_spec_check.py` 已改为每个用例开始时 `torch._dynamo.reset()`；
onesample 与放宽容差两批中的报错与候选用例正在重跑（`opinfo_onesample_rerun/`、`opinfo_loose_rerun/`）。B014–B016 早已
用独立脚本单独复现，不受影响。

## 影响

把这些 pooling 函数当作值传递的编译代码：按配置选择 pooling 的模块（构造参数）、闭包工厂、函数式的模型定义，
以及区域编译（逐块 `compile`，同类块共用编译缓存）。结果静默错误；形状不同时通常会在下游报错，形状相同时
（如 `max_pool1d(x, 3)` 与 `adaptive_max_pool1d(x, 3)` 在长度 10 上）完全静默。

## 证据

`4_bugs_cases/bugs/repro/B017_repro_dynamo_shared_code_in_graph_functions.py`（三种模式：原样、`--pr197845`、`--fixed`），输出
`pre-reorg-20261009:results/tool_spec/final/dynamo_guard/B017_repro_output_torch2.10.txt`。

## 状态

检索未见报告。main 的 `create_with_source` 与 `CLOSURE_MATCH`（2026-10-06）相同，trace_rules 仍把这些函数列为图内函数，
`functional.py` 仍有 8 处 `boolean_dispatch`。**nightly 实测（2.15.0.dev20261005+cpu，`B017_BACKEND=eager`）：五种写法全部仍然错误**（`pre-reorg-20261009:results/tool_spec/final/dynamo_guard/B017_repro_output_nightly_cpu.txt`）；描述符对照也仍错（PR #197845 未合入）。上游 issue 草稿：`4_bugs_cases/bugs/upstream_drafts/B017_*.md`（由用户提交）。
