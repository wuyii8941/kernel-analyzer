# 合同编写规范（规格 v3，2026-10-16）

适用于 `contracts/` 下的全部新合同。样板是 `contracts/ARITH.yaml` 与 `adapters/ARITH.py`；
已审阅的 `CE.yaml`、`ADAMW.yaml`、`RANKMEAN.yaml` 是 schema 1.1 的样板。
方法与口径见 `METHOD.md`、`CONTRACT_PROTOCOL.md`、`PLAN_v3.md`。

## 1 一份合同由什么组成

| 文件 | 内容 |
|---|---|
| `contracts/<ID>.yaml` | schema 1.2：来源依据、profile、决定、条款、例子、域、变异要求、非主张、修改记录 |
| `adapters/<ID>.py` | `reference(profile, inp) -> Result`、`MUTANTS = {变异编号: 函数}`、`READINGS = {读法: 函数}` |
| `SOURCES.json` 中的条目 | 由 `tools/fetch_sources.py` 写入；原文快照存在 `sources/` |

schema 1.2 在 1.1 之上增加的字段：
- `review.author`：起草人；
- `covers`：这份合同对应的算子名，如 ATen 名、torch 接口、库函数；
- `readings`：诊断读法，不是任务定义；
- `changes`：修改记录。

新合同一律写：
- `review.status: drafted`；
- `activation.implementation_status: pending`。

## 2 来源

- **每条条款、决定的 `basis`** 都必须解析到 `SOURCES.json` 的某个 `来源编号.条目`，或本合同的一条决定。检查器会拒绝解析不到的依据。
- **取得来源**：先把清单写到 `.cache/spec_v3/sources_<组名>.json`，再运行

  ```bash
  /data1/tzh/envs/ka_main/bin/python tools/fetch_sources.py .cache/spec_v3/sources_<组名>.json
  ```

  - 网页存正文文本，并记下原始 HTML 与文本的 SHA-256；
  - torch 源文件用 `"file": "torch/..."`，存原文件；
  - 已登记的条目要改 `items`，用 `--items-only`；
  - 不要手改 `SOURCES.json`：几个人会同时写它，工具带文件锁。
- **条目 `items` 的写法**：读过快照原文之后再写。只写原文说了什么，可以转述，但不能加原文没有的内容。
- **来源的优先次序**：
  1. 版本钉死的官方文档：PyTorch 2.10 `docs.pytorch.org/docs/2.10/...`；
  2. 标准（IEEE 754、OCP）：付费标准写 `kind: standard`、`edition`，按章节引用；
  3. 定义该算法的论文（arXiv 摘要页）；
  4. 参照程序：库源码、模型实现。
- **参照程序的身份**：来源条目写 `kind: reference_program`。条款依据只有参照程序时，结论只能说「与参照程序一致」，在 `nonclaims` 里写明。

## 3 原文没规定或有歧义的地方

- **写成决定**：写进 `decisions`，`status: drafted`，`statement` 里列出所有备选和取舍理由。
- **几种读法都合法时**：分成几个 profile。由任务方选的，写 `task_choice_required: true`；一个读法是诊断用的，放进 `readings`。
- **不能为了让某个实现通过而选读法**，也不能把被选中的读法说成唯一答案。
- **原文对某种输入不作要求时**：结果写 `domain_undefined`，并在决定里说清后果，例如「无要求」或「设备返回什么都不判」。

## 4 结果类别

| 类别 | 用于 |
|---|---|
| `value` | 精确值、严格包围、允许集合（期望写 `{set: [...]}`）、声明近似 |
| `domain_undefined` | 合法输入，但实数目标无定义：0/0、空集均值、全 −∞ 行的 softmax 等 |
| `invalid_input` | 违反文档写明的形状、类型、范围条件 |
| `unsupported` | API 合法，但不在任何 profile 内。**测过的变体不得落在这里**（PLAN_v3 V3） |
| `not_established` | 预算内达不到声明的保证 |

- 适配器抛出 `kernel_spec.num` 里的 `Invalid`、`Undefined`、`Unsupported`、`NotEstablished`。
- 其他异常都算实现错误。不要用 try/except 把实现错误转成数学结果。

## 5 例子

- **覆盖**：每条条款至少一个例子。
- **测过的边界情形**都要有例子：空维、size-1 维、广播、并列、−∞ 行、不是块大小倍数的形状、零梯度、None 梯度等。
- **期望值必须独立得到**：用手算或 sympy 推导，写成精确表达式，例如 `log(2)`、`1/(1+exp(1))`、`'7/2'`。**不得运行适配器取值**。写错了检查器会报矛盾，这正是它的用途。
- **期望值的语法**：
  - 允许：整数、`'p/q'`、`exp log sqrt tanh sin cos erf log1p expm1`、`pi`、`e`、`inf`、`-inf`、布尔值；
  - 嵌套列表表示张量；
  - `{set: [...]}` 表示允许集合。
- **输入的写法**：
  - 张量写 `{t: f32, v: [...]}`，元素用整数、`'p/q'`、`inf`、`-inf`、布尔；
  - 用 ATen 语义的合同，输入写 `{op: aten.xxx, args: [...], kwargs: {...}}`，见 `kernel_spec/aten_adapter.py`。

## 6 变异要求

- **列出典型误读**，包括：
  - `specification_v2/SPECIFICATION_v2.md` §3 列出的 v1 错误；
  - `4_bugs_cases/README.md` 里与本角色相关的缺陷；
  - 文档最容易读错的地方。
- **每条误读要有见证例子**：误读的输出要在指定输出上与期望矛盾，或者给出不同的结果类别。
- **误读由独立代码实现**：不要只给参照加一个开关。抛异常不算被拒绝。

## 7 适配器

- **先校验输入，再计算**：用 `kernel_spec.num` 的 `rational / integer / vector / matrix`。不截断、不丢多余项、不静默改形状。
- **数值部件**：
  - `kernel_spec.real.R`：精确有理数、严格包围、±∞，无定义处抛 `Undefined`；
  - `kernel_spec.aten`：ATen 运算的实数语义；
  - `kernel_spec.v24.module("spec_xxx")`：v2.4 参照，结果用 `v24.value / v24.tree` 转换。
- **输出**：每个声明的输出都要返回，类型为 Fraction、`mpmath` 区间、`R`、布尔，或 frozenset（允许集合）。
- **不得修改共享代码**：`kernel_spec/`、`tools/`、`tests/` 与别人的合同都不要改。需要共用的辅助代码，放在自己的适配器里，或者新建 `kernel_spec/<组名>_*.py`。

## 8 检查

```bash
cd /data1/tzh/kernel-analyzer/5_specification
export HOME=/data1/tzh/kernel-analyzer/.cache TMPDIR=/data1/tzh/kernel-analyzer/.cache/tmp
/data1/tzh/envs/liger/bin/python tools/check_contracts.py --only <ID> ...
```

- 退出 0 才算完成。
- 不要运行 `--lock`。

## 9 覆盖说明

每组写 `corpus/bindings/<组名>.json`，说明清点表（`corpus/instances.jsonl.gz`）中属于本组的每种算子由哪份合同的哪个 profile 覆盖：
- 测过的参数组合要全部列到；
- 还缺的照实写，并说明原因。
