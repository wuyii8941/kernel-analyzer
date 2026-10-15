# kernel_spec_package_2026-10-15 合同预审

## 决定

**方法通过。原三份 YAML 可以保留为草案，但原检查器不能作为认证或上线准入。采用 handoff 的定向补充后，可以交给 Codex 实施，不需要再重写方法。**

输入 SHA-256：`a9a2a9123c2f39f5f76d1c82d61d70869bc0ec8236cce16d978f9556293d15a3`。
本次是上传包审阅，不是 GitHub HEAD 审计。不声称重跑 GPU/全部工具/ORT。源文件位置以下均相对于原包根目录。

## 本次已运行

- 9 项关系、4 项舍入、105 项 phase1、89 项 phase2、15 项 phase3，共 222 项通过。
- 原合同检查复现 11 条 established、11/11 原有误读被拒绝的输出；该数字是原检查器结果，不是条款正确性的证明。
- 历史 SGD/BN 示例与撤回理论的检查运行。
- 在临时副本/内存 monkeypatch 上复現检查器问题，未改原包源文件。
- handoff 的三份合同：15 条条款、6 个 profile、29 个明确用例、20 条变异实施要求。结构校验与 9 项校验器回归通过。
- 直接调用原规格，22 个正例/数学无定义例通过，核查 58 个标量字段。另 7 个适配器前提/未支持用例待 Codex 实施。高精度点值核验与精确有理成员检查分开记录。

## 应保留的进展

来源、数学合同、可执行参照已经分开；三份试点不是继续堆角色数量；旧 THEORY 的强主张已经明确撤回。AdamW 四个基础公式与无平滑 CE/聚合例子的方向正确。现有 oracle 大部分代码可复用。

## 阻塞项与证据

### F1. 样例绿灯被升级成条款成立，草案/来源/实现绑定均没有门槛

位置：`contracts/contract_check.py:16-46`；`contracts/README.md:7`；`NEXT_STEPS.md:9,15`。
- `basis` 只检查非空，不解析引用；不存在的 `MISSING.D0` 仍成立。
- `status_of_contract=draft-for-review` 不影响 established；任务负责人未选择的 RANKMEAN 也成立。
- 替代读法 E5[R_B] 矛盾不进入条款状态。
- 没有将条款审阅与实际规格/适配器/后端哈希绑定。
处理：内容审阅、实现证据、query 接纳分别存储，不从样例数推导全域保证。数值相容结果只能按其实际证据等级使用。

### F2. 检查器和顶层脚本出现“失败仍返回成功”

位置：`contracts/contract_check.py:47-51`；`run_all.sh:3,6-9,15`。
- 将 CE E1 expected 改为 999 后报告 contradiction，但 exit code=0。
- 将 test_relations.py 替换为显式 exit(9) 的临时测试，`run_all.sh` 仍 exit(0)。原因是 POSIX sh 管道看 tail 的退出状态。
- 错读法没有影响整体退出码。
处理：错误非零退出，未知/未实施独立状态；不得以 tail 输出或日志中的 OK 代替完整运行结果。

### F3. 实际可核查接口小于合同声明接口

位置：`contracts/ADAMW.yaml:14-29` 与 `adapters/ADAMW.py:3-7`；`CE.yaml:2,14-18` 与 `adapters/CE.py:3-8`。
- ADAMW 声明输入前态和全部后态，但适配器总是零初始化，只返回 theta；添加 m/v/t 输入不会改变结果。
- CE 声明梯度/none/概率目标，适配器只返回标量 loss，hard-code ignore_index=-100。
- CE reduction=none 会因 list 转标量失败；概率目标没有传 prob_target=True，报 class-index 错误。
- 在内存中把全部 CE gradient 改为零，五条 CE 合同仍全部 established。
处理：补实际状态和完整 VJP 输出，或明确把未支持模式留在 pending。不能标题承诺完整、测试却只比较标量。

### F4. Domain 只是文字，计数/形状不验证

位置：`contracts/RANKMEAN.yaml:12-19`、`adapters/RANKMEAN.py:3-6`。
- n=[3.9,1] 被 int 转为 [3,1]，返回 2.5。
- S=[6,4], n=[3,1,1] 长度不一致仍返回 2。
- n=[-1,2]、n=[0,1] 配非零空-rank S 也返回数值。
处理：验证整数、非负、长度、成员与空分片。无证据的成员关系只能明确作前提，不伪称验证完成。

### F5. 比较器的“严格”标签缺乏依据

位置：`contracts/contract_check.py:8-14,34-35`、`adapters/common.py:9-18`。
- 有理数转为 mpf 丢失 exact 身份；Decimal/iv 端点经字符串及有限精度转换，未保证向外舍入。
- 即使 expected=1，而参照点严格等于 1+1e-51，仍 established。
- 甚至 lo=1+1e-52, hi=1-1e-52 的倒置区间也 established。
- 1e-50 的放宽和 1e-30 的宽度阈值是经验检查，不是严格等式证书。
- 使用 eval 解析合同期望，没有限制语法。
处理：按结果类型验证；保留精确端点；区间不相交可否定，相交只支持相容；限定公式 AST；数学无定义与允许集合分开。

## 来源及数学合同审查

### AdamW

核对官方 PyTorch 2.10 算法框：旧参数衰减、moment 不含衰减项、当前步偏差修正、eps 在 sqrt 外与 draft 一致。
原 draft 的 stable URL 与待锁定版本已在 handoff 改为 2.10。试点明确 minimize、amsgrad=False、dense、有限输入，未知前态不重置。None 与零梯度按同一官方页面的 zero_grad 说明区分。
来源：https://docs.pytorch.org/docs/2.10/generated/torch.optim.AdamW.html

### CE

核对同一版本的类别目标损失、权重分母、概率目标 mean 的 N 分母、ignore_index、label smoothing 及输出形状。
加权 smoothing 的组合读法不是把当前实现符合的解释都自动列为正确；handoff 保留 R_A 显式 profile，R_B 只作诊断。概率目标和 none/VJP 都给出具体合同与样例。
`exact` 可以用于索引/计数/明确有理关系，但不能把包含 log 的 mean loss 全体改称 exact。
来源：https://docs.pytorch.org/docs/2.10/generated/torch.nn.CrossEntropyLoss.html

### RANKMEAN

数学上，固定 token 等权目标给出 sum(S)/sum(n)。这是任务负责人/调用者的明确选择，不是所有通信或所有训练问题的默认真值。
HF 文章支持该任务示例，不能代替当前任务选择。合同还要要求传入的是 sums，不是 local mean；class-weighted CE 的 mean 分母和 token count 不是一回事。
来源：https://huggingface.co/blog/gradient_accumulation

## 非阻塞但需要纠正的口径

- 四种 guarantee 之外另设 outcome；undefined 不叫 allowed_set。
- `METHOD.md` 对 Gπ 的“精确实数”简称应保留 π 控制的格式/位/集合/效应语义。
- 历史证据不能单独认证，但可信分歧必须能暂停相关条款，不允许以“额外证据”之名忽略反例。
- 不要求先把三套区间库合为一个；先统一结果接口和保证传递，避免为试点引入全库重构风险。
- ADAMW 新 ID 是 ADAM 角色的子 profile，不能遗漏 coupled Adam 后宣称全角色迁移。
- 原 `SHA256SUMS` 把自身列入而校验失败，这只是核账缺陷，不据此认定 payload 损坏。打包时哈希清单排除自身、冻结原始 tar 哈希。
- ORT 与 Liger 声明转录旁证本次未复核执行，不计入本次通过数；来源版本和代码哈希之后按条款挂接。

## 这次直接补齐了什么

`handoff/` 是审阅后可供 Codex 实施的差分包：三份有来源/域/profile/全部输出/完整见证的合同、固定协议、实施任务、结构校验器和测试、实际见证核查代码与结果。
未直接修改旧数值参照和 production 适配器；这正是 Codex 下一步工作。合同结构校验成功不生成 runtime_ready，相关命令在要求运行准入时返回 2。

## 结论

**可批准按补充合同开展实现；不可批准用原 `11 条 established` 作为已完成认证。**
本轮目标已经足够明确：修检查器、接完整输出和域验证、核对三份合同，不再追加更大理论或迁移全部角色。
