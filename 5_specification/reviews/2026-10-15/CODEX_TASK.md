# Codex：三份规格合同实施任务

## 固定决定

使用本包 `contracts/*.yaml` 与 `CONTRACT_PROTOCOL.md` 为本轮实施依据。方法不再改动；只做 CE、ADAMW、RANKMEAN，不扩规格全集，不开展全训练研究。
原包 SHA-256：`a9a2a9123c2f39f5f76d1c82d61d70869bc0ec8236cce16d978f9556293d15a3`。
已有 `specification_v2/algorithm_specs` 数学代码先复用；不要通过更改固定 expected 来使测试通过。

## 先读后实施

1. 读取实际仓库 HEAD 与当前规则，确认与上传包差异。原包的旧目录/理论/审查报告保留为历史，不覆盖已修的新代码。
2. 留存 SOURCES.json 指定的 PyTorch 2.10 官方文档条款；不存在的 source/decision ID 必须报错。原文与当前数学条款不符则提交差异，仅暂停相关条款。
3. ADAMW 是 catalogue 的 ADAM 角色下一个 profile，不是悄悄把 coupled Adam 的覆盖删除。RANKMEAN 的 token 等权、CE.R_A 都由运行时显式选择，未选不得进入对应语义裁决。

## P0：先修检查器，不要先迁移其他角色

- 修正 shell 管道吞退出码：使用无管道的 subprocess.run(check=True) 或 bash 的 set -euo pipefail；不能一边测试失败，一边 echo 全部通过。
- schema 检查必须解析 basis 到存在的来源/决定，识别引用冲突、缺例子、缺结果字段、无效保证类型、重复 ID。
- 合同草案不能靠样例通过自动升级审阅状态。内容审阅、实现验证、query 接纳分开。
- 分别记录主 profile 和诊断读法的结果；某读法矛盾不得只出现在日志尾部而仍返回成功。
- 不用 eval 执行 YAML；本包 expected 仅允许整数、有理式、exp/log/sqrt 的 AST。
- 保留 exact 值和区间端点；拒绝倒置、NaN、非有限和结构不完整的结果。不要把 Float point tolerance 用成严格相等。
- 辅助证据不直接签发认证，但有可靠矛盾必须暂停受影响结论。

必须先写反例再修：
A. expected=999 与原结果矛盾时退出非零。
B. 正确源 ID 改为不存在 ID，必须拒绝。
C. 测试程序 exit(9)，顶层任务不能 exit(0)。
D. 区间下界大于上界，即便差值非常小也不能通过。
E. 缺少 gradient、缺少一个状态字段、缺少输出行，必须失败。
F. 替代读法 expected 错误不能被主读法通过覆盖。
G. 未批准的任务 profile、未绑定的实现与过期哈希不能自动启用。

## P1：三个实际适配器

### ADAMW
- 输入从时间序列 g 的旧简写升级为显式 `state` 与一步 `gradient`。另提供 sequence wrapper 时只调用相同单步语义，不重写另一套规则。
- 接受前态 theta/m/v/t，校验全长度与数值域；返回所有后态。使用 amsgrad=False、maximize=False 的本次 profile。
- 按已锁定文档处理 None，与零梯度分别测。分母为零记有限实数目标无定义，不判设备必须返回某个特殊值。
- 跑 AW.E1–E6 及 AW.N1–N2，加入 mutation_requirements 所列误读和合法对照。

### CE
- 实现每个明确 profile 的目标类型、weights、eps、ignore_index、reduction 与 upstream。
- loss 输出的结构必须完整；梯度是指定 upstream 的 VJP。当前规格 none 的 grad 是 row-loss sum 的梯度，需要乘对应上游，不能直接当任意 VJP。
- 不再硬编码 ignore_index=-100。概率目标必须传 prob_target=True 并做 profile 域验证。
- R_A 是有依据的已选解释，R_B 不自动参与“挑最接近实现”的裁决。
- 跑 CE.E1–E11 与 CE.N1；尤其 gradient 置零的变异必须失败，loss 不变不能掩盖它。

### RANKMEAN
- 不得 int() 截短 token 计数。要求 S/n/participant_ids 长度一致、ID 唯一、n 非负整数，n=0 的 S=0。
- 输入明确为每个 rank 的 loss sum，不是本地 mean；成员/权重/一次贡献绑定不明则给条件性或未建立，不能假定已证。
- 全零分母是 domain_undefined。S/n 形状失配、负数和非整数是 invalid_input。
- 只声明聚合目标，不宣布整个 DDP 与单卡训练等价。CE mean 的权重分母与 token mean 要分别绑定。
- 跑 RM.E1–E5 与 RM.N1–N4。

## P2：实施后的验收与交付

- 新合同包含 29 个明确用例。本包审阅阶段已直接调用旧规格核对其中 22 个正例/无定义例；7 个适配器/域验证例仍待实现，必须全部执行，不得跳过后计通过。
- 新的 20 条变异要求逐项登记实际执行和结果。不用“某变异能编译”或“示例算过”替代语义区分证据。
- 跑原来的 222 项回归以及新检查器负例。保存完整日志、源码/合同/后端哈希、失败与未建立原因，禁止使用 tail 后的退出码。
- 用一个已绑定目标的模式 B 调用和一个无 f 的模式 A 调用核验报告传递。无 f 不阻断 K-Gπ，也不能虚报 f 已认证。
- 单独写明内容已审阅、实施已验证、来源未固化、任务选择待绑定等状态；不得汇总为 ground truth 全覆盖。
- ORT 非必需；缺该依赖继续其他工作。未跑的旁证不计入成功。数值模块整合和其余角色迁移不作为本轮验收要求。

## 完成标志

三份合同的真实接口与文本一致；规定观察不遗漏；错误测试不能假绿；所选 profile 和来源可追溯；正常与错误对照结果均合理。做到这些即可结束本轮，不再扩大哲学或理论问题。
