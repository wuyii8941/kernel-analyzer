# Operator Oracle v0.1：一页说明

## 一句话

> 对一个明确的 operator 和一批明确的真实输入，先规定哪些结果是合法的，再检查 compiled 是否落在合法范围内；证据不足就说不知道，不能用“差得不多”冒充正确。

## 1. Oracle 首先问什么

不是先问 eager 和 compiled 差多少，而是先问：

```text
对于输入 x，规范允许这个 operator 产生哪些结果 S_o(x)？
```

- shape、index、boolean、argmax 等通常有精确或有限合法结果；
- top-k 遇到 ties 时可能有多个合法结果；
- 浮点 sum/matmul 可以有由舍入、精度和数值分析确定的合法误差包络；
- sampling 的合法对象是概率分布，不是某一次 token；
- optimizer/AMP 的合法对象是完整下一状态，不只是参数 norm。

如果无法说明合法范围来自哪里，就不能做 correctness 判定。

## 2. eager 扮演什么角色

eager 默认只是 baseline，不是真值。

```text
compiled - eager              implementation discrepancy
compiled - independent truth  truth-relative error
```

只有高精度结果、数学规范、API 规范或已确认 wrong-code relation 才能支持 correctness。单纯和 eager 不同，最多说明 compatibility/reproducibility 改变。

## 3. bias 和 variance 在哪里

先定义合法行为，再解释 compiled/eager 差异：

| 名称 | 直白含义 |
|---|---|
| average relative bias | 在指定真实输入总体上，compiled 相对 eager 是否长期偏向某个方向 |
| input heterogeneity | 这个方向/大小是否随 input、shape、dtype、checkpoint 改变 |
| runtime variability | 完全相同 input 和配置重复执行，结果是否仍改变 |
| sampling uncertainty | 我们观察的 inputs/repeats 有限，因此对总体判断有多不确定 |

固定 reduction tree、固定 reassociation 或 cast placement 通常属于确定性 input-dependent effect，不是 runtime variance。浮点计算也不自动等于 variance。

这些量解释“差异怎样产生”，但不自动决定“是否正确”。

## 4. Oracle 输出什么

每个结果保留三个轴：

```text
Conformance：是否违反规范/数值包络
Discrepancy：bias、input heterogeneity、runtime variability
Impact：是否改变事件、update 或下一状态
```

可能出现：

- conformance 通过，但相对 eager 有稳定 bias；
- conformance 失败，但当前 workload 没有观察到 impact；
- correctness 无法判断，但 compatibility 明显改变；
- operator 本身正确，只是把上游误差转换成 argmax/top-k 变化。

这些结论并不矛盾。

## 5. 五种 verdict

```text
UNINSTANTIATED：没有定义合法范围/容许边界
INVALID：输入、执行路径或 operator 身份证据无效
ACCEPT：有效证据完整落在合同允许范围内
REJECT：有效证据确认违反合同
INDETERMINATE：证据跨过边界，当前无法判断
```

“没有显著差异”通常对应 `INDETERMINATE`，不是自动 `ACCEPT`。

## 6. operator、region 和 kernel

- operator：我们要判断的语义计算；
- region：几个计算组成的联合编译/干预边界；
- kernel：实际执行 artifact。

三者不能互换。若 fusion 后只能确认一个 region，就只能给 region verdict。单独 replay 一个 operator，只能说明 isolated implementation，不能自动说明它在原始 fused program 中的因果贡献。

## 7. 最小结果卡

```text
Subject:       哪个 operator instance/signature
Population:    哪批 nominal/stress inputs
Contract:      exact / numerical / distributional / transition
Truth level:   specification / high precision / baseline only
Allowed set:   S_o(x) 和 population acceptable set
Identity:      operator / isolated operator / region / unidentified
Verdict:       ACCEPT / REJECT / INDETERMINATE / INVALID / UNINSTANTIATED
Claim:         correctness / compatibility / impact
Explanation:   bias / heterogeneity / runtime variability / tails
Coverage:      这个结论明确覆盖什么、不覆盖什么
```

## 8. 三个常识例子

### 例 1：top-k ties

eager 和 compiled 返回不同 tied indices，但两者都属于 API 允许集合：correctness `ACCEPT`，paired reproducibility 可以不同。

### 例 2：固定 reduction 差异

完全相同 input 重复执行结果不变，但 compiled 相对 eager 有固定差异：runtime variability 为零，relative bias/heterogeneity 可能非零。若结果仍在分析误差包络内，numerical conformance 可以通过。

### 例 3：sampling

相同 seed 得到不同 token，不足以说明 sampling law 错误；必须判断目标概率分布是否改变。若 API 没有承诺相同 RNG algorithm，单 token mismatch 只是 coupling/reproducibility 现象。

## 9. 当前能说什么

现在已经有：

- 明确的 Oracle 判定函数；
- exact、numerical、distributional、transition 四类合同；
- PyTorch API 的真实 contract records；
- sum/matmul 的条件化数值包络；
- operator/region/kernel 身份和因果降级规则；
- 系统反例与验证标准。
- confirmed broken/fixed cases 和部分 precommitted case-family evidence，包括 forward raw delta 排序失效、eager 错而 XLA 对、以及两边零差异但共同错误的真实案例；原 v0.1 验证合同尚不是字段齐全的 executable preregistration manifest。
- 分别在输出前冻结的真实 controls：CUDA sum 的 eager/compiled delta=2 且 default allclose 失败，但两者都在分析包络内；multinomial 100 draws 因置信集合跨越 ±1% law boundary 而返回 `INDETERMINATE`。

现在还不能说：

- 所有 PyTorch floating operators 都有已验证 tolerance；
- 所有 fused operators 都能做 operator-level attribution；
- Oracle 已经完成 coverage-balanced held-out bugs/controls 的统一评估，或普遍优于 raw delta。

因此当前准确名称是：

> **逻辑和框架语义已经建立、部分合同可直接实例化且已有初步真实增量证据的 Operator Oracle v0.1；尚待完整 held-out coverage 验证。**
