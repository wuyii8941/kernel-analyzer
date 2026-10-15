# 合同层

- *.yaml：每个角色一份合同。字段见 TEMPLATE.yaml。
- adapters/<ID>.py：把合同绑定到可执行参照（reference），并按误读的文字描述独立实现误读（mutants），不调用规格代码。
- contract_check.py：逐例判定 成立 / 矛盾 / 未建立，检查每条误读是否被拒绝，输出条款追溯表。

条款状态：例子全部成立且相关误读全部被拒绝 → established；有矛盾 → contradiction；有误读未被拒绝 → weak；其余 → open。

规则：
1. 每条条款写明依据（原文引用编号或决定编号）和保证标签。
2. 例子由审阅人按条款手算，写成精确有理数或闭式表达式（log、exp、sqrt、e）。
3. 读法是有依据的备选解释，不是误读；在例子的 readings 字段里给出各读法的期望值。
4. 合同的 status_of_contract 只有在审阅人（不是参照代码作者）对照原文审过后才能改为 reviewed。
5. 改代码不改合同；改合同要重新审阅。
