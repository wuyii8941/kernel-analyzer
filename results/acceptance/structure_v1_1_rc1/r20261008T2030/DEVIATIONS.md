# 运行 r20261008T2030 的偏离记录

冻结记录 `RUN_FREEZE.json`（提交 098fb74）之后的每一处改动都记在这里，按时间先后排列。规则、输入总体、种子、family 与统计方法都没有改。

## D1：补跑不需要 K_R 的比较对象（2026-10-08 23:10 前后，在 96 个作业全部结束、生成报告时发现）

- **现象**：`scripts/acceptance/structure_v11.py` 只给自动参照已建立的输出计算 F_total（K − f）和 baseline（K − ref_fp64）。
  主轮与复现轮里，5 个 T1 程序（prog_08、17、20、29、32）的 10 个输出参照未建立（语义缺失：`unrecognized reduction combiner`），
  于是这两类比较对象也没有算。两轮共 40 个登记 family 成了缺格。
- **为什么算偏离**：协议 §5 要求「声明不支持的性质不影响其他方法运行」；F 与 baseline 不依赖 K_R，本应照常运行。这是适配器的
  遗漏，不是工具或规则的变化。
- **处理**：新增 `scripts/acceptance/structure_v11_blackbox.py`，在冻结工具上补跑这 10 个输出：
  F_total 用冻结的 `check.run_black_box` 记录；baseline 用冻结测试框架的定义（点残差、R1/R2/R3/R5、不跑检测器），作用在同一次
  执行的 K 上；E 只做描述。种子、单位、Holm family（缺成员按 p = 1）都与 RUN_FREEZE 相同。结果放在 `supplement_D1/`，
  报告里注明来源。
- **补跑前已看到的东西**：这 10 个输出参照未建立及原因；其他程序的全部主轮与复现轮结果。还没看到 T1 的任何 K − f 或 K − f64 统计
  （冒烟测试只用了开发种子 0–7）。
- **不变的部分**：FR（K−G、G−f）对这 10 个输出仍是未建立；完整率不变。
- 另外：`scripts/acceptance/structure_v11_report.py`（只做汇总、不做测量）写于冻结之后，见报告首部说明。
