# 规则注册表报告（通用能力轮，Triton 3.6.0 锁定构建）

由 `2_tool/scripts/general/build_rule_registry.py` 生成；条目键为（操作、属性、类型、子区域）。枚举测试 `2_tool/tests/test_rule_registry.py`：锁定构建的每个操作都有条目、引用的测试都存在、入库 JSON 与代码一致、通用路径中没有按 kernel 名字做语义分支。

状态计数：{'SUPPORTED': 396, 'REJECTED': 102, 'DECLARED_PREMISE': 2, 'NOT_ESTABLISHED': 2}

| 类别 | 支持 | 声明前提 | 未建立 | 拒绝 |
|---|---|---|---|---|
| A | 52 | 0 | 0 | 15 |
| B | 8 | 0 | 0 | 0 |
| C | 20 | 0 | 0 | 1 |
| D | 209 | 0 | 0 | 2 |
| E | 27 | 2 | 0 | 3 |
| F | 61 | 0 | 0 | 2 |
| G | 2 | 0 | 0 | 3 |
| H | 14 | 0 | 2 | 72 |
| I | 3 | 0 | 0 | 4 |

## 声明前提（交审阅方）

- `tt.scan` / generic fold (one operand)：The combine region is folded in scan order with interval semantics; exact in the reals only if the combiner is associative, which Triton's associative_scan requires but this code does not check (premise for the reviewer).
- `tt.scan` / generic fold (several operands)：The combine region is folded in scan order with interval semantics; exact in the reals only if the combiner is associative, which Triton's associative_scan requires but this code does not check (premise for the reviewer).

## 拒绝（语义缺失）

- `tt.advance`  ：block pointers are not supported in this version
- `tt.atomic_cas`  ：compare-and-swap results depend on an undeclared interleaving
- `tt.cat`  ：tt.cat may reorder elements; the element order is not declared
- `tt.descriptor_gather`  ：tensor descriptors (TMA) are not supported
- `tt.descriptor_load`  ：tensor descriptors (TMA) are not supported
- `tt.descriptor_reduce`  ：tensor descriptors (TMA) are not supported
- `tt.descriptor_scatter`  ：tensor descriptors (TMA) are not supported
- `tt.descriptor_store`  ：tensor descriptors (TMA) are not supported
- `tt.dot_scaled`  ：scaled (microscaling) dot is not supported
- `tt.histogram`  ：histogram is not supported in this version
- `tt.make_tensor_descriptor`  ：tensor descriptors (TMA) are not supported
- `tt.make_tensor_ptr`  ：block pointers are not supported in this version
- `tt.map_elementwise`  ：map_elementwise regions are not supported in this version
- `tt.map_elementwise.return`  ：map_elementwise regions are not supported in this version
- `arith.addui_extended`  ：extended-precision integer add is not supported
- `arith.mulsi_extended`  ：extended-precision integer multiply is not supported
- `arith.mului_extended`  ：extended-precision integer multiply is not supported
- `arith.scaling_extf`  ：microscaling conversions are not supported
- `arith.scaling_truncf`  ：microscaling conversions are not supported
- `math.atan2`  ：atan2 is not supported in this version
- `math.ctlz`  ：bit counting is not supported in this version
- `math.ctpop`  ：bit counting is not supported in this version
- `math.cttz`  ：bit counting is not supported in this version
- `math.fpowi`  ：fpowi is not supported in this version
- `math.ipowi`  ：ipowi is not supported in this version
- `math.isnormal`  ：isnormal depends on the storage format; not supported
- `scf.execute_region`  ：execute_region is not produced by the Triton frontend
- `scf.forall`  ：not produced by the Triton frontend
- `scf.forall.in_parallel`  ：not produced by the Triton frontend
- `scf.index_switch`  ：index_switch is not supported in this version
- `scf.parallel`  ：not produced by the Triton frontend
- `scf.reduce`  ：not produced by the Triton frontend
- `scf.reduce.return`  ：not produced by the Triton frontend
- `cf.switch`  ：cf.switch is not supported in this version
- `tt.reduce`  other：unregistered combine region: reduction tree not declared, semantics not guessed
- `tt.extern_elementwise` symbol=other ：libdevice symbol without declared semantics
- `tt.elementwise_inline_asm` asm=other ：unregistered inline assembly
- gpu 方言 66 个操作：TTIR 阶段不出现，拒绝
