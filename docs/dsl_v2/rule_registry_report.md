# 规则注册表报告（DSL v2 分支，Triton 3.6.0 回归 profile）

由 `scripts/dsl_v2/build_rule_registry.py` 生成；条目键为（操作、属性、类型、子区域）。general-v3.1 的注册表（results/general/rule_registry.json）冻结不改。这只是旧 profile 的规则账，不是 rc3 官方目标面的覆盖率。

状态计数：{'SUPPORTED': 398, 'REJECTED': 99, 'DECLARED_PREMISE': 3, 'NOT_ESTABLISHED': 2}

| 类别 | 支持 | 声明前提 | 未建立 | 拒绝 |
|---|---|---|---|---|
| A | 52 | 0 | 0 | 15 |
| B | 8 | 0 | 0 | 0 |
| C | 20 | 0 | 0 | 1 |
| D | 209 | 0 | 0 | 2 |
| E | 28 | 3 | 0 | 1 |
| F | 61 | 0 | 0 | 2 |
| G | 2 | 0 | 0 | 3 |
| H | 15 | 0 | 2 | 71 |
| I | 3 | 0 | 0 | 4 |

## 声明前提（交审阅方）

- `tt.reduce` / other：Any other combine region is interpreted step by step with interval semantics along the combination order of the locked lowering, read from the captured TTGIR layout (sequential within a thread in register order, butterfly over lanes, then over warps; a warp-synchronous result is the hull over its lanes).  The target is order-specific (DSL v2 rc3 02 6.2).  Premise for the reviewer: the order model (checked bit-exactly against the device for float sums by the emulator; the operand order inside a combine follows the lowering source).  No TTGIR layout -> not established, never guessed.
- `tt.scan` / generic fold (one operand)：The combine region is folded in scan order with interval semantics; exact in the reals only if the combiner is associative, which Triton's associative_scan requires but this code does not check (premise for the reviewer).
- `tt.scan` / generic fold (several operands)：The combine region is folded in scan order with interval semantics; exact in the reals only if the combiner is associative, which Triton's associative_scan requires but this code does not check (premise for the reviewer).
