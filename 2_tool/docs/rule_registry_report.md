# 规则注册表报告（DSL v2 分支，Triton 3.6.0 回归 profile）

由 `scripts/dsl_v2/build_rule_registry.py` 生成；条目键为（操作、属性、类型、子区域）。general-v3.1 的注册表（results/general/rule_registry.json）冻结不改。这只是旧 profile 的规则账，不是 rc3 官方目标面的覆盖率。

状态计数：{'SUPPORTED': 426, 'REJECTED': 93, 'DECLARED_PREMISE': 4, 'NOT_ESTABLISHED': 1}

| 类别 | 支持 | 声明前提 | 未建立 | 拒绝 |
|---|---|---|---|---|
| A | 54 | 0 | 0 | 15 |
| B | 8 | 0 | 0 | 0 |
| C | 20 | 0 | 0 | 1 |
| D | 230 | 0 | 0 | 2 |
| E | 29 | 3 | 0 | 0 |
| F | 61 | 0 | 0 | 2 |
| G | 5 | 0 | 0 | 0 |
| H | 16 | 1 | 1 | 71 |
| I | 3 | 0 | 0 | 2 |

## 声明前提（交审阅方）

- `tt.reduce` / other：Any other combine region is interpreted step by step with interval semantics along the combination order of the locked lowering, read from the captured TTGIR layout (sequential within a thread in register order, butterfly over lanes, then over warps; a warp-synchronous result is the hull over its lanes).  The target is order-specific (DSL v2 rc3 02 6.2).  Premise for the reviewer: the order model (checked bit-exactly against the device for float sums by the emulator; the operand order inside a combine follows the lowering source).  No TTGIR layout -> not established, never guessed.
- `tt.atomic_cas` / *：Contended compare-and-swap (DSL v2 increment 7, rc3 02 6.9 declared order + evidence): the launch is evaluated in every program order when it has at most 4 program instances, otherwise in program order and reverse program order, each from the memory before the launch; only elements every evaluated order establishes and agrees on keep a value (hull).  For a spin lock (acquire loop + exchange release, one critical section per program) the commutativity certificate replays every program's critical section symbolically and z3 proves the sections commute pairwise: the result then holds for every serialization (unconditional).  Otherwise the values are complete under the premise (proof status axis, audit F03), never in the unconditional complete class.  Happens-before through release / acquire with vector clocks; relaxed locks keep racing.  Premise for the reviewer when no certificate: agreement of the evaluated orders stands for every serialization (evidence, not a proof).
- `tt.scan` / generic fold (one operand)：The combine region is folded in scan order with interval semantics and checked against a second bracketing (Hillis-Steele doubling) on the same inputs; prefixes the two disagree on are not established.  Exact in the reals only if the combiner is associative (the tl.associative_scan precondition): when z3 proves the combine region associative for all arguments (reals / bit-vectors, finite scanned values) the results are unconditional; otherwise agreement on these inputs is evidence, not a proof, and the results are complete under the premise (proof status axis, audit F03; premise for the reviewer).
- `tt.scan` / generic fold (several operands)：The combine region is folded in scan order with interval semantics and checked against a second bracketing (Hillis-Steele doubling) on the same inputs; prefixes the two disagree on are not established.  Exact in the reals only if the combiner is associative (the tl.associative_scan precondition): when z3 proves the combine region associative for all arguments (reals / bit-vectors, finite scanned values) the results are unconditional; otherwise agreement on these inputs is evidence, not a proof, and the results are complete under the premise (proof status axis, audit F03; premise for the reviewer).
