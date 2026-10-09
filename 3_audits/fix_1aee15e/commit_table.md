# 提交逐项表（4156422..1aee15e，32 个提交）

由 `commit_table.py` 从 git 生成；文件路径是提交当时（整理前）的路径，当前位置见 `CURRENT.json`。「发现相关代码」按 diff 块所在函数匹配七项发现涉及的函数，是定位线索，不是逐行审阅结论。

| # | 提交 | 日期 | 说明 | 改动文件 | 测试文件 | 发现相关代码 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `3be713e8b0` | 2026-10-09 | DSL v2 increment 1 (tool 4.0, unreleased): generic combine regions and execution validity | 18 ((root) 3, docs 3, results 1, scripts 1, src 6, tests 4) | 4 | F05: evaluate_sequence |
| 2 | `6718f00c36` | 2026-10-09 | DSL v2 increment 1, part C: repeated launches in check.run; regression runner | 4 (docs 1, scripts 1, src 1, tests 1) | 1 | F04: torch_intermediates; F05: run; F07: _execution_status, _within_input_mean_residual |
| 3 | `a355614753` | 2026-10-09 | DSL v2 increment 1: regression results on the v1.1 programs; W0 official inventories started | 42 ((root) 1, docs 2, results 37, scripts 2) | 0 | — |
| 4 | `3d94b9ef03` | 2026-10-09 | DSL v2 increment 2 (part): trigger tracing, rule contracts baseline, bounded route and design sensitivity | 12 (docs 1, results 3, scripts 1, src 5, tests 2) | 2 | F04: run_level, torch_intermediates; F05: run; F06: expand |
| 5 | `50b0606075` | 2026-10-09 | DSL v2 increments 2-3: per-signature evidence, five evaluator defects fixed, W0 registry, main TTIR import | 18 (docs 1, results 4, scripts 5, src 4, tests 4) | 3 | — |
| 6 | `23939a6764` | 2026-10-09 | DSL v2 increment 3: dot_scaled, contention-free CAS, unsigned atomics; eight-class calibration; increment 4 registered | 13 (docs 2, results 4, scripts 2, src 3, tests 2) | 2 | F01: _int_op, _op_atomic_cas; F02: _op_dot_scaled |
| 7 | `94f530f09c` | 2026-10-09 | DSL v2 increment 4: atomic return values (point / set target), bit-view atomics, order laws; increment 5 registered | 18 (docs 2, results 2, scripts 3, src 6, tests 5) | 4 | F01: _unsigned; F04: run_level; F05: run |
| 8 | `f3d8a57b2c` | 2026-10-09 | DSL v2 increment 5: atomic load/store/poll with happens-before, histogram, approximate division; results of increments 2-3 | 21 (docs 3, results 11, scripts 2, src 4, tests 1) | 1 | — |
| 9 | `00fd9db9f5` | 2026-10-09 | DSL v2 increment 6: tuple arguments, argmax over NaN, Bessel functions, narrow bit casts, map_elementwise | 12 (docs 2, results 1, scripts 2, src 4, tests 3) | 2 | — |
| 10 | `2e306813af` | 2026-10-09 | DSL v2 increment 6 evaluation fixes: constexpr tuples and host tensor descriptors in the capture | 2 (scripts 1, src 1) | 0 | — |
| 11 | `c79ded136b` | 2026-10-09 | DSL v2 W1 contracts after increment 6: 335/418 supported entries with three test categories and trigger evidence | 2 (results 2) | 0 | — |
| 12 | `954e9af95a` | 2026-10-09 | DSL v2 increment 7: CAS spin locks by serialization with reverse-order evidence; vector-clock happens-before | 3 (docs 1, src 1, tests 1) | 1 | F03: _cas_reverse_order |
| 13 | `126c398cf2` | 2026-10-09 | DSL v2 increment 8: W1 evidence for 83 signatures; three evaluator defects and two soundness gaps fixed | 9 (docs 2, results 1, src 5, tests 1) | 1 | F03: _check_scan_bracketing |
| 14 | `8861752dbc` | 2026-10-09 | DSL v2 increment 8: the lane hull states why it leaves a reduction result not established | 1 (src 1) | 0 | — |
| 15 | `27237ff161` | 2026-10-09 | DSL v2: results of increments 4-6, W0 reconciliation with an enumerated registry, W1 contracts 416/419 | 28 (docs 2, results 23, scripts 3) | 0 | — |
| 16 | `53d754b30a` | 2026-10-09 | DSL v2 increment 9: bit-level inline PTX with constraints, packing and several outputs | 9 ((root) 1, docs 3, src 4, tests 1) | 1 | — |
| 17 | `45dc5e7566` | 2026-10-09 | DSL v2 increment 7 results: tutorial 05 layer-norm backward complete under the CAS order premise; captures archived | 6 (docs 1, results 5) | 0 | — |
| 18 | `16bd4080eb` | 2026-10-09 | DSL v2 increment 9 results: inline asm captures 80/81 complete (fp8e4b15 dot and conversions), name coverage 5949/5952 | 6 (docs 2, results 4) | 0 | — |
| 19 | `7610f0867c` | 2026-10-09 | CURRENT.json: DSL v2 development block lists increment 9 | 1 ((root) 1) | 0 | — |
| 20 | `91f285fa70` | 2026-10-09 | DSL v2 increment 10 registered: Gluon importer (TTGIR), shared memory, async copies and mbarriers | 1 (docs 1) | 0 | — |
| 21 | `12cc500f3f` | 2026-10-09 | DSL v2 increment 10: deviation recorded before implementation (shared memory ordered by the compiler's Membar pass) | 1 (docs 1) | 0 | — |
| 22 | `cba4ee75e0` | 2026-10-09 | DSL v2 increment 10: Gluon importer from TTGIR, shared memory, async copies and mbarriers | 14 ((root) 1, docs 3, results 5, scripts 1, src 3, tests 1) | 1 | F05: evaluate_sequence |
| 23 | `40bb9160d5` | 2026-10-09 | DSL v2 increment 11 registered: NVIDIA Hopper / Blackwell TTGIR modules (CPU semantics, device validation pending) and the TTIR / TTGIR cross-check | 1 (docs 1) | 0 | — |
| 24 | `a8e2f940a7` | 2026-10-09 | DSL v2 increment 11: NVIDIA Hopper / Blackwell TTGIR modules (device validation pending) | 5 (scripts 1, src 3, tests 1) | 1 | — |
| 25 | `fffc0a12dc` | 2026-10-09 | DSL v2 W1 complete (420/420); a store through an undefined address now invalidates its target at once | 4 (results 2, src 1, tests 1) | 1 | F05: evaluate_sequence |
| 26 | `eae796812f` | 2026-10-09 | DSL v2 increment 11 results: TTIR and sm_90 / sm_100 TTGIR references agree on 7.3M elements (0 disjoint) | 14 ((root) 1, docs 2, results 8, src 2, tests 1) | 1 | — |
| 27 | `a92ac31ed7` | 2026-10-09 | DSL v2 increment 12 registered: AMD gfx942 / gfx950 TTGIR modules (buffer ops, scaled upcast, in-thread transpose) | 16 (docs 2, results 9, scripts 4, src 1) | 0 | — |
| 28 | `681beda5e2` | 2026-10-09 | DSL v2 increment 12: AMD gfx9xx TTGIR modules (device validation pending) | 6 (results 1, src 3, tests 2) | 1 | F02: _op_scaled_upcast_fp4, _op_scaled_upcast_fp8, _scaled_upcast |
| 29 | `96fc2ed9a3` | 2026-10-09 | DSL v2 increment 12 results: TTIR and AMD gfx942 / gfx950 TTGIR references agree on 7.3M elements (0 disjoint) | 11 ((root) 1, docs 2, results 8) | 0 | — |
| 30 | `c9472ad03b` | 2026-10-09 | DSL v2 increment 13 registered: tc_gen5_mma_scaled (Blackwell scaled MMA; device validation pending) | 1 (docs 1) | 0 | — |
| 31 | `d8639997c3` | 2026-10-09 | DSL v2 increment 13: tc_gen5_mma_scaled (Blackwell scaled MMA; device validation pending) | 5 (src 3, tests 2) | 1 | F02: _op_tc_gen5_mma_scaled, _scaled_dot |
| 32 | `1aee15e9df` | 2026-10-09 | DSL v2 increment 13 results: 5 official scaled-MMA kernels agree with their TTIR (0 disjoint); sm_90 / sm_100 names 129/129 | 7 ((root) 1, docs 2, results 4) | 0 | — |
