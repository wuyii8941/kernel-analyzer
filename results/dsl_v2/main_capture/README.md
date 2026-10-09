# Captures under the official main build (DSL v2 W3)

Official Triton `e50b186e` (3.9.0+gite50b186e, built locally, env `/data1/tzh/envs/triton_main`), official unit tests copied
out of the source tree, run on an RTX 3090 (sm_86) with `scripts/dsl_v2/main_capture_plugin.py`: every captured launch
(at most 4 per test, operands up to 64 MiB) is evaluated by the tool-4.0 reference evaluator of the commit named below.
One JSON line per launch.  Measurement only; the official tests' own assertions are unaffected.

| file | tests (`-k`) | evaluator | run script |
| --- | --- | --- | --- |
| `inc3_test_core_subset.jsonl` | bin/unary ops, reductions, scans, where, casts, dot, atomic_rmw, masked loads, math, abs, umulhi | increment 3 (`50b0606` + uncommitted increment-3 work, committed as `23939a6`) | `w3_capture_run.sh` |
| `inc4_rerun_atomic_dot_fp8.jsonl` | atomic_rmw, dot*, abs_fp8 | increment 4 (`94f530f`) | `w4_capture_run.sh` |
| `inc5_sync_histogram_div.jsonl` | atomic_load_store*, atomic_poll*, histogram*, fdiv_approx, math_divide_op | increment 5 before the histogram deviation | `w5_capture_run.sh` |
| `inc5_histogram_after_deviation.jsonl` | histogram*, atomic_poll_waits_for_remote_cta | increment 5 after the deviation (docs/dsl_v2/increment_05.md §6) | `w5b_capture_run.sh` |
| `inc4_broad.jsonl` | every test of test_core, test_random, test_standard, test_libdevice, test_conversions, test_tensor_descriptor | increment 4 (`94f530f`) | `w4_broad_run.sh` |
| `inc4_tutorials.jsonl` | official tutorials 01, 02, 03, 05, 06, 08, 09 (first two launches per kernel) | increments 4-5 | `w4_tutorials_run.sh` |
| `inc6_broad.jsonl` | as inc4_broad | increment 6 (`00fd9db`) | `w6_broad_run.sh` |
| `inc6_rerun_after_capture_fixes.jsonl` | test_standard, test_tensor_descriptor, test_core -k "test_cat_nd or test_reshape or test_const" | increment 6 + capture fixes (`2e30681`) | `w6c_rerun.sh` |
| `inc6_combined.jsonl` | inc6_broad with the rerun's files / families replaced by their rerun rows | — | — |
| `inc7_atomic_cas_rmw.jsonl` | test_atomic_cas, test_atomic_rmw | increment 7 (`954e9af`) | `w7_capture_run.sh` |
| `inc7_tutorial05.jsonl` | official tutorial 05 (layer norm, CAS spin lock in the backward pass) | increments 7-8 | `w7_tutorials_run.sh` |
| `inc9_inline_asm.jsonl` | test_inline_asm*, test_dot_max_num_imprecise_acc, test_typeconvert_upcast | increment 9 (`53d754b`) | `w9_capture_run.sh` |
| `inc10_gluon_test_core.jsonl` | official Gluon unit tests (python/test/gluon/test_core.py), evaluated from TTGIR | increment 10 | `w10_gluon_run.sh` |
| `inc11_cross_level.jsonl` | test_core dot / reduce1d / reduce2d / scan2d / where / cast: each launch's TTIR reference vs the references of the sm_90 and sm_100 TTGIR compiled from it (cross_level_plugin.py) | increment 11 (`a8e2f94`) | `w11_cross_run.sh` |
| `inc11_cross_level_tf32_after_fix.jsonl` | the tf32 subset after the cvt.rna.tf32.f32 rule | increment 11 | `w11_cross_run.sh` (KSEL) |
| `inc11_cross_level_combined.jsonl` | inc11_cross_level with the tf32 subset replaced | — | — |
| `inc12_selection_census.jsonl` | as inc11_cross_level, targets gfx942 / gfx950 (official AMD stages to TTGIR); the selection run before the increment-12 registration (docs/dsl_v2/increment_12.md §2) | increment 11 + uncommitted plugin extension | `w12_cross_run.sh` |
| `inc12_selection_recheck_after_fix.jsonl` | the selection run's disjoint cases (test_cast_bf16_rounding, cumprod scans, dot add-*) after the aborted-launch fix | increment 12 fix (before registration) | `w12_cross_run.sh` (KSEL) |
| `inc6_combined_summary.json` | `scripts/dsl_v2/summarize_capture.py inc6_combined.jsonl --baseline inc4_broad.jsonl` | — | — |

Status per launch: `complete` (every written element established), `partial`, `aborted` (a program instance aborted),
`error` (evaluator or binding failure: a tool defect), `nothing written`, `skipped` (above the size bound).  Before the
increment-5 plugin fix, `complete` counted only point values; launches whose only non-point outputs were special values
(NaN, +-inf: established values) appear as `partial` without a reason.
