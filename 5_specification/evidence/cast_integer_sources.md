# CAST: integer and boolean sources (path A closure package F, 2026-10-20)

Draft pair: `contracts/draft/CAST_path_a_A2_20261020.yaml` and `contracts/draft/CAST_adapter_path_a_A2_20261020.py`.
The top-level `contracts/CAST.yaml` / `adapters/CAST.py` (validated, float-only) are unchanged. Review status: drafted;
implementation status: pending.

## Why

Path A recorded `aten._to_copy.default(T[int64,[1,128,1],stride=[128,1,1]], dtype=torch.float32)` in Qwen2.5-0.5B
(fp32 and bf16 runs; the position ids). The validated `finite_rne` profile admitted only f16/bf16/f32/f64 sources, so the
call had no applicable profile.

## What the sources say

| Source | Content used |
|---|---|
| `T_TO` (PyTorch 2.10 Tensor.to) | Returns a tensor of the requested dtype; points to "C++ type conversion rules" and states truncation for float to integer (not used: CA.F2I keeps that direction out). Says nothing about integer to floating rounding. |
| `CPP23_CONV_FPINT` (N4950 [conv.fpint], new) | Integer to floating: exact if possible; otherwise an implementation-defined choice of the next lower or higher value; out of range undefined; bool false -> 0, true -> 1. |
| `CPP23_CONV_DOUBLE` (N4950 [conv.double], new) | Floating to floating: exact representation if possible; between two adjacent values an implementation-defined choice. |
| `IEEE754.RNE` | roundTiesToEven, the rule selected by CA.PROFILE. |

So integer to floating conversion has exactly the same structure as floating narrowing: the value is fixed when it is
representable, and the neighbour is a choice otherwise. The profile's existing task choice (roundTiesToEven) is reused.

## Decisions written

- **CA.INTSRC**: integers are their mathematical values (u8 is 0..255); exact when representable; otherwise one
  roundTiesToEven step straight to the target (no intermediate format); bool -> 0/1; out-of-range or non-boolean
  records are invalid_input. The target range follows IEEE 754 (round first, Section 4.3.1, then judge overflow,
  Section 7.4; `IEEE754.OVERFLOW`), not C++: 65505..65519 round to 65504 in f16 and are admitted, |n| >= 65520 overflows
  and is unsupported. u8 being unsigned is cited from `T_TENSOR_ATTR.INT_DTYPES` (item added 2026-10-21).
- CA.PROFILE (review fix 2026-10-21): the Tensor.to page cites C++ only for float -> integer truncation; applying
  [conv.double]/[conv.fpint] to floating targets is labelled as the contract's own reading.
- **CA.SPLIT**: alternatives (a) a separate non-choice profile for exactly representable values, (b) a separate
  task-choice integer profile, (c) extend `finite_rne` (chosen). Path A records a task choice for CAST/finite_rne on
  `aten._to_copy` (`sufficiency/operator_entry/task_choices_A2.json`), and `bind_key.py` reports "more than one
  profile applies" when two profiles fit one call. Option (a) would therefore turn the recorded call (and every
  bf16->f32 widening) into an ambiguity even though both profiles give the same exact value. For the recorded call
  every value is a small non-negative integer, so the result is independent of the rounding choice.
- **CA.F2I**: float to integer / any to bool remain unsupported (alternative: truncation per T_TO; deferred).
- CA.PROFILE and CA.FINITE restated for the larger source domain.

## Independent expected values (hand-derived)

- f32 has 24 significand bits: 2^24+1 = 16777217 is the midpoint of 2^24 (even significand) and 2^24+2, -> 2^24;
  2^24+3 is the midpoint of 2^24+2 (odd significand 2^23+1) and 2^24+4, -> 2^24+4 (truncation would give 2^24+2).
- f64: 2^53+1 -> 2^53, 2^53+3 -> 2^53+4; 2^40+1 is exact in f64 but becomes 2^40 through an f32 intermediate.
- 2^63-1 -> f32 2^63 (truncation gives 2^63-2^39); -2^63 exact.
- bf16 (8 bits): 32767 -> 32768, 257 -> 256 (tie, even significand 128), -32768 exact.
- f16: 65519 -> 65504 (below the midpoint 65520 of 65504 and 2^16); 65520 rounds to 2^16 > 65504 -> overflow;
  2049 -> 2048 (tie).

## Mutants (all rejected by `check_contracts.py`)

CA.M5 truncation for inexact integers (witness CA.E14, CA.E17), CA.M6 True -> -1 (CA.E18), CA.M7 u8 read as i8
(CA.E19), CA.M8 integer converted through f32 first (CA.E16), plus the earlier CA.M1-M4.

## PyTorch 2.10 cross-check (supporting evidence only)

torch 2.10.0+cu128, CPU and CUDA (RTX A6000), via `torch.ops.aten._to_copy.default(x, dtype=...)` and
`torch.ops.prims.convert_element_type.default(x, dtype)`:

- Recorded structure `aten._to_copy(int64 arange(128).view(1,128,1), dtype=float32)`: float32, shape (1,128,1),
  values 0..127 exactly, input unchanged; agrees with the reference (the test reproduces the reference side).
- Examples CA.E9, E14-E22: both ops agree on CPU and CUDA (dtype and every value).
- Exhaustive u8, i8, i16 -> f16/bf16/f32/f64 and exhaustive i32 in [-70000, 70000] -> f16: agree with an independent
  integer RNE encoder on CPU and CUDA; the smallest integer giving +inf in f16 is 65520 on both.
- 40000 random i64 values (half in full range, half in |n| < 2^40) -> bf16/f32/f64: agree on CPU and CUDA.
- bool [True, False] -> f32 gives [1, 0].
- Refusal difference (expected, not a disagreement about semantics): PyTorch returns +-inf for 65520 -> f16 (E23);
  the contract classifies target overflow as unsupported, as it does for floating sources.

Scripts and raw output were kept in the session scratchpad (not part of the repository); the commands are reproducible
from the bullets above.

## Checks

```
cd 5_specification
S=/data1/tzh/kernel-analyzer/.cache/spec_a2_stage/F   # draft pair copied to $S/contracts/CAST.yaml, $S/adapters/CAST.py
/data1/tzh/envs/liger/bin/python tools/check_contracts.py --contracts $S/contracts --adapters $S/adapters --only CAST
# CAST | drafted | 28/28 examples | 8/8 mutations rejected | consistent
/data1/tzh/envs/hf_ga_5190/bin/python -m pytest -q tests/test_cast_integer_sources.py   # stages the draft pair itself
```

## Known consequence for existing tests

`tests/test_cast_profile_domains.py::test_unreviewed_types_nonfinite_and_overflow_are_refused` expects i64 and b8
sources to be unsupported. It runs against the top-level validated adapter and stays green; if the draft is promoted,
those two subcases must be updated by the test owner (the remaining refusals still hold).
