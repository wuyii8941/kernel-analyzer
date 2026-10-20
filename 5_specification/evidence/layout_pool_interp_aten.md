# Path A closure package C: ATen interfaces for LAYOUT, POOL and INTERP (2026-10-20)

Status: drafts for independent review. Nothing here is validated, and nothing was copied into `validated/`.

- Contracts: `contracts/draft/{LAYOUT,POOL,INTERP}_path_a_A2_20261020.yaml`
- Adapters: `contracts/draft/{LAYOUT,POOL,INTERP}_adapter_path_a_A2_20261020.py`

Each adapter reads the contract next to it. Staged as `<stage>/adapters/<ID>.py`, it reads `<stage>/contracts/<ID>.yaml`. Read from `contracts/draft/`, it reads its paired draft contract.

## Recorded calls and where they go

| Recorded call (blocked_keys.txt) | Contract / profile | Task choice |
|---|---|---|
| `aten.pad.default(i64 [1,128] stride [43xx,1], [0,1], 'constant', -100.0)` (6 keys) | LAYOUT / `real` | none |
| `aten.slice_backward.default(bf16/f32 [1,H,128,32], [1,H,128,64], 3, 32, INT64_MAX or 0, 32, 1)` (6 keys) | LAYOUT / `real` | none |
| `aten.repeat_interleave.self_int(f32 [1,2,128,64] stride [16384,64,128,1], 7, -3)` | LAYOUT / `real` | none |
| `aten.tril.default(b8 [128,128])` | LAYOUT / `boolean` (output `bool_out`) | none |
| D8 `aten.avg_pool2d.default(f32 [1,1,4,4], [3,3], [2,2], [0,0], True)` | POOL / `torch210_aten_avg2d` | **required** (PA.DIVISOR) |
| D8 `aten.avg_pool2d_backward.default(f32 [1,1,2,2], f32 [1,1,4,4], [3,3], [2,2], [0,0], True, True, None)` | POOL / `torch210_aten_avg2d` | **required** (PA.DIVISOR) |
| D18 `aten.upsample_nearest2d.default(f32 [2,3,10,8] stride [240,80,1,10], [20,16], 2.0, 2.0)` | INTERP / `torch210_aten_upsample_nearest2d` | none |

Profile uniqueness for the harness:
- LAYOUT: `real` refuses b8 data and `boolean` refuses all other data, both as unsupported. The functional autograd profiles (pad, cat) refuse ATen-call records as unsupported.
- POOL: `finite_max2d_values` now evaluates only the max-pooling names. The avg calls therefore reach only task-choice profiles, so the harness classifies them as step 2.
- INTERP: the only profile without a task choice is the new ATen profile.

## Decisions (all `drafted`)

- **LY.EXACT_DATA**: integer and boolean data for the data-movement operations, and how a fill value is converted.
  - An integral fill in range is stored as that integer (-100.0 becomes -100).
  - A non-integral or out-of-range fill on an integer tensor is unsupported.
  - On b8 data, only False/True and 0/1 are accepted.
  - Alternatives that are undocumented and were not chosen: truncation, rounding, and nonzero → True.
- **LY.ATEN_PAD**:
  - Constant mode is `constant_pad_nd`, and value None means 0.
  - reflect, replicate and circular are unsupported.
  - Any other mode string is invalid input.
- **LY.SLICE_BACKWARD**:
  - The result is zeros(input_sizes) with grad_output placed at the normalized slice. Bounds are normalized like slice, so INT64_MAX clamps to the dimension end.
  - This is the VJP of the documented slice. It is also the formula of the derivative table and of the official decomposition.
- **LY.REPEAT_INTERLEAVE**:
  - Output index j along dim reads input index floor(j/repeats); dim None flattens the input.
  - Negative repeats are invalid input.
  - A given output_size must equal size*repeats.
  - Tensor-valued repeats are unsupported.
- **PA.ATEN**: the ATen forward returns `finite_avg2d_autograd.out`. The ATen backward returns `finite_avg2d_autograd.grad` with upstream = grad_output (from the derivative table). The divisor task choice is kept: the AvgPool2d and avg_pool2d pages say only "size of the pooling region", and no retained PyTorch document fixes the divisor of a ceil-mode window that extends past the input.
- **IN.ATEN**:
  - The overload name already chooses legacy nearest, because nearest-exact is the separate operator `_upsample_nearest_exact2d`. So no task choice remains for this ATen profile.
  - The backward is the scatter-add VJP, as given by the derivative table.
  - The `.vec` overload is unsupported.
- **IN.ATEN_SCALES**:
  - Without a scale, the index is floor(o*isize/osize).
  - With scale s, the index is floor(o/s). This is the index the retained decomposition uses: `scale = isize/(isize*s)`.
  - The exact reading is admitted only where the decomposition's floating evaluation is provably exact: isize*s exact in binary64, 1/s exact in binary32, and o/s exact in binary32 for every o. Otherwise the call is unsupported.
  - An index that reaches isize is unsupported, because `_unsafe_index` defines no value there.
  - For D18 (s = 2.0, 10→20 and 8→16), both readings give floor(o/2).

New retained sources (fetched with `tools/fetch_sources.py`):
- `LPI_T_REPEAT_INTERLEAVE`: the docs 2.10 page.
- `LPI_NATIVE_FUNCTIONS`: v2.10.0 native_functions.yaml.
- `LPI_DERIVATIVES`: v2.10.0 derivatives.yaml.
- `LPI_DECOMP`: torch/_decomp/decompositions.py.

The two YAML snapshots and the decompositions file are byte-identical to the installed torch 2.10.0+cu128 package files (checked with `cmp`).

## Independent examples and mutants

Expected values were derived by hand, for example the D8 window sums:
- Forward: 54/9 = 6, 45/6 = 15/2, 72/6 = 12, 54/4 = 27/2.
- Backward grid with upstream [[1,2],[3,4]]: rows [1/9, 1/9, 4/9, 1/3] ×2, [11/18, 11/18, 35/18, 4/3], [1/2, 1/2, 3/2, 1].

Each mutant is an independent wrong implementation and is rejected by its witness:
- LAYOUT LY.M19–M25:
  - zero fill;
  - silent truncating conversion;
  - slice placed at offset 0;
  - step ignored;
  - Tensor.repeat tiling;
  - dim ignored;
  - tril keeping the upper triangle.
- POOL PA.M6–M8:
  - full-kernel ceil divisor;
  - divisor_override ignored;
  - grad_output/self swapped.
- INTERP IN.M6–M10:
  - scales ignored;
  - scale multiplied instead of divided;
  - h/w swapped;
  - backward overwrite;
  - half-pixel coordinates.

## Checks

```
cd 5_specification
S=/data1/tzh/kernel-analyzer/.cache/spec_a2_stage/C
/data1/tzh/envs/liger/bin/python tools/check_contracts.py --contracts $S/contracts --adapters $S/adapters --only LAYOUT  # 73/73, 25/25, consistent
... --only POOL    # 40/40, 20/20, consistent
... --only INTERP  # 20/20, 10/10, consistent
/data1/tzh/envs/hf_ga_5190/bin/python -m pytest -q 5_specification/tests/test_layout_pool_interp_*.py   # 39 passed
```

Reference time on recorded sizes (CPU, exact objects):
- pad: about 1 ms;
- slice_backward [1,14,128,32]: about 3 s;
- repeat_interleave: about 1 s;
- tril 128×128: about 0.03 s.

## PyTorch 2.10 eager cross-check (supporting evidence only)

`tests/test_layout_pool_interp_torch.py` was run on CPU and CUDA (torch 2.10.0+cu128). It rebuilds every recorded size and stride with random values and compares eager output with the staged reference.

Exact agreement in values, dtype and shape:
- `aten.pad` i64 with -100.0;
- all four `slice_backward` recorded layouts (bf16 and f32);
- `repeat_interleave.self_int(…, 7, -3)`;
- b8 `tril`;
- `upsample_nearest2d([2,3,10,8] non-contiguous, [20,16], 2.0, 2.0)`;
- the D8 `avg_pool2d` forward (the exact value rounded once to f32).

The D8 backward agrees within f32 rounding. The eager D8 forward was [[6, 7.5], [12, 13.5]], which matches the selected R2 divisor. That agreement is not used to resolve the task choice.

Eager behaviour on the refused or edge cases (probe on CPU):

| Call | Eager result | Profile |
|---|---|---|
| `pad(i64, value=0.5)` | 0 (truncates) | unsupported |
| `pad(i64, value=-2.7)` | -2 (truncates) | unsupported |
| `pad(i64, value=1e30)` | RuntimeError (overflow) | unsupported |
| `pad(b8, value=2.0 or 0.5)` | True | unsupported |
| `pad(b8, value=0.0 or None)` | False | value |
| `pad(b8, value=1.0)` | True | value |
| `slice_backward` with a grad that does not match the slice | RuntimeError | invalid_input |
| `repeat_interleave` with output_size 5 instead of 6 | RuntimeError | invalid_input |
| `repeat_interleave` with repeats -1 | RuntimeError | invalid_input |
| `repeat_interleave` with repeats 0 | empty axis | value |
| `upsample_nearest2d([1,1,1,11], [1,2], 1.0, 0.25)` | [0, 4] (uses the scale) | [0, 4] |
| `upsample_nearest2d([1,1,1,11], [1,2])` (no scales) | [0, 5] | [0, 5] |
| `upsample_nearest2d([1,1,8,8], [2,2], 0.5, 0.25)` | [[0,4],[16,20]] | [[0,4],[16,20]] |
| `upsample_nearest2d([1,1,1,3], [1,6], 1.0, 1.0)` | [0,0,1,1,2,2] (output == 2*input shortcut ignores the scale) | unsupported (index beyond input) |

## Review fixes (2026-10-21, A2_package_C_review.md)

- **LAYOUT-1**: an infinite fill received as a binary64 infinity (aten.pad / constant_pad_nd on floating, integer or b8 data) used to raise OverflowError, which surfaced as implementation_error. It now returns unsupported, as the domain states. New examples: LY.E74–E76.
- **INTERP-1**: the no-scale branch now uses the same standard as the scale branch.
  - It returns a value only where q = isize/osize and every o*q are exactly representable in binary32. Only then does the exact floor(o*isize/osize) provably equal the retained source's trunc(fl32(o)*fl32(q)); everywhere else the result is unsupported.
  - The reviewer's counterexample (2 → 82, o = 41: exact rule 1, source and eager 0) is now unsupported (IN.E21).
  - IN.E12 now uses an exact size pair; IN.E22 and IN.E23 are new; mutant IN.M11 (exact rule without the exactness check) is new.
  - D18 stays evaluable: 10/20 and 8/16 are 1/2, and with s = 2.0 the scale is exact.
  - The alternatives (exact rule everywhere, float32 source arithmetic as the definition, task choice) are listed in IN.ATEN_SCALES.
- **POOL**: PA.ATEN now notes that the functional avg_pool2d page calls kernel_size the "size of the pooling region", which textually supports the full-kernel divisor. The ceil-overhang divisor remains the task choice. New source item: `T_F_AVG_POOL2D.KERNEL_SIZE`.
- **Reviewer tests**: the strict xfails for the fixed cases were removed. The no-scale grid and backward cases now expect unsupported for inexact size pairs, using an independent float32 exactness predicate. For exact pairs they also assert that the exact index equals the source's float32 index.
