# NORM / EMB / NLL: ATen-call interfaces for path A (package A, 2026-10-20)

Status: drafted, implementation pending. The candidates are in `contracts/draft/` and are not promoted. The top-level
`contracts/NORM.yaml`, `contracts/EMB.yaml`, `adapters/NORM.py` and `adapters/EMB.py` are still the validated versions.

| contract | draft files | new non-task-choice profile | ATen overloads |
|---|---|---|---|
| NORM | `NORM_path_a_A2_20261020.yaml`, `NORM_adapter_path_a_A2_20261020.py` | `aten_native` (output `out`) | `aten.native_layer_norm`, `aten.native_layer_norm_backward`, `aten.rms_norm` |
| EMB | `EMB_path_a_A2_20261020.yaml`, `EMB_adapter_path_a_A2_20261020.py` | `aten_dense_backward` (output `out`) | `aten.embedding_backward` (sparse=false), `aten.embedding_dense_backward` |
| NLL (new) | `NLL_path_a_A2_20261020.yaml`, `NLL_adapter_path_a_A2_20261020.py` | `aten_real` (output `out`) | `aten.nll_loss_forward`, `aten.nll_loss_backward` |

The shared semantics are in `kernel_spec/norm_emb_nll_aten.py`. They are exact rationals, and `1/sqrt` is a rigorous
256-bit enclosure. Each adapter builds its interface with `kernel_spec.aten_adapter.make_reference` and reads
`covers` from its own staged contract (`<stage>/contracts/<ID>.yaml`). The existing NORM profiles (`torch`,
`torch_autograd`, `rms_offset`) and the EMB profile `torch210_dense_real` keep their semantics. Their outputs are
now listed explicitly, and they refuse `aten.*` calls with `unsupported`. As a result, exactly one non-task-choice
profile answers each recorded call. The test `only_aten_profile_answers` checks this.

## Decisions (summary; full text in the drafts)

- **NM.ATEN_INTERFACE / EM.ATEN_INTERFACE**: the ATen interface lives in a separate profile, not in `torch` or
  `torch210_dense_real`, because those profiles promise differently arranged outputs (y/saved/grad;
  lookup+dweight). `aten.embedding` (forward) stays with INDEX, so a forward call never has two applicable profiles.
- **NM.ATEN_STATS**: `mean` and `rstd = 1/sqrt(var_biased + eps)` have the outer shape followed by 1s. In the
  backward, any mean/rstd with M values is read in row-major order. This follows the native kernels; eager accepts a
  flat `[M]` and gives identical results.
- **NM.ATEN_BWD_SUPPLIED**: the backward uses the supplied mean/rstd, following the SOFTMAX precedent, the
  decomposition, the native kernel and derivatives.yaml. Recomputing is not defined, because the schema has no
  `eps`. Mutant NM.M18 implements that misreading. Positions where `output_mask` is false are `null`. A mask of true
  for an absent weight or bias is `unsupported` (eager raises).
- **NM.ATEN_RMS_EPS_NONE**: `eps=None` is `torch.finfo(x.dtype).eps` (documented): 2^-23 for f32 and 2^-52 for f64,
  which match the native default. For f16/bf16 the documentation (2^-10 / 2^-7) and the native code (2^-23, confirmed
  in eager below) disagree, so that case is `unsupported`. It is never 0 or 1e-5; FlagGems D5 uses 1e-5 (mutants
  NM.M20, NM.M21).
- **NM.ATEN_DTYPES**: the dtype of mean/rstd is metadata. The check accepts the input dtype or f32 for the returned
  stats and any floating dtype for the backward's stats. Values are what is judged.
- **EM.ATEN_PADDING**: at the ATen level `padding_idx` is compared literally, so `-1` pads no row. The functional
  wrapper has already normalised it (mutant EM.M8).
- **EM.ATEN_BACKWARD**: for sparse=false the composite equals `embedding_dense_backward`. sparse=true is
  `unsupported`.
- **NL.TASK, NL.FORWARD, NL.TOTAL_WEIGHT, NL.EMPTY_MEAN, NL.BWD_SUPPLIED (reading NL.R1), NL.BWD_EMPTY**:
  - Reduction enum: 0 none, 1 mean, 2 sum.
  - `total_weight` is the sum of `w[t]` over non-ignored targets, or their count when there is no weight. It is 0 for
    reduction none with 2D input.
  - A mean over total weight 0 is `domain_undefined`. Eager returns NaN.
  - The backward uses the supplied `total_weight`. An all-ignored backward gives zeros.

## Return arrangement (harness rule v2)

`ATEN_RETURNS` is defined in the NORM adapter (`aten.native_layer_norm`: 3, `aten.native_layer_norm_backward`: 3)
and the NLL adapter (`aten.nll_loss_forward`: 2). The task defines every position, so `not_compared` is empty
throughout. The only `null` positions come from `output_mask` false, which is a call argument.

## PyTorch 2.10 eager cross-check of the recorded argument structures

Environment: torch 2.10.0+cu128, CPU and one CUDA GPU of this host. Inputs are random values at the recorded shapes, dtypes
and literal arguments (`blocked_keys.txt`). The staged draft reference is compared against two things:

- (a) eager evaluated in float64 on the same exactly representable inputs. This agreement checks the formula.
- (b) eager in the recorded dtype on CPU and CUDA, for dtype, shape and refusal, and as a rounding indication.

Columns: max relative difference against the reference.

| call (recorded structure) | ref outcome | (a) eager f64 | (b) eager dtype/shape (CPU / CUDA) |
|---|---|---|---|
| native_layer_norm f32 [1,128,128], w,b [128], eps 1e-12 | value | ≤ 3.5e-13 | out f32 [1,128,128]; mean/rstd f32 [1,128,1] on both |
| native_layer_norm bf16 [1,128,128], eps 1e-12 | value | ≤ 1.4e-13 | mean/rstd **bf16** on CPU, **f32** on CUDA, shape [1,128,1] (NM.ATEN_DTYPES) |
| native_layer_norm f32 [1,128,768], eps 1e-5 | value (8 s) | ≤ 5.0e-13 | f32, [1,128,1] |
| native_layer_norm_backward bf16 [1,128,128], bf16 mean/rstd of shape [128] (as recorded), mask [T,T,T] | value | ≤ 1.0e-13 | CPU accepts bf16 [128] stats. **CUDA refuses** bf16 stats ("expected scalar type Float but found BFloat16") and accepts f32 [128] stats. Outputs: grad_input bf16 [1,128,128], grad_weight/grad_bias bf16 [128] |
| rms_norm f32 [4,64], weight [64], eps not passed (D5) | value | 4.3e-16 against explicit eps 2^-23 | CPU and CUDA f32 differ from the reference by 1.4e-7 (rounding). eps=1e-5 (the D5 reading) differs by 5.7e-6 |
| embedding_backward f32/bf16 grad [1,128,128], indices [1,128], num_weights 512 / 2, padding_idx -1 | value | 0 | [num_weights,128] in the grad dtype |
| embedding_backward f32/bf16 grad [1,128,128], num_weights 30522, padding_idx 0 | value (12 s) | 0 | [30522,128] |
| embedding_backward bf16 grad [1,128,768], num_weights 1024, padding_idx -1 | value | 0 | [1024,768] bf16 |
| nll_loss_forward f32 [128,30522] (log_softmax), target with 20 × −100, None, 1, −100 | value (17 s) | 0, 0 | output and total_weight f32 [] on both. total_weight is exact |
| nll_loss_backward f32 grad_output [], self [128,30522], total_weight [] | value (24 s) | 0 | f32 [128,30522] |

Further eager checks used in the decisions:

- rms_norm with eps=None on bf16 and f16 inputs on CPU and CUDA gives 0.9453125 for x = 1e-3. That matches eps
  2^-23; the documented eps would give 0.0113 and 0.0320. This motivates NM.ATEN_RMS_EPS_NONE.
- `native_layer_norm_backward` with weight None and mask [T,T,T] raises on CPU and CUDA ("tensor does not have a
  device"). This motivates the `unsupported` case.
- nll_loss: an all-ignored mean gives NaN with total_weight 0, and an all-ignored backward gives zeros.
  `nll_loss_backward` with a supplied total_weight of 4 (recomputed value 3) uses 4. An out-of-range target raises
  IndexError on CPU and fails a device assertion on CUDA.

Rounding notes (K − G, not semantics):

- CPU eager in bf16 shows large relative differences at coordinates with cancellation: up to 0.56 in grad_weight of
  `native_layer_norm_backward` with bf16 stats, and up to 35× on a near-zero row of `embedding_backward` with
  num_weights 2.
- CUDA with f32 statistics stays within 0.004 (bf16 resolution).

Not cross-checked with the exact reference:

- embedding_backward with num_weights 50257 / 151936 (gpt2, Qwen; up to a 151936 × 896 output).
- nll_loss with V = 50257 / 151936.

These need the bulk evaluator (`kernel_spec/bulk.py`, outside this package). Their argument structure (dtypes,
padding_idx −1/0, reduction 1, ignore_index −100) is the same as in the rows above.

## Checks

```
cd 5_specification
# stage: copy contracts/draft/<ID>_path_a_A2_20261020.yaml -> .cache/spec_a2_stage/A/contracts/<ID>.yaml and the adapter likewise
/data1/tzh/envs/liger/bin/python tools/check_contracts.py --contracts ../.cache/spec_a2_stage/A/contracts \
    --adapters ../.cache/spec_a2_stage/A/adapters --only NORM   # 34/34 examples, 21/21 mutants, exit 0
... --only EMB    # 26/26 examples, 11/11 mutants, exit 0
... --only NLL    # 18/18 examples, 2/2 readings, 11/11 mutants, exit 0
/data1/tzh/envs/hf_ga_5190/bin/python -m pytest -q tests/test_norm_emb_nll_aten.py   # 15 passed
```

Sources were added by `tools/fetch_sources.py`:

- New retained sources: `T_NLLLOSS`, `T_NLL_LOSS_F`, `NEN_DECOMP`, `NEN_REFS`, `NEN_REDUCTION_PY`,
  `NEN_FUNCTIONAL`, `NEN_LOSSNLL_CPP`, `NEN_LAYER_NORM_CPP`, `NEN_DERIVATIVES`.
- New items on existing sources: `TS_NATIVE_FUNCTIONS.NORM_EMB_NLL_SCHEMAS` and
  `TS_EMBEDDING_CPP.BACKWARD_DISPATCH`.
