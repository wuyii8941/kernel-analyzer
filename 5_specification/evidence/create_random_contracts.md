# CREATE (new) and RANDOM native_forward_given_mask: path A closure package E (2026-10-20)

Drafts (pending independent review; not in `validated/`, not top-level):

- `contracts/draft/CREATE_path_a_A2_20261020.yaml` and `contracts/draft/CREATE_adapter_path_a_A2_20261020.py`
- `contracts/draft/RANDOM_path_a_A2_20261020.yaml` and `contracts/draft/RANDOM_adapter_path_a_A2_20261020.py`

Staged for checking as `<stage>/contracts/<ID>.yaml` and `<stage>/adapters/<ID>.py`. The adapters read the contract
next to them (`<stage>/contracts/<ID>.yaml`).

## 1 What is covered

| Recorded call (blocked_keys.txt) | Contract / profile | Output |
|---|---|---|
| `aten.arange.default(128)` (device, pin_memory dropped) | CREATE / explicit | `out`: int64 `[0..127]` |
| `aten.zeros.default([0], dtype=bf16 / f32)` | CREATE / explicit | `out`: empty `[0]` |
| `aten.ones.default([128,128], dtype=bool)` | CREATE / explicit | `out`: all true (encoded 1) |
| `aten.ones_like.default(f32 [])`, `(f32 [1,1,2,2])` | CREATE / explicit | `out`: ones, f32 |
| `aten.scalar_tensor.default('-inf' or -inf, dtype=f32)`, `(0.0, dtype=f32)` | CREATE / explicit | `out`: `-inf`, `0` |
| `aten.native_dropout.default(x f32/bf16 [1,128,128] / [1,128,768], 0.1, True)` | RANDOM / native_forward_given_mask | `out = [output, mask]` |

Review fixes (2026-10-21, reviews/2026-10-20/A2_package_E_review.md): Boolean observations must hold real Booleans
(CR-1, RN-1); an infinite arange step is unsupported (CR-2; eager arange(0, 5, inf) returns an empty tensor);
message profile name (CR-3); RANDOM decision / clause wording (RN-2); the p range basis (RN-3).

Also defined: arange.start / start_step (length `ceil((end-start)/step)`), full, zeros_like, full_like; native_dropout
with train False, p = 0 and p = 1. Task-choice profile: CREATE / default_float32 (result dtype = process
default floating dtype, declared float32). `empty` is not covered (values uninitialized, T_EMPTY.UNINIT).

## 2 Decisions

CREATE: CR.SCOPE, CR.DEFAULT_DTYPE (needs_binding), CR.ARANGE_DTYPE, CR.ARANGE_LENGTH (exact formula; refusal band
2^-30 around integer quotients when an argument is non-integer, following the page's rounding warning), CR.CONVERSION,
CR.LIKE, CR.BOOL_ENCODING (Boolean results as exact 1/0 in the single numeric field `out`; the checker
requires a bool-dtype observation), CR.SCALAR_DTYPE (no dtype: `_refs` infers from the Python type, the native factory uses the default float
dtype; unsupported). RANDOM: RN3.CONDITIONED (returned mask as reference keyword `observed_mask`), RN3.SCALING
(`input * mask * 1/(1-p)`, p = exact received double), RN3.MASK_POSITION (mask compared in the numeric encoding 1/0
because `out` is a numeric field), RN3.TRAIN_NONE (decomposition: eval; CPU/CUDA kernels: training; unsupported).

What native_forward_given_mask checks: the output is consistent with the returned mask and the documented scaling,
and the mask position equals the documented mask (deterministic cases) or the observed mask (conditioned case, a
binding consistency check only). It does not check the randomness or keep probability of the mask; the reading
`keep_fraction` reports `[kept fraction, 1-p]` as a diagnostic outside the verdict.

## 3 Checks

- `tools/check_contracts.py --contracts <stage>/contracts --adapters <stage>/adapters --only CREATE RANDOM`:
  CREATE 44/44 examples, 7/7 mutants; RANDOM 32/32 examples, 1/1 reading, 13/13 mutants; verdict consistent.
- `pytest 5_specification/tests/test_create_random_contracts.py`: 20 passed.

## 4 PyTorch 2.10 cross-check (supporting evidence, not the definition)

hf_ga_5190 (torch 2.10.0+cu128), CPU and CUDA (sm_86).

Factories: every recorded structure (arange(128), arange(5,1,-2), zeros([0]) bf16/f32, ones([128,128], bool),
scalar_tensor(-inf / 0.0, f32), ones_like of f32 [] and [1,1,2,2]) gives values, dtype and shape equal to the
reference (`check_output` true, CPU and CUDA). Refusals agree: arange(5,1,1) and arange(0,5,0) raise / invalid_input;
zeros([-1]), full([2],300,u8), scalar_tensor(-inf, int64) raise / invalid_input. arange(0,2.5) -> f32 [0,1,2] equals
default_float32. Differences, by decision: scalar_tensor(1) without dtype -> eager f32 (unsupported,
CR.SCALAR_DTYPE); arange(1,2.5,0.5) -> eager [1,1.5,2] (unsupported, ambiguity band of CR.ARANGE_LENGTH);
scalar_tensor(1e39, f32) -> eager raises overflow (unsupported, CR.CONVERSION).

native_dropout, p = 0.1, train True, reference given the eager mask (exact real output vs eager output):

| device | dtype | shape | keep fraction | dropped exact 0 | worst deviation | kept coords > 0.5 ulp |
|---|---|---|---|---|---|---|
| cpu | f32 | [1,128,128] | 0.8965 | yes | 1.222 ulp | 8429 / 14689 |
| cpu | f32 | [1,128,768] | 0.9000 | yes | 1.222 ulp | 51222 / 88476 |
| cpu | bf16 | [1,128,128] | 0.8965 | yes | 0.889 ulp | 4154 / 14689 |
| cpu | bf16 | [1,128,768] | 0.9000 | yes | 0.889 ulp | 25096 / 88476 |
| cuda | f32 | [1,128,128] | 0.8995 | yes | 1.222 ulp | 8457 / 14738 |
| cuda | f32 | [1,128,768] | 0.9011 | yes | 1.222 ulp | 51256 / 88579 |
| cuda | bf16 | [1,128,128] | 0.8994 | yes | 0.444 ulp | 0 / 14735 |
| cuda | bf16 | [1,128,768] | 0.9004 | yes | 0.444 ulp | 0 / 88516 |

The returned mask is Boolean with the input's shape in every case; the mask position of the reference equals it.
Eager itself is not within half an ulp of the real task value: f32 (CPU and CUDA) and CPU bf16 deviate up to about
1.2 ulp, consistent with the scale 1/(1-p) being rounded to the tensor's dtype (CPU bf16: to bfloat16) before the
product is rounded. Exact membership is therefore false for eager as well; the size of such deviations is for the
measurement (`measure.py`) to judge, not a tolerance in f. Eager checks of refusals: p outside [0, 1] is invalid_input on the documentation basis (p is a probability, T_DROPOUT; RN3.SCALING), not because eager rejects it uniformly: CPU training with p = 1.5 raises in bernoulli_, but evaluation accepts it and returns the input, CUDA training with p = 1.5 returns -0 and an all-false mask, and p = -0.5 scales by 2/3;
an int64 input raises (unsupported); train=None on CPU draws a random mask (training), unlike the decomposition
(RN3.TRAIN_NONE, unsupported); p = 1 gives zeros and an all-false mask, train False the input and an all-true mask
(both equal the reference). Reference time: about 1 s for 16384 and 6-7 s for 98304 coordinates (exact Python
values).
