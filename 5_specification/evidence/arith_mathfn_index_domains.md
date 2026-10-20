# ARITH / MATHFN / INDEX domain extensions for path A (package D, 2026-10-20)

Status: drafted, pending independent review. Nothing here is validated. The top-level `contracts/<ID>.yaml` and
`adapters/<ID>.py` are still the validated versions. The candidates are:

| contract | draft contract | draft adapter |
|---|---|---|
| ARITH | `contracts/draft/ARITH_path_a_A2_20261020.yaml` | `contracts/draft/ARITH_adapter_path_a_A2_20261020.py` |
| MATHFN | `contracts/draft/MATHFN_path_a_A2_20261020.yaml` | `contracts/draft/MATHFN_adapter_path_a_A2_20261020.py` |
| INDEX | `contracts/draft/INDEX_path_a_A2_20261020.yaml` | `contracts/draft/INDEX_adapter_path_a_A2_20261020.py` |

New helper: `kernel_spec/arith_mathfn_index_arb.py` (Arb enclosures for MATHFN). Each draft adapter reads its own
contract, either next to the draft file or at `<stage>/contracts/<ID>.yaml` when it is staged as
`<stage>/adapters/<ID>.py`. It never reads the top-level contract.

## 1 What changed

All the changes extend the existing `real` profiles (`task_choice_required: false`). No task-choice profile was added.

**ARITH `real`** (`aten.pow` added to covers)
- AR.SCALAR_VALUE: a Python scalar operand, alpha, bound or fill value is the exact value received, meaning the binary64
  value of a Python float or the exact value of a Python int in [-2^63, 2^64), which reaches ATen as an int64 or
  uint64 Scalar (CONTRACT_PROTOCOL §4.1). Boolean scalars and other integers are unsupported. The decision also rejects
  reading (d), the scalar at its float32 (default-dtype) value. That reading is suggested by T_TENSOR_ATTR.SCALAR_DTYPE,
  but the sentence assigns a promotion category, not a value conversion. Eager follows the op-math reading (c), so on the
  recorded f32 calls eager itself deviates from RN(f) by up to about 1 ulp (section 3); that deviation is K-G. The PyTorch 2.10 type-promotion
  page (T_TENSOR_ATTR) says only that a same-category scalar does not change the result dtype and that values are not
  inspected. torch.add and torch.mul say only `input + alpha*other` and `input*other`. No retained page says that the
  scalar's value is converted first. Rounding to the tensor dtype and rounding to the op-math type are implemented as the
  diagnostic readings `scalar_rounded_to_dtype` and `scalar_rounded_to_opmath`. A rational that no ATen Scalar can carry
  (for example 1/3) is invalid_input.
- AR.FLOOR_INFINITE (review A1): floor division or floor_divide with an infinite operand is unsupported. The torch.div
  page says floor mode equals Python // and np.floor_divide (inf//2 = nan, -1//inf = -1, as eager gives), while the
  torch.floor_divide page says floor(x/y) (extended reals: inf, 0). The pages disagree, so the profile gives no answer.
- AR.EXTENDED_REAL: ±inf are admitted wherever the extended-real result exists (IEEE754.INF_ARITH). inf−inf, 0·inf,
  inf/inf and x/0 are domain_undefined (IEEE754.INVALID; the contract does not demand NaN). NaN stays invalid_input.
- AR.POW / AR.ZERO_POW: when the exponent's exact value is an integer n (a Python int, or a float such as 3.0), the task
  value is the exact integer power for every base, negative bases included. x**0 = 1, including 0**0 and inf**0
  (IEEE754.POW0). A zero base with n < 0 is undefined. Non-integral exponents, |n| > 64, tensor exponents, scalar bases
  and integer tensors are unsupported, each with a stated reason.
- AR.INTEGER / AR.INT_OVERFLOW: on same-dtype integer tensors, add, sub, rsub, mul, neg, abs, sign, maximum, minimum,
  clamp*, where and masked_fill use exact integer arithmetic. The documentation is silent about overflow, and signed
  overflow is undefined behaviour in C++. A result outside the dtype is therefore domain_undefined: neither the wrap
  nor any other device value is judged. An integer scalar enters exactly as the value received; only
  when an exact result, or a selected scalar in where/masked_fill/clamp, leaves the dtype range is the call
  domain_undefined (AR.E97, AR.E103, AR.E104). A float scalar or float alpha promotes the result and is outside this
  same-dtype profile.
- The autograd profiles keep their finite, representable-scalar scope.

**MATHFN `real`**
- MF.MAGNITUDE: the |x| ≤ 100 bound is removed for sin, cos and tanh. The bound was a reference-evaluation scope, not
  part of any function definition. It kept exp-type endpoints of moderate size and avoided relying on argument reduction
  at huge arguments. Arb reduces sin and cos arguments rigorously and bounds tanh's error. The enclosure endpoints stay
  small rationals from 2^-1074 up to the binary64 maximum, at about 0.1 ms per element. exp, expm1 and sigmoid keep the
  bound because their exact endpoints would become astronomically large or small. log, log1p, sqrt, rsqrt and erf keep it
  because no recorded call needs more.
- MF.EVALUATION: exp, expm1, log, log1p, sin, cos, tanh and sigmoid are enclosed with python-flint Arb at 320 bits and
  re-enclosed outward at 256 bits. Rational points stay exact.

**INDEX `real`**
- IX.INTEGER_DATA: gather and index_select copy integer data (i64/i32/i16/i8/u8) exactly. embedding's weight is
  documented as floating, so integer weights stay unsupported.
- IX.PADDING_FORWARD: the forward value does not depend on padding_idx. T_EMBEDDING_F.PADDING and T_EMBEDDING.PADDING
  define it only through the gradient, TS_EMBEDDING_CPP.FORWARD is `index_select`, and EMB's reviewed EM.LOOKUP says the
  same. Accepted values are native −1 or 0 ≤ p < V, and public None or [−V, V), with negatives normalized
  (TS_FUNCTIONAL_EMB.EMB_PADDING). Other native values are unsupported and out-of-range public values are invalid.
  max_norm, frequency scaling and sparse stay unsupported.

## 2 Defect found in the shared evaluator (not fixed here; shared code)

With mpmath 1.3.0, `mpmath.iv.exp` at 256 bits returns intervals that exclude the true value of exp(x) for |x| between
about 2^-271 and 2^-55. Many of them are zero-width points. Examples checked against Arb at 4000 bits: exp(2^-100) is a
point that lies 8.2e-92 below the true value, and exp(2^-55) has both endpoints below the true value. `kernel_spec.real`
uses iv.exp for exp, expm1 and sigmoid, uses it in tanh as 1−2/(exp(2x)+1), and computes log1p as log(1+x). All of these
inherit wrong enclosures for small arguments. A scan of k = 1..1100 in steps of 3, with ±2^-k, found 97 failing points
for exp and expm1, 41 for sigmoid, 40 for tanh and 39 for log1p. sin, cos and erf had none. f32 subnormals and small f64
values fall in this range, so the validated MATHFN adapter returns wrong enclosures there. The MATHFN draft avoids the
problem through MF.EVALUATION. Other users of `kernel_spec.real` are still exposed, including SOFTMAX/ACT through rl.exp
and `kernel_spec/expr.py`, the checker's own evaluator for expected expressions. It runs at 512 bits, so its failing
band is shifted but present. Contract examples in this package avoid tiny arguments for that reason. Fixing it is a
shared-code change for the owners of `kernel_spec/real.py` and `kernel_spec/expr.py`.

## 3 PyTorch 2.10 cross-check (supporting evidence, not the definition)

PyTorch 2.10.0+cu128 eager ran on CUDA (RTX A6000) with random data in the recorded shapes and dtypes. The draft
references were evaluated on the full tensors. "RN(x)" means x rounded to nearest-even in the output dtype, compared on
4096 sampled coordinates. The full log is in `.cache/tmp/pkgD_amix_xcheck.txt`, from the script
`.../scratchpad/pkgD/amix_xcheck.py`. Dtype, shape and the pattern of infinite and finite coordinates agreed on every
coordinate in every case.

| recorded call | eager == RN(f) | eager == RN(dtype-rounded scalar) | eager == RN(op-math-rounded scalar) | reference time |
|---|---|---|---|---|
| mul.Tensor f32 [1,128,3072] × 0.044715 | 2365/4096 (≤1.06 ulp) | 4096/4096 | 4096/4096 | 25 s |
| mul.Tensor f32 [1,128,3072] × 0.7978845608028654 | 2764/4096 (≤0.96 ulp) | 4096/4096 | 4096/4096 | 24 s |
| mul.Tensor bf16 [1,128,3072] × 0.044715 | 4096/4096 | 3449/4096 | 4096/4096 | 22 s |
| mul.Tensor bf16 [1,128,3072] × 0.7978845608028654 | 4096/4096 | 3098/4096 | 4096/4096 | 23 s |
| mul.Scalar f32 [1,14,128,64] × 0.3535533905932738 | 3226/4096 (≤0.78 ulp) | 4096/4096 | 4096/4096 | 7 s |
| add.Tensor f32 [1,128,1] + 1e-06 | 127/128 (≤0.52 ulp) | 128/128 | 128/128 | <1 s |

Reading: eager converts the scalar to the op-math type (float32) before multiplying. For f32 tensors it therefore
differs from RN of the exact-scalar product by up to about 1 ulp. For bf16 the extra float32 precision makes RN(exact)
and RN(op-math) coincide on every sampled point. This is the K-G effect that AR.SCALAR_VALUE assigns to the
implementation. Under the exact-value decision, an f32 kernel that converts the scalar first, as eager does, shows
deviations of up to about 1 ulp. Whether that counts as a defect is decided by the evaluation layer's numerical rule,
not by this contract. The diagnostic readings identify the cause.

| recorded call | eager == RN(f) | notes |
|---|---|---|
| pow f32 / bf16 [1,128,3072] ^ 3.0 | 3001/4096 (≤1.24 ulp) / 3247/4096 (≤0.98 ulp) | x*x*x with intermediate rounding |
| pow f32 / bf16 [1,128,3072] ^ 2.0; f32 [1,128,896] ^ 2 and ^ 1.0 | 4096/4096 each | |
| pow f32 [1,128,1] ^ 3 | 91/128 (≤1.01 ulp) | |
| add f32 [1,14,128,128] + mask [128,128] containing -inf | 4096/4096; -inf pattern identical on all 229376 | |
| where.self(b8 [128,128], -inf [], 0 []) | 4096/4096 | |
| add i64 [128] + 0 | 128/128 | |
| cos / sin f32 [1,128,64], RoPE angles up to 127 | 3591/4096 (≤1.13 ulp) / 3660/4096 (≤1.35 ulp) | reference 1 s |
| tanh f32 [1,128,3072], abs(x) up to 269 | 4063/4096 (≤1.52 ulp) | reference 46 s |
| gather i64 [1,512] dim 1, idx [1,128] (values above 2^24) | 128/128 | |
| embedding f32/bf16 [30522,128], idx [1,128], padding_idx 0 | eager returns the stored, nonzero row 0 at padding positions; equals index_select | reference on a [300,128] slice: 4096/4096 |

Special cases and refusals, eager compared with the draft:
- inf + (-inf) gives nan in eager; the draft says domain_undefined. pow(0, -1) gives inf; draft domain_undefined.
  pow(-8, 1/3) with the Python float 1/3 gives nan; the draft calls it unsupported (non-integral exponent). An exact
  rational 1/3, which no Scalar can carry, is invalid_input.
- pow(-2, 3.0) = -8 and pow([0, inf, -inf], 0) = [1, 1, 1]: agree with AR.POW / AR.ZERO_POW.
- int64 max + 1 wraps to -2^63, int16 300*200 gives -5536, uint8 -3 gives 253 and int8 + 1000 gives -23 (the scalar
  wraps too). The draft gives domain_undefined for all of these (AR.INT_OVERFLOW). Because eager does not reject
  int8 + 1000, the draft treats an int64 scalar exactly and tests only the result range. An earlier candidate called
  this invalid_input; it was changed after this run.
- int64 + 0.5 promotes to float32 (draft: unsupported, mixed dtypes). int64 add with alpha=0.5 raises RuntimeError in
  eager (draft: unsupported).
- F.embedding with padding_idx=2 and V=2 raises AssertionError in eager (draft: invalid_input). Native aten.embedding
  with padding_idx=5 returns the lookup (draft: unsupported, undocumented value). F.embedding with padding_idx=-1 and
  V=1 returns the row (draft: value).
- Native embedding with an int64 weight works in eager. The draft leaves it unsupported because the page documents a
  floating weight. gather on uint8 data agrees.
- sin(f32 max) = -0.52187651... and cos(127) = 0.2323590964 (f32) lie within 1 ulp of the reference enclosures.

## 4 Checks

```
cd 5_specification
ST=/data1/tzh/kernel-analyzer/.cache/spec_a2_stage/D   # draft pairs copied as contracts/<ID>.yaml, adapters/<ID>.py
/data1/tzh/envs/liger/bin/python tools/check_contracts.py --contracts $ST/contracts --adapters $ST/adapters --only ARITH MATHFN INDEX
/data1/tzh/envs/hf_ga_5190/bin/python -m pytest -q tests/test_arith_mathfn_index_domains.py
```
