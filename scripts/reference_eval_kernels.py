"""Triton kernels used to exercise the TTIR mapping and the reference evaluator.

Each kernel isolates instruction families from the coverage table: masked
memory access, reductions, dot, scan, control flow, atomics, conversions,
elementary functions, libdevice calls, bit-level operations and inline asm.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def scale_masked(X, Y, n, alpha, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(Y + offs, tl.load(X + offs, mask=mask) * alpha, mask=mask)


@triton.jit
def add_then_mul(A, B, C, Y, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(A + offs, mask=mask) * tl.load(B + offs, mask=mask)
    tl.store(Y + offs, p + tl.load(C + offs, mask=mask), mask=mask)


@triton.jit
def sequential_sum4(A, B, C, D, Y, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    s = tl.load(A + offs, mask=mask) + tl.load(B + offs, mask=mask)
    s = s + tl.load(C + offs, mask=mask)
    s = s + tl.load(D + offs, mask=mask)
    tl.store(Y + offs, s, mask=mask)


@triton.jit
def softmax_rows(X, Y, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(Y + row * stride + cols, e / tl.sum(e, axis=0), mask=mask)


@triton.jit
def layernorm_rows(X, W, B, Y, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_cols
    y = d * tl.math.rsqrt(var + eps) * tl.load(W + cols, mask=mask) + tl.load(B + cols, mask=mask)
    tl.store(Y + row * n_cols + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def matmul(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PRECISION: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        kk = k * BK + rk
        a = tl.load(A + rm[:, None] * sam + kk[None, :] * sak,
                    mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(B + kk[:, None] * sbk + rn[None, :] * sbn,
                    mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision=PRECISION)
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc.to(C.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def atomic_accumulate(X, OUT, OLD, n, BLOCK: tl.constexpr, USE_OLD: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask, other=0.0)
    old = tl.atomic_add(OUT + (offs % 4), x, mask=mask)
    if USE_OLD:
        tl.store(OLD + offs, old, mask=mask)


@triton.jit
def branch_on_scalar(X, Y, threshold, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    if s > threshold:
        y = x * 2.0
    else:
        y = x - 1.0
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def conversions(X, Y16, YBF, Y8, Y32, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask)
    h = x.to(tl.float16)
    b = x.to(tl.bfloat16)
    f8 = x.to(tl.float8e5)
    tl.store(Y16 + offs, h, mask=mask)
    tl.store(YBF + offs, b, mask=mask)
    tl.store(Y8 + offs, f8, mask=mask)
    tl.store(Y32 + offs, h.to(tl.float32) + b.to(tl.float32) + f8.to(tl.float32), mask=mask)


@triton.jit
def elementary(X, Y, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask, other=1.0)
    a = tl.exp(x) + tl.exp2(x) + tl.log(tl.abs(x) + 1.0) + tl.log2(tl.abs(x) + 1.0)
    b = tl.sin(x) + tl.cos(x) + tl.sqrt(tl.abs(x)) + tl.math.rsqrt(tl.abs(x) + 1.0)
    c = tl.sqrt_rn(tl.abs(x)) + tl.div_rn(x, 3.0) + tl.fdiv(x, 7.0) + tl.math.fma(x, x, 1.0)
    d = libdevice.tanh(x) + libdevice.erf(x) + libdevice.log1p(tl.abs(x)) + tl.sigmoid(x)
    e = tl.maximum(x, 0.5) + tl.minimum(x, -0.5) + tl.clamp(x, -1.0, 1.0) + tl.floor(x) + tl.ceil(x)
    tl.store(Y + offs, a + b + c + d + e, mask=mask)


@triton.jit
def nan_rules(X, Y, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask)
    m1 = tl.maximum(x, 1.0, propagate_nan=tl.PropagateNan.ALL)
    m2 = tl.maximum(x, 1.0)
    isn = x != x
    tl.store(Y + offs, tl.where(isn, m1, m2), mask=mask)


@triton.jit
def scan_and_argmax(X, CS, AM, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask, other=0.0)
    tl.store(CS + offs, tl.cumsum(x, axis=0), mask=mask)
    tl.store(AM, tl.argmax(x, axis=0))


@triton.jit
def bit_level(X, Y, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask)
    bits = x.to(tl.int32, bitcast=True)
    bits = bits & 0x7FFFFFFF
    y = bits.to(tl.float32, bitcast=True)
    tl.store(Y + offs, y + tl.umulhi(bits.to(tl.uint32), bits.to(tl.uint32)).to(tl.float32), mask=mask)


@triton.jit
def inline_asm(X, Y, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask)
    y = tl.inline_asm_elementwise("ex2.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32,
                                  is_pure=True, pack=1)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def while_loop(X, Y, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask, other=0.0)
    i = 0
    while i < 3:
        x = x * 0.5 + 1.0
        i += 1
    tl.store(Y + offs, x, mask=mask)


@triton.jit
def store_load_chain(X, TMP, Y, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    t = tl.load(X + offs, mask=mask) + 16777216.0
    tl.store(TMP + offs, t, mask=mask)
    tl.debug_barrier()
    u = tl.load(TMP + offs, mask=mask)
    tl.store(Y + offs, u - 16777216.0, mask=mask)


@triton.jit
def exp_sum_divide(X, Y, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask, other=-float("inf"))
    e = tl.exp(x)
    tl.store(Y + offs, e / tl.sum(e, axis=0), mask=mask)


@triton.jit
def branch_after_rounding(X, Y, threshold, BLOCK: tl.constexpr):
    # t = x + 1 rounds to 2**24 in FP32 when x = 2**24; the reference keeps 2**24 + 1.
    offs = tl.arange(0, BLOCK)
    t = tl.load(X + offs) + 1.0
    s = tl.max(t, axis=0)
    if s > threshold:
        y = tl.full((BLOCK,), 1.0, tl.float32)
    else:
        y = tl.full((BLOCK,), 0.0, tl.float32)
    tl.store(Y + offs, y)


@triton.jit
def masked_copy(X, Y, Z, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, v, mask=mask)
    tl.store(Z + offs, v)


@triton.jit
def accumulate(ACC, C, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(ACC + offs, tl.load(ACC + offs, mask=mask) + tl.load(C + offs, mask=mask), mask=mask)


@triton.jit
def row_sum(X, Y, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * n_cols + cols, mask=cols < n_cols, other=0.0)
    tl.store(Y + row, tl.sum(x, axis=0))


@triton.jit
def row_sum_drops_last(X, Y, n_cols, BLOCK: tl.constexpr):
    # Deliberate semantic defect: the last element of each row is omitted.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * n_cols + cols, mask=cols < n_cols - 1, other=0.0)
    tl.store(Y + row, tl.sum(x, axis=0))


@triton.jit
def softmax_exp(X, Y, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < n, other=-float("inf"))
    e = tl.exp(x - tl.max(x, axis=0))
    tl.store(Y + offs, e / tl.sum(e, axis=0), mask=offs < n)


@triton.jit
def softmax_exp2(X, Y, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < n, other=-float("inf"))
    e = tl.exp2((x - tl.max(x, axis=0)) * 1.4426950408889634)
    tl.store(Y + offs, e / tl.sum(e, axis=0), mask=offs < n)


@triton.jit
def softmax_reciprocal(X, Y, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < n, other=-float("inf"))
    e = tl.exp(x - tl.max(x, axis=0))
    tl.store(Y + offs, e * (1.0 / tl.sum(e, axis=0)), mask=offs < n)


@triton.jit
def uint8_codes(CODES, OUT, SIGNED, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    c = tl.load(CODES + offs, mask=mask)
    tl.store(OUT + offs, c + 100, mask=mask)  # wraps modulo 256
    tl.store(SIGNED + offs, c.to(tl.int8, bitcast=True).to(tl.int32), mask=mask)


@triton.jit
def cube(X, Y, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x * x * x, mask=mask)


@triton.jit
def int64_to_float64(X, Y):
    # sitofp of 2**53 + 1 loses the 1 in float64; the reference must still contain it.
    tl.store(Y, tl.load(X).to(tl.float64) - 9007199254740992.0)


@triton.jit
def signed_zero(X, Y, S, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    tl.store(Y + offs, libdevice.copysign(tl.full((BLOCK,), 1.0, tl.float32), x))
    tl.store(S + offs, libdevice.signbit(x).to(tl.int32))


@triton.jit
def fp8_e4b15_round_trip(X, Y, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(Y + offs, tl.load(X + offs).to(tl.float8e4b15).to(tl.float32))


@triton.jit
def branch_on_loaded(C, Y, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    c = tl.load(C)
    if c > 0.5:
        tl.store(Y + offs, tl.full((BLOCK,), 1.0, tl.float32))
    else:
        tl.store(Y + offs, tl.full((BLOCK,), 0.0, tl.float32))


@triton.jit
def early_return_on_loaded(C, Y, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    c = tl.load(C)
    if c > 0.5:
        tl.store(Y + offs, tl.full((BLOCK,), 1.0, tl.float32))
        return
    tl.store(Y + offs, tl.full((BLOCK,), 0.0, tl.float32))


@triton.jit
def loop_bound_from_loaded(N, Y, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    n = tl.load(N)
    acc = tl.zeros((BLOCK,), tl.float32)
    for i in range(0, n):
        acc += 1.0
    tl.store(Y + offs, acc)


@triton.jit
def scale_directed(X, Y, alpha, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask)
    if MODE == 0:
        y = libdevice.mul_rd(x, alpha)
    else:
        y = libdevice.mul_ru(x, alpha)
    tl.store(Y + offs, y, mask=mask)
