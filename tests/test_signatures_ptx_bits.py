"""Bit-level inline PTX (DSL v2 increment 9, rc3 02 6.10): constraints, packing, several outputs, integer and f16x2
instructions.  The kernels are the official inline-asm tests and the official fp8e4b15 conversion; expected values are
computed independently (Python integers, the documented fp8e4b15 decoding).  Compiled-only kernels, CPU only."""
from __future__ import annotations

import numpy as np
import pytest

triton = pytest.importorskip("triton")

import signature_harness as H  # noqa: E402
from test_signatures_structural import _run  # noqa: E402

CATS = ("positive", "boundary", "premise_violation")
U32 = 2 ** 32


def _asm_body(call, store):
    return "    i = tl.arange(0, N)\n" + call + store


@pytest.mark.parametrize("category", CATS)
def test_funnel_shift_signature(category):
    # official test_inline_asm: shf.l.wrap.b32 (y << n) | (x >> (32 - n))
    rng = np.random.default_rng(1)
    x = rng.integers(0, U32, 8, dtype=np.uint64).astype(np.uint32)
    y = rng.integers(0, U32, 8, dtype=np.uint64).astype(np.uint32)
    n = {"positive": 17, "boundary": 0, "premise_violation": 33}[category]   # 33 wraps to 1 (shift amount mod 32)
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    y = tl.load(y_ptr + i)\n"
            f"    s = tl.full([N], {n}, tl.int32)\n"
            "    z = tl.inline_asm_elementwise(\"shf.l.wrap.b32 $0, $1, $2, $3;\", \"=r,r, r, r\", [x, y, s], "
            "dtype=tl.int32, is_pure=True, pack=1)\n    tl.store(out + i, z)\n")
    lo, hi, st = _run(f"shf_{n}", body, {"x_ptr": ("int32", x.view(np.int32)), "y_ptr": ("int32", y.view(np.int32)),
                                         "out": ("int32", np.zeros(8))})
    k = n % 32
    want = [(((int(b) << 32 | int(a)) << k) >> 32) % U32 for a, b in zip(x, y)]
    assert (st == H.ST_OK).all() and [int(v) % U32 for v in lo] == want


@pytest.mark.parametrize("category", CATS)
def test_packed_bit_ops_signature(category):
    # official test_inline_asm_packed: four int8 lanes per 32-bit register
    rng = np.random.default_rng(2)
    x = {"positive": rng.integers(0, 256, 8), "boundary": np.array([0, 255, 31, 32, 224, 1, 128, 127]),
         "premise_violation": rng.integers(0, 256, 8)}[category].astype(np.int64)
    load = ("tl.load(x_ptr + i)" if category != "premise_violation" else "tl.load(x_ptr + i, mask=i < 4)")
    body = (f"    i = tl.arange(0, N)\n    x = {load}\n"
            "    y = tl.inline_asm_elementwise(\"and.b32 $0, $1, 0x1F1F1F1F; shl.b32 $0, $0, 3;\", \"=r,r\", [x], "
            "dtype=tl.int8, is_pure=True, pack=4)\n    tl.store(out + i, y)\n")
    lo, hi, st = _run("packed_" + category[:3], body, {"x_ptr": ("int8", x.astype(np.uint8).view(np.int8)),
                                                       "out": ("int8", np.zeros(8))})
    want = [((int(v) & 0x1F) << 3) & 0xFF for v in x]
    if category == "premise_violation":   # lanes 4..7 undefined: their packed register is not definite
        assert (st[:4] == H.ST_OK).all() and [int(v) % 256 for v in lo[:4]] == want[:4]
        assert (st[4:] != H.ST_OK).all()
        return
    assert (st == H.ST_OK).all() and [int(v) % 256 for v in lo] == want


@pytest.mark.parametrize("category", CATS)
def test_multiple_outputs_signature(category):
    # official test_inline_asm_multiple_outputs
    rng = np.random.default_rng(3)
    a = rng.integers(0, U32, 8, dtype=np.uint64)
    b = rng.integers(0, U32, 8, dtype=np.uint64)
    if category == "boundary":
        a[:3], b[:3] = [0, U32 - 1, 5], [U32 - 1, 0, 5]
    asm = "sub.u32 $0, $2, $3; sub.u32 $1, $3, $2;"
    body = ("    i = tl.arange(0, N)\n    a = tl.load(a_ptr + i)\n    b = tl.load(b_ptr + i)\n"
            f"    (c, d) = tl.inline_asm_elementwise(\"{asm}\", \"=r,=r,r,r\", [a, b], dtype=(tl.uint32, tl.uint32), "
            "is_pure=True, pack=1)\n    tl.store(c_ptr + i, c)\n    tl.store(d_ptr + i, d)\n")
    bufs = {"a_ptr": ("uint32", a.astype(np.uint32)), "b_ptr": ("uint32", b.astype(np.uint32)),
            "c_ptr": ("uint32", np.zeros(8)), "d_ptr": ("uint32", np.zeros(8))}
    if category == "premise_violation":
        body = body.replace("tl.load(b_ptr + i)", "tl.load(b_ptr + i, mask=i < 2)")
    for name, f in (("c_ptr", lambda p, q: (p - q) % U32), ("d_ptr", lambda p, q: (q - p) % U32)):
        lo, hi, st = _run("multi_out_" + category[:3], body, bufs, out_name=name)
        for j in range(8):
            if category == "premise_violation" and j >= 2:
                assert st[j] != H.ST_OK
            else:
                assert st[j] == H.ST_OK and int(lo[j]) % U32 == f(int(a[j]), int(b[j]))


PACKED_MULTI = """
            {
                .reg .b8 tmp<4>;
                mov.b32 {tmp0, tmp1, tmp2, tmp3}, $8;
                cvt.u32.u8 $0, tmp0;
                cvt.u32.u8 $1, tmp1;
                cvt.u32.u8 $2, tmp2;
                cvt.u32.u8 $3, tmp3;
            }
            cvt.rn.f32.s32 $4, $0;
            cvt.rn.f32.s32 $5, $1;
            cvt.rn.f32.s32 $6, $2;
            cvt.rn.f32.s32 $7, $3;
            max.f32 $4, $4, $9;
            max.f32 $5, $5, $10;
            max.f32 $6, $6, $11;
            max.f32 $7, $7, $12;
"""


@pytest.mark.parametrize("category", CATS)
def test_packed_multiple_outputs_signature(category):
    # official test_inline_asm_packed_multiple_outputs: unpack bytes, int -> float, max with b
    rng = np.random.default_rng(4)
    a = {"positive": rng.integers(0, 256, 8), "boundary": np.array([0, 255, 1, 254, 128, 127, 0, 255]),
         "premise_violation": rng.integers(0, 256, 8)}[category].astype(np.int64)
    b = (rng.standard_normal(8) * 100).astype(np.float32)
    if category == "premise_violation":
        b[5] = np.nan                                      # max with NaN: not modelled (no lane-wise NaN rule)
    asm = PACKED_MULTI.replace("\n", "\\n")
    body = ("    i = tl.arange(0, N)\n    a = tl.load(a_ptr + i)\n    b = tl.load(b_ptr + i)\n"
            f"    (c, d) = tl.inline_asm_elementwise(\"{asm}\", \"=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r,r\", [a, b], "
            "dtype=(tl.int32, tl.float32), is_pure=True, pack=4)\n"
            "    tl.store(c_ptr + i, c)\n    tl.store(d_ptr + i, d)\n")
    bufs = {"a_ptr": ("int8", a.astype(np.uint8).view(np.int8)), "b_ptr": ("fp32", b),
            "c_ptr": ("int32", np.zeros(8)), "d_ptr": ("fp32", np.zeros(8))}
    lo, hi, st = _run("packed_multi_" + category[:3], body, bufs, out_name="c_ptr")
    if category == "premise_violation":
        assert (st != H.ST_OK).all()                       # the program is not evaluated past the unsupported max
        return
    assert (st == H.ST_OK).all() and [int(v) for v in lo] == [int(v) for v in a]
    lo, hi, st = _run("packed_multi_" + category[:3], body, bufs, out_name="d_ptr")
    assert (st == H.ST_OK).all() and [float(v) for v in lo] == [max(float(p), float(q)) for p, q in zip(a, b)]


E4B15_TO_F16 = ("{ .reg .b32 a<2>, b<2>; prmt.b32 a0, 0, $2, 0x5746; and.b32 b0, a0, 0x7f007f00; "
                "and.b32 b1, a0, 0x00ff00ff; and.b32 a1, a0, 0x00800080; shr.b32 b0, b0, 1; add.u32 b1, b1, a1; "
                "lop3.b32 $0, b0, 0x80008000, a0, 0xf8; shl.b32 $1, b1, 7; }")


def _e4b15(byte):
    s, e, m = byte >> 7, (byte >> 3) & 0xF, byte & 7
    v = (2.0 ** (e - 15)) * (1 + m / 8) if e else (2.0 ** -14) * (m / 8)
    return -v if s else v


@pytest.mark.parametrize("category", CATS)
def test_fp8e4b15_to_fp16_conversion_signature(category):
    # the official NVIDIA conversion (triton/language/extra/cuda/utils.py convert_fp8e4b15_to_float16), pack 4
    rng = np.random.default_rng(5)
    x = {"positive": rng.integers(0, 256, 8), "boundary": np.array([0x00, 0x80, 0x7F, 0xFF, 0x01, 0x81, 0x08, 0x78]),
         "premise_violation": rng.integers(0, 256, 8)}[category].astype(np.int64)
    load = "tl.load(x_ptr + i)" if category != "premise_violation" else "tl.load(x_ptr + i, mask=i < 4)"
    body = (f"    i = tl.arange(0, N)\n    x = {load}\n"
            f"    y = tl.inline_asm_elementwise(\"{E4B15_TO_F16}\", \"=r,=r,r\", [x], dtype=tl.float16, is_pure=True, "
            "pack=4)\n    tl.store(out + i, y.to(tl.float32))\n")
    lo, hi, st = _run("e4b15_" + category[:3], body, {"x_ptr": ("int8", x.astype(np.uint8).view(np.int8)),
                                                      "out": ("fp32", np.zeros(8))})
    for j, byte in enumerate(x.tolist()):
        if category == "premise_violation" and j >= 4:
            assert st[j] != H.ST_OK
            continue
        want = _e4b15(byte)
        assert st[j] == H.ST_OK and lo[j] == hi[j] == want and (np.signbit(lo[j]) == (byte >= 0x80) or want != 0), \
            (j, hex(byte), lo[j], want)


def test_environment_read_has_no_reference_value_and_the_program_continues():
    body = ("    i = tl.arange(0, N)\n"
            "    t = tl.inline_asm_elementwise(\"mov.u32 $0, %smid;\", \"=r\", [], dtype=tl.int32, is_pure=False, pack=1)\n"
            "    tl.store(out + i, t + tl.zeros([N], dtype=tl.int32))\n    tl.store(out2 + i, i * 2)\n")
    lo, hi, st = _run("smid", body, {"out": ("int32", np.zeros(8)), "out2": ("int32", np.zeros(8))})
    assert (st != H.ST_OK).all()
    lo, hi, st = _run("smid", body, {"out": ("int32", np.zeros(8)), "out2": ("int32", np.zeros(8))}, out_name="out2")
    assert (st == H.ST_OK).all() and [int(v) for v in lo] == [2 * j for j in range(8)]


def test_memory_effect_inside_asm_is_not_established():
    body = ("    i = tl.arange(0, N)\n    xp = x_ptr + i\n    yp = out + i\n"
            "    tl.inline_asm_elementwise(\"ld.global.b8 $0, [$1]; shl.b32 $0, $0, 3; st.global.b8 [$2], $0;\", "
            "\"=r,l,l\", [xp, yp], dtype=tl.int8, is_pure=False, pack=1)\n")
    x = np.arange(8, dtype=np.int8)
    lo, hi, st = _run("asm_mem", body, {"x_ptr": ("int8", x), "out": ("int8", np.zeros(8))})
    assert (st != H.ST_OK).all()
