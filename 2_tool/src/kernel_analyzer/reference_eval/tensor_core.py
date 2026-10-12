"""Bit-exact model of a tensor-core ``tt.dot`` on the locked build (Triton 3.6.0, sm_86: mma.sync m16n8k8 for tf32,
m16n8k16 for bf16 / f16, fp32 accumulator).

The rule was fitted on constructed inputs and then checked bit for bit on held-out inputs, including tutorial
matmuls with K up to 4864 and several block shapes (1_experiments/composition_study: mma_probe.py, dot_check.py):

* operand conversion: tf32 takes the fp32 operand cut toward zero to 11 significant bits (no cvt instruction is
  emitted; the tensor core ignores the low 13 bits); bf16 / f16 operands are used as they are;
* products are exact;
* the k products are taken in k order in groups of G (tf32: 4, bf16 / f16: 8).  A group and the incoming
  accumulator are aligned to the largest exponent E among them -- for a product the exponent is e(a) + e(b), the
  sum of the operands' exponents, not the exponent of the product itself -- every term is cut toward zero at
  2^(E - 24), the cut terms are added exactly, and the sum is cut toward zero to 24 significant bits.  The next group
  starts from that accumulator, so a dot over K chains K / G such steps.

* a subnormal f16 operand enters the alignment with the format's minimum exponent (-14), its value exact (checked
  with 37% subnormal operands).

Only finite operands (normal or zero for tf32 / bf16), products and results in the fp32 normal range are inside the
model; ``mma_chain`` returns a mask of the elements it covers and the caller marks the rest as not emulable.  tf32x3, bf16x3 / bf16x6 and an fp16
accumulator are different lowerings and are not modelled here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

GROUP = {"tf32": 4, "bf16": 8, "f16": 8}
SIGNIFICAND = {"tf32": 11, "bf16": 8, "f16": 11}
MIN_NORMAL_EXP = {"tf32": -126, "bf16": -126, "f16": -14}
EXTRA_BITS = 1


def mma_format(a_elem: str, b_elem: str, c_elem: str, precision: str) -> Optional[str]:
    """The modelled mma input format of a tt.dot, or None."""
    if c_elem != "f32" or a_elem != b_elem:
        return None
    if a_elem == "f32" and precision.strip().startswith("tf32") and not precision.strip().startswith("tf32x3"):
        return "tf32"
    if a_elem in ("bf16", "f16"):
        return a_elem
    return None


def round_sig(x: np.ndarray, p: int, mode: str) -> np.ndarray:
    """float64 values to p significant bits: "rz" toward zero, "rne" nearest even, "rna" nearest away."""
    m, e = np.frexp(x)
    s = np.ldexp(m, p)
    if mode == "rz":
        r = np.trunc(s)
    elif mode == "rne":
        r = np.rint(s)
    elif mode == "rna":
        r = np.sign(s) * np.floor(np.abs(s) + 0.5)
    else:
        raise ValueError(mode)
    return np.ldexp(r, e - p)


def _exponent(x: np.ndarray) -> np.ndarray:
    _, e = np.frexp(x)
    return e.astype(np.int64) - 1


def mma_chain(a: np.ndarray, b: np.ndarray, c: np.ndarray, fmt: str, conv_mode: str = "rz",
              final: str = "rz") -> tuple:
    """dot(a, b) + c under the model.  a (..., M, K), b (..., K, N), c (..., M, N) float64 values of the declared
    operand types.  Returns (result, covered): covered is False where an operand, a product, the accumulator or the
    result leaves the model's domain (non-finite or subnormal).  ``conv_mode`` / ``final`` other than "rz" give the
    counterfactual rules used for attribution (round-to-nearest conversion or accumulation)."""
    p = SIGNIFICAND[fmt]
    G = GROUP[fmt]
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if fmt == "tf32":
        a, b = round_sig(a, p, conv_mode), round_sig(b, p, conv_mode)
    acc = np.array(c, dtype=np.float64)
    emin = MIN_NORMAL_EXP[fmt]
    sub_ok = fmt == "f16"
    ok_a = np.isfinite(a) & ((a == 0) | sub_ok | (_exponent(np.where(a == 0, 1.0, a)) >= emin))
    ok_b = np.isfinite(b) & ((b == 0) | sub_ok | (_exponent(np.where(b == 0, 1.0, b)) >= emin))
    covered = np.all(ok_a, axis=-1)[..., :, None] & np.all(ok_b, axis=-2)[..., None, :]
    a = np.where(ok_a, a, 0.0)
    b = np.where(ok_b, b, 0.0)
    ea = np.maximum(_exponent(np.where(a == 0, 1.0, a)), emin)      # the exponent field: subnormals at emin
    eb = np.maximum(_exponent(np.where(b == 0, 1.0, b)), emin)
    K = a.shape[-1]
    for s in range(0, K, G):
        P = a[..., :, s:s + G, None] * b[..., None, s:s + G, :]                      # (..., M, G, N), exact
        pe = ea[..., :, s:s + G, None] + eb[..., None, s:s + G, :]
        T = np.concatenate([acc[..., :, None, :], P], axis=-2)
        nz = T != 0
        e = np.concatenate([_exponent(np.where(acc == 0, 1.0, acc))[..., :, None, :], pe], axis=-2)
        covered &= ~np.any(nz[..., 1:, :] & (pe < -126), axis=-2)                  # product below fp32 normal range
        E = np.where(nz, e, -100000).max(axis=-2)
        allzero = ~nz.any(axis=-2)
        q = np.where(allzero, 0, E - 23 - EXTRA_BITS)
        S = np.trunc(np.ldexp(T, -q[..., None, :])).sum(axis=-2)
        out = np.ldexp(S, q)
        acc = np.where(allzero, acc, round_sig(out, 24, final))
        tiny = (acc != 0) & (np.abs(acc) < 2.0 ** -126)
        covered &= np.isfinite(acc) & ~tiny & (np.abs(acc) <= np.finfo(np.float32).max)
        acc = np.where(np.isfinite(acc), acc, 0.0)
    return acc, covered
