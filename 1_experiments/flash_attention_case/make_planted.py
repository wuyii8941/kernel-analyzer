"""Generate planted_fwd.py: the tutorial forward kernel (tutorial_fwd.py) with one planted defect per variant, each a
line-level edit of the verbatim source (so the rest of the kernel is the tutorial's):

  tf32      fp32 inputs, no cast of P: both tl.dot run at the default tf32 precision (numerical bias)
  scale     qk_scale *= 1.44 instead of 1.44269504 (semantic bias: the declared program is not the specification)
  l_bf16    the softmax denominator l_i kept in bf16 across iterations (low-precision accumulation)
  acc_fp16  the output accumulator in fp16 (a tensor-core fp16 accumulator: outside the lowering rules)
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

EDITS = {
    "tf32": [("    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16\n", "    dtype = tl.float32\n")],
    "scale": [("    qk_scale *= 1.44269504  # 1/log(2)\n", "    qk_scale *= 1.44  # 1/log(2), truncated\n")],
    "l_bf16": [("        l_i = l_i * alpha + l_ij\n",
                "        l_i = (l_i * alpha + l_ij).to(tl.bfloat16).to(tl.float32)\n")],
    "acc_fp16": [("    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)\n",
                  "    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float16)\n"),
                 ("    acc = acc / l_i[:, None]\n", "    acc = acc.to(tl.float32) / l_i[:, None]\n"),
                 ("            acc = acc * alpha[:, None]\n", "            acc = (acc * alpha[:, None]).to(tl.float16)\n"),
                 ("        acc = tl.dot(p, v, acc)\n", "        acc = tl.dot(p, v, acc, out_dtype=tl.float16)\n")],
}

TAIL = '''

def cases(run_case, a, progs):
    """Case name -> thunk (fa_case.py)."""
    return {
        "planted_tf32": lambda: run_case("planted_tf32", _attn_fwd_tf32, "float32", 1, 2, a.N, 64, False, 128, 64,
                                         progs),
        "planted_scale": lambda: run_case("planted_scale", _attn_fwd_scale, "float16", 1, 2, a.N, 64, False, 128, 64,
                                          progs),
        "planted_l_bf16": lambda: run_case("planted_l_bf16", _attn_fwd_l_bf16, "float16", 1, 2, a.N, 64, False, 128,
                                           16, progs),
        "planted_acc_fp16": lambda: run_case("planted_acc_fp16", _attn_fwd_acc_fp16, "float16", 1, 2, a.N, 64, False,
                                             128, 64, progs),
    }
'''


def main():
    src = (HERE / "tutorial_fwd.py").read_text()
    head, body = src.split("@triton.jit\ndef _attn_fwd_inner(", 1)
    body = "@triton.jit\ndef _attn_fwd_inner(" + body
    mk_start = body.index("@triton.jit\ndef _maybe_make_tensor_desc(")
    fwd_start = body.index("@triton.jit\ndef _attn_fwd(sm_scale")
    inner, mk, fwd = body[:mk_start], body[mk_start:fwd_start], body[fwd_start:]
    out = [head.replace('"""The forward kernel', '"""Planted variants (make_planted.py) of the forward kernel', 1), mk]
    for name, edits in EDITS.items():
        i, f = inner, fwd
        for old, new in edits:
            hits = i.count(old) + f.count(old)
            assert hits >= 1, (name, old)
            i, f = i.replace(old, new), f.replace(old, new)
        i = i.replace("def _attn_fwd_inner(", f"def _attn_fwd_inner_{name}(")
        f = f.replace("def _attn_fwd(sm_scale", f"def _attn_fwd_{name}(sm_scale").replace(
            "= _attn_fwd_inner(", f"= _attn_fwd_inner_{name}(")
        out += [f"# ---- variant {name}\n", i, f]
    (HERE / "planted_fwd.py").write_text("".join(out).rstrip() + "\n" + TAIL)


if __name__ == "__main__":
    main()
