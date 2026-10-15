"""The tool as a user would call it: one audit() per flash attention variant (tutorial and planted defects), with the
reference programs of fa_inputs (torch scaled_dot_product_attention for O, the base-2 logsumexp for M, float64).  Reports: .cache/flash_attention_case/audit_<case>.{txt,json}"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "2_tool/src"))
sys.path.insert(0, str(HERE))
OUT = ROOT / ".cache/flash_attention_case"

# case -> (module attribute, dtype, N, BLOCK_N, causal)
CASES = {
    "tutorial_fp16": ("tutorial_fwd._attn_fwd", "float16", 1024, 64, False),
    "tutorial_fp16_causal": ("tutorial_fwd._attn_fwd", "float16", 1024, 64, True),
    "planted_tf32": ("planted_fwd._attn_fwd_tf32", "float32", 1024, 64, False),
    "planted_scale": ("planted_fwd._attn_fwd_scale", "float16", 1024, 64, False),
    "planted_l_bf16": ("planted_fwd._attn_fwd_l_bf16", "float16", 2048, 16, False),
    "planted_acc_fp16": ("planted_fwd._attn_fwd_acc_fp16", "float16", 1024, 64, False),
    "planted_mask_diag": ("planted_fwd._attn_fwd_mask_diag", "float16", 1024, 64, True),
}
REFERENCE = "O：torch.nn.functional.scaled_dot_product_attention；M：logsumexp(q kᵀ · sm_scale) / ln 2（教程以 2 为底存 M）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--programs", default="",
                    help="program indices, or 'all' (default: the tool's sample, incl. the last block)")
    ap.add_argument("--N", type=int, default=0, help="sequence length (default: the case's)")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    import importlib

    import torch
    import triton

    from fa_inputs import make_inputs, reference_m, reference_o, spec_attention
    from kernel_analyzer.reference_eval.audit import audit
    mod_name, dtype, N, block_n, causal = CASES[a.case]
    N = a.N or N
    kernel = getattr(importlib.import_module(mod_name.split(".")[0]), mod_name.split(".")[1])
    Z, H, D, block_m = 1, 2, 64, 128
    sm_scale = 1.0 / math.sqrt(D)
    q, k, v = make_inputs(getattr(torch, dtype), Z, H, N, D, 0)
    o = torch.zeros_like(q)
    M = torch.zeros(Z, H, N, device="cuda")
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    grid = (triton.cdiv(N, block_m), Z * H, 1)

    def launch():
        kernel[grid](sm_scale, M, Z, H, q, k, v, o, N_CTX=N, HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
                     FP8_OUTPUT=False, STAGE=3 if causal else 1, warp_specialize=False, IS_HOPPER=False,
                     num_stages=2, num_warps=4)
    progs = None if not a.programs else "all" if a.programs == "all" else \
        [(int(p) % grid[0], int(p) // grid[0], 0) for p in a.programs.split(",")]
    rep = audit(launch, spec={"desc_o": lambda: reference_o(q, k, v, sm_scale, causal),
                              "M": lambda: reference_m(q, k, sm_scale, causal)}, programs=progs,
                names={"desc_o": "O", "M": "M（logsumexp）"}, reference=REFERENCE)
    # the two references for O against each other (registration: max |difference| < 1e-12)
    rep["reference_crosscheck_max_abs"] = float((reference_o(q, k, v, sm_scale, causal)
                                                 - spec_attention(q, k, v, sm_scale, causal)).abs().max())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"audit_{a.case}{a.suffix}.txt").write_text(rep["text"])
    (OUT / f"audit_{a.case}{a.suffix}.json").write_text(json.dumps({k_: v_ for k_, v_ in rep.items() if k_ != "text"},
                                                         indent=1, ensure_ascii=False, default=str))
    print(rep["text"])


if __name__ == "__main__":
    main()
