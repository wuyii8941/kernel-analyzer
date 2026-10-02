"""Collect TTIR of real Triton launches into a version-tagged corpus.

Runs the exercise kernels, a few TorchInductor workloads and, when the
packages are importable, Liger and torchao kernels.  Only compiled IR and
launch configuration are stored (no operands).

    python scripts/build_ttir_corpus.py --out results/reference_eval/ttir_corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402


def exercise_kernels():
    from scripts import reference_eval_kernels as k

    dev = "cuda"
    n = 1000
    x = torch.randn(n, device=dev)
    y = torch.empty_like(x)
    a, b, c, d = (torch.randn(n, device=dev) for _ in range(4))
    k.scale_masked[(4,)](x, y, n, 2.0, BLOCK=256)
    k.add_then_mul[(4,)](a, b, c, y, n, BLOCK=256)
    k.sequential_sum4[(4,)](a, b, c, d, y, n, BLOCK=256)
    rows = torch.randn(8, 100, device=dev)
    k.softmax_rows[(8,)](rows, torch.empty_like(rows), 100, 100, BLOCK=128)
    w, bb = torch.randn(100, device=dev), torch.randn(100, device=dev)
    k.layernorm_rows[(8,)](rows, w, bb, torch.empty_like(rows), 100, 1e-5, BLOCK=128)
    for precision in ("tf32", "ieee", "tf32x3"):
        A = torch.randn(64, 48, device=dev)
        B = torch.randn(48, 32, device=dev)
        C = torch.empty(64, 32, device=dev)
        k.matmul[(2, 2)](A, B, C, 64, 32, 48, *A.stride(), *B.stride(), *C.stride(),
                         BM=32, BN=16, BK=16, PRECISION=precision)
    A16 = torch.randn(64, 48, device=dev, dtype=torch.bfloat16)
    B16 = torch.randn(48, 32, device=dev, dtype=torch.bfloat16)
    C16 = torch.empty(64, 32, device=dev, dtype=torch.bfloat16)
    k.matmul[(2, 2)](A16, B16, C16, 64, 32, 48, *A16.stride(), *B16.stride(), *C16.stride(),
                     BM=32, BN=16, BK=16, PRECISION="ieee")
    out = torch.zeros(4, device=dev)
    k.atomic_accumulate[(4,)](x, out, y, n, BLOCK=256, USE_OLD=False)
    k.atomic_accumulate[(4,)](x, out, y, n, BLOCK=256, USE_OLD=True)
    k.branch_on_scalar[(4,)](x, y, 0.0, n, BLOCK=256)
    k.conversions[(4,)](x, torch.empty(n, device=dev, dtype=torch.float16),
                        torch.empty(n, device=dev, dtype=torch.bfloat16),
                        torch.empty(n, device=dev, dtype=torch.float8_e5m2), y, n, BLOCK=256)
    k.elementary[(4,)](x, y, n, BLOCK=256)
    k.nan_rules[(4,)](x, y, n, BLOCK=256)
    k.scan_and_argmax[(1,)](x, y, torch.empty(1, device=dev, dtype=torch.int32), 200, BLOCK=256)
    k.bit_level[(4,)](x, y, n, BLOCK=256)
    k.inline_asm[(4,)](x, y, n, BLOCK=256)
    k.while_loop[(4,)](x, y, n, BLOCK=256)
    k.store_load_chain[(4,)](x, torch.empty_like(x), y, n, BLOCK=256)
    k.exp_sum_divide[(1,)](x, y, 200, BLOCK=256)


def inductor_workloads():
    torch._dynamo.reset()
    dev = "cuda"

    @torch.compile
    def chunk_sum(c0, c1, c2, c3, c4, c5, c6, c7):
        return ((((((c0 + c1) + c2) + c3) + c4) + c5) + c6) + c7

    chunk_sum(*(torch.randn(4096, 64, device=dev) for _ in range(8)))

    @torch.compile
    def softmax_ce(logits, target):
        return torch.nn.functional.cross_entropy(logits, target)

    logits = torch.randn(64, 1000, device=dev, requires_grad=True)
    softmax_ce(logits, torch.randint(0, 1000, (64,), device=dev)).backward()

    @torch.compile
    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w

    rms(torch.randn(64, 512, device=dev, dtype=torch.bfloat16), torch.randn(512, device=dev, dtype=torch.bfloat16))

    p = torch.nn.Parameter(torch.randn(8192, device=dev))
    opt = torch.optim.AdamW([p], lr=1e-3, foreach=False)

    @torch.compile
    def step():
        opt.step()

    for _ in range(2):
        p.grad = torch.randn_like(p)
        step()


def liger_workloads():
    from liger_kernel.transformers.functional import liger_cross_entropy
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    from liger_kernel.transformers.rms_norm import LigerRMSNorm

    dev = "cuda"
    logits = torch.randn(32, 512, device=dev, requires_grad=True)
    liger_cross_entropy(logits, torch.randint(0, 512, (32,), device=dev)).backward()
    h = torch.randn(64, 128, device=dev, requires_grad=True)
    lin = torch.nn.Linear(128, 1000, bias=False, device=dev)
    LigerFusedLinearCrossEntropyLoss()(lin.weight, h, torch.randint(0, 1000, (64,), device=dev)).backward()
    norm = LigerRMSNorm(256).to(dev)
    norm(torch.randn(16, 256, device=dev, requires_grad=True)).sum().backward()


def torchao_workloads():
    from torchao.optim import AdamW8bit

    torch._dynamo.reset()
    p = torch.nn.Parameter(torch.randn(4096, 256, device="cuda"))
    opt = AdamW8bit([p], lr=1e-3)
    for _ in range(2):
        p.grad = torch.randn_like(p)
        opt.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results/reference_eval/ttir_corpus")
    args = parser.parse_args()
    import triton

    env_tag = f"py{sys.version_info.major}{sys.version_info.minor}_triton{triton.__version__}"
    out = args.out / env_tag
    out.mkdir(parents=True, exist_ok=True)
    groups = {"exercise": exercise_kernels, "inductor": inductor_workloads,
              "liger": liger_workloads, "torchao": torchao_workloads}
    manifest = {"environment": env_tag, "torch": torch.__version__, "groups": {}, "kernels": []}
    seen = set()
    for group, fn in groups.items():
        recorder = TritonLaunchRecorder(copy_tensors=False)
        try:
            with recorder:
                fn()
                torch.cuda.synchronize()
        except ImportError as exc:
            manifest["groups"][group] = f"skipped: {exc}"
            continue
        manifest["groups"][group] = f"{len(recorder.launches)} launches"
        for launch in recorder.launches:
            ttir = launch.asm["ttir"]
            digest = hashlib.sha256(ttir.encode()).hexdigest()[:16]
            if digest in seen:
                continue
            seen.add(digest)
            name = f"{group}__{launch.kernel_name}__{digest}.ttir"
            (out / name).write_text(ttir)
            manifest["kernels"].append({
                "file": name, "group": group, "kernel": launch.kernel_name, "grid": list(launch.grid),
                "args": [{"name": a.name, "kind": a.kind, "constexpr": a.constexpr,
                          "signature": a.signature_type, "value": a.value if a.constexpr else None}
                         for a in launch.args],
                "libtriton_sha256": launch.libtriton_sha256,
            })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["groups"], indent=2), len(manifest["kernels"]), "unique kernels")


if __name__ == "__main__":
    main()
