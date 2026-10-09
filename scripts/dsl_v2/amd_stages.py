"""TTIR -> TTGIR with the official AMD backend's own stages (DSL v2 increment 12).

The AMD backend's LLVM code generation needs a newer glibc than this host, so ``triton.compile`` cannot finish for hip
targets here; the stages up to TTGIR are the official pass pipelines and run unchanged.  Nothing after TTGIR runs.
"""
from __future__ import annotations


def amd_ttgir(path: str, arch: str) -> str:
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget, Language
    from triton.compiler.compiler import make_backend
    backend = make_backend(GPUTarget("hip", arch, 32 if arch.startswith("gfx1") else 64))
    options = backend.parse_options({})
    stages = {}
    backend.add_stages(stages, options, Language.TRITON)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    mod = ir.parse_mlir_module(path, context)
    mod.context = context
    metadata = {}
    for name in ("ttir", "ttgir"):
        mod = stages[name](mod, metadata)
    return str(mod)
