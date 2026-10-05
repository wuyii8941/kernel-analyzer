"""Runtime patch for B016 mechanism 1: an atomic_add whose index is provably constant gets no mask.

After TritonKernel.store emits `tl.atomic_add(ptr + (idx), value, None, sem='relaxed')` in a pointwise kernel, the
patch replaces the mask `None` by the iteration mask `xmask`, so padding lanes (xindex >= xnumel) no longer add.
Mechanism 2 (a reader fused ahead of the atomics) is not addressed.

    import B016_mask_patch; B016_mask_patch.apply()
or, for tool runs:  KA_PRELOAD=bugs/repro/B016_mask_patch.py python scripts/tool_spec_check.py ...
"""
PATCHED = []


def apply():
    from torch._inductor.codegen.triton import TritonKernel

    if PATCHED:
        return
    orig = TritonKernel.store

    def store(self, name, index, value, mode=None):
        n_before = len(self.stores._lines)
        out = orig(self, name, index, value, mode)
        if mode == "atomic_add" and not self.inside_reduction:
            for ln in self.stores._lines[n_before:]:
                text = getattr(ln, "line", None)
                if isinstance(text, str) and text.startswith("tl.atomic_add(") and text.endswith(", None, sem='relaxed')"):
                    ln.line = text[: -len(", None, sem='relaxed')")] + ", xmask, sem='relaxed')"
                    PATCHED.append(name)
        return out

    TritonKernel.store = store
    PATCHED.append("<installed>")
