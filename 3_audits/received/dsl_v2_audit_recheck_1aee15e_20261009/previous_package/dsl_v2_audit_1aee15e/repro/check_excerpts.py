"""Manually transcribed from check.py at commit 1aee15e9df7a45f434699b7016056e6650fdae99.
Blob ead94d221112873ecb3771fa53ddaf0f55f607b5. Only the provenance function;
read through the GitHub connector, not a locally checked-out production module.
"""

def torch_intermediates(launches, seq, inp, digests_before=None):
    """Per written storage: the float buffers upstream of it (through the recorded launches) that the reference
    loaded but that are neither inputs of the case nor written by an earlier recorded launch, i.e. produced by a
    torch / ATen op in between.  K_R treats their captured values as exact inputs, so an output depending on them
    carries K's upstream numerical error in K_R, and its e_sem is mixed rather than purely semantic."""
    import hashlib

    def digest(a):
        return hashlib.sha1(np.ascontiguousarray(np.asarray(a)).view(np.uint8).tobytes()).hexdigest()

    def tensors(obj):
        if torch.is_tensor(obj):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from tensors(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from tensors(v)

    inputs = digests_before if digests_before is not None else \
        {digest(v.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy()) for v in tensors(inp)
         if v.is_floating_point()}
    patterns = {}
    for v in tensors(inp):
        if v.is_floating_point():
            n = v.element_size()
            u = v.detach().contiguous().cpu().reshape(-1).view({2: torch.int16, 4: torch.int32, 8: torch.int64}[n]).numpy()
            patterns.setdefault(n, []).append(np.unique(u))
    patterns = {n: np.unique(np.concatenate(p)) for n, p in patterns.items()}
    sizes = {"float16": 2, "bfloat16": 2, "float32": 4, "fp32": 4, "float64": 8, "fp64": 8, "fp16": 2, "bf16": 2}

    def data_movement(raw, dtype):
        n = sizes.get(str(dtype).replace("torch.", ""))
        if n is None or n not in patterns:
            return False
        b = np.ascontiguousarray(np.asarray(raw)).view(np.uint8)
        if b.size % n:
            return False
        el = b.view({2: np.int16, 4: np.int32, 8: np.int64}[n])
        return bool(el.size) and bool(np.isin(el, patterns[n]).all())
    deps = {}
    last_after = {}
    for i, (l, ref) in enumerate(zip(launches, seq.launches)):
        tensors = [a for a in l.args if a.kind == "tensor"]
        upstream = set()
        for a in tensors:
            if a.storage_ptr not in ref.loaded_any:
                continue
            raw = a.before.numpy() if hasattr(a.before, "numpy") else a.before
            if a.storage_ptr in deps and last_after.get(a.storage_ptr) == digest(raw):
                upstream |= deps[a.storage_ptr]
            elif a.storage_ptr in ref.loaded and str(a.dtype).startswith(("float", "bfloat")):
                if digest(raw) not in inputs and np.asarray(raw).view(np.uint8).any():
                    tag = " [copy of inputs]" if data_movement(raw, a.dtype) else ""
                    upstream.add(f"L{i}:{l.kernel_name[:40]}:{a.name}{tag}")
        stored = getattr(ref, "stored", set())
        for a in tensors:
            before = np.asarray(a.before.numpy() if hasattr(a.before, "numpy") else a.before)
            after = np.asarray(a.after.numpy() if hasattr(a.after, "numpy") else a.after)
            if a.storage_ptr in stored or before.shape != after.shape or \
                    not np.array_equal(before.view(np.uint8), after.view(np.uint8)):
                deps[a.storage_ptr] = set(upstream)
                last_after[a.storage_ptr] = digest(after)
    return {k: sorted(v) for k, v in deps.items()}
