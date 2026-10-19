"""Bind read-only region inputs by compiler origin, never by shape or value."""
import math


def bind_inputs(observations, written_storages):
    """Rows contain launch, buffer, arg, provenance and proven write_only.

    Source/capture correspondence and inter-launch continuity are prerequisites.
    Read-modify-write regions require a separate explicit pre-state binding.
    """
    inputs, descriptions, first_seen = {}, {}, {}
    for row in observations:
        arg, info = row["arg"], row["provenance"]
        identity = arg.storage_id
        if identity is None:
            raise ValueError("input storage identity missing")
        first_seen.setdefault(identity, row["launch"])
        if identity in written_storages:
            if row["launch"] == first_seen[identity] and row["write_only"] is not True:
                raise ValueError("first use of a written storage is not write-only; explicit pre-state required")
            continue
        if info.get("buffer") != row["buffer"] or info.get("role") != "input":
            raise ValueError("external input and compiler buffer role disagree")
        name = info.get("origin_node")
        if not isinstance(name, str) or not name or name not in info.get("origins", []):
            raise ValueError("external input requires an explicit compiler origin node")
        if arg.before is None or arg.window is not None:
            raise ValueError("external input pre-state is incomplete")
        shape, stride = tuple(arg.shape), tuple(arg.stride)
        if (len(shape) != len(stride) or any(type(v) is not int or v < 0 for v in (*shape, *stride))
                or type(arg.element_size) is not int or arg.element_size <= 0):
            raise ValueError("external input layout is unsupported")
        offset = arg.data_ptr - arg.storage_ptr
        span = 1 + sum((n - 1) * s for n, s in zip(shape, stride)) if math.prod(shape) else 0
        if offset < 0 or offset % arg.element_size or offset + span * arg.element_size > arg.storage_nbytes:
            raise ValueError("external input view exceeds captured storage")
        description = {"launch": row["launch"], "parameter": arg.name, "buffer": row["buffer"],
                       "storage_id": identity, "dtype": arg.dtype, "shape": list(shape),
                       "stride": list(stride), "byte_offset": offset}
        if name in inputs:
            keys = ("storage_id", "dtype", "shape", "stride", "byte_offset")
            if any(description[k] != descriptions[name][k] for k in keys):
                raise ValueError("one task input has conflicting storage or view mappings")
        else:
            inputs[name], descriptions[name] = arg, description
    if not inputs:
        raise ValueError("region has no bound external task inputs")
    return inputs, descriptions
