"""Map explicit wrapper returns to captured storage, retaining unresolved effects."""
import copy
import math
import re


def bind_outputs(task_graph, returns, observations, written_storages):
    """observations: (launch index, wrapper buffer name, CapturedArg).

    Call sites, scalar arguments, graph ownership and storage continuity must
    already have been checked. This function does not certify those conditions.
    Returned tensor views are retained, including duplicate return positions.
    """
    nodes = {n["name"]: n for n in task_graph["nodes"]}
    by_buffer, final_storage = {}, {}
    for index, buffer, arg in observations:
        if arg.storage_id is None:
            raise ValueError("captured storage identity missing")
        old = by_buffer.get(buffer)
        if old is not None and old[1].storage_id != arg.storage_id:
            raise ValueError("buffer name reused for a different storage needs lifetime mapping")
        if old is None or index >= old[0]:
            by_buffer[buffer] = (index, arg)
        old = final_storage.get(arg.storage_id)
        if old is None or index >= old[0]:
            final_storage[arg.storage_id] = (index, arg)
    outputs, passthrough, absent, constants, covered = [], [], [], [], set()
    for returned in returns:
        if "constant" in returned:
            if set(returned) != {"position", "constant"} or returned["constant"] is not None:
                raise ValueError("unsupported constant return mapping")
            constants.append(returned)
            continue
        if returned["buffer"] not in by_buffer:
            absent.append(returned)
            continue
        _, source = by_buffer[returned["buffer"]]
        if source.storage_id not in written_storages:
            passthrough.append(returned)
            continue
        meta = nodes[returned["node"]].get("value", {})
        if meta.get("kind") != "tensor" or meta.get("dtype") != "torch." + source.dtype:
            raise ValueError("task output and capture dtype disagree or are unsupported")
        try:
            if any(not isinstance(n, str) or not re.fullmatch(r"\d+", n) for n in meta["shape"]):
                raise ValueError("nonliteral shape")
            expected_shape = tuple(int(n) for n in meta["shape"])
        except (ValueError, TypeError, KeyError):
            raise ValueError("task output requires a concrete shape") from None
        view = returned["view"]
        mapped = copy.copy(source)
        if view is not None:
            if view["offset"] != 0:
                raise ValueError("return view offset is unsupported")
            mapped.shape, mapped.stride = tuple(view["shape"]), tuple(view["stride"])
        if tuple(mapped.shape) != expected_shape:
            raise ValueError("task output and returned view shape disagree")
        if (len(mapped.shape) != len(mapped.stride)
                or any(type(n) is not int or n < 0 for n in (*mapped.shape, *mapped.stride))):
            raise ValueError("output layout is unsupported")
        offset = mapped.data_ptr - mapped.storage_ptr
        span = (1 + sum((n - 1) * s for n, s in zip(mapped.shape, mapped.stride))) if math.prod(mapped.shape) else 0
        if (type(mapped.element_size) is not int or mapped.element_size <= 0 or offset < 0
                or offset % mapped.element_size or offset + span * mapped.element_size > mapped.storage_nbytes):
            raise ValueError("returned view exceeds captured storage")
        index, final = final_storage[source.storage_id]
        if final.after is None or final.window is not None:
            raise ValueError("final output storage is incomplete")
        mapped.after = final.after
        outputs.append({"return": returned, "last_observed_launch": index, "arg": mapped})
        covered.add(source.storage_id)
    return {"outputs": outputs, "passthrough_returns": passthrough, "unobserved_returns": absent,
            "constant_returns": constants,
            "unreturned_written_storages": sorted(set(written_storages) - covered),
            "binding_status": "not_established", "semantic_verdict": "not_established"}
