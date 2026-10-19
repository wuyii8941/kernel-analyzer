"""Check storage continuity within one recorder run, before task-region binding.

This checks captured observations only. It does not prove that all intervening
device work was recorded, or identify the mathematical outputs of the region.
"""
import numpy as np


def check_sequence(launches, *, same_recorder_run=False, storages_retained=False):
    base = {"semantic_verdict": "not_established", "binding_status": "not_established"}

    def refuse(reason):
        return dict(base, status="not_established", reason=reason)

    if same_recorder_run is not True or storages_retained is not True:
        return refuse("common recorder run and retained storage lifetimes must be established")
    if not launches:
        return refuse("empty launch sequence")
    indices = [x.index for x in launches]
    if any(type(i) is not int or i < 0 for i in indices) or indices != list(range(indices[0], indices[0] + len(indices))):
        return refuse("launch indices are not consecutive and ordered")
    previous, links = {}, []
    for launch in launches:
        if launch.executed is not True or not launch.implicit_recorded or launch.implicit:
            return refuse("execution or complete direct-memory capture not established")
        current = {}
        for arg in launch.args:
            if arg.kind != "tensor":
                continue
            if arg.storage_id is None or arg.storage_ptr is None or arg.window is not None:
                return refuse("storage identity or complete storage capture missing")
            if arg.before is None or arg.after is None:
                return refuse("storage pre-state or post-state missing")
            before, after = np.asarray(arg.before), np.asarray(arg.after)
            if (before.dtype != np.uint8 or after.dtype != np.uint8 or before.ndim != 1 or after.ndim != 1
                    or before.size != arg.storage_nbytes or after.size != arg.storage_nbytes):
                return refuse("storage byte capture has an invalid type or size")
            state = (arg.storage_ptr, arg.storage_nbytes, before, after, arg.name)
            if arg.storage_id in current:
                old = current[arg.storage_id]
                if (old[:2] != state[:2] or not np.array_equal(old[2], before)
                        or not np.array_equal(old[3], after)):
                    return refuse("aliased parameters disagree on captured storage")
            else:
                current[arg.storage_id] = state
        for identity, state in current.items():
            if identity in previous:
                step, old = previous[identity]
                if old[:2] != state[:2]:
                    return refuse("storage identity has changed address or size")
                if not np.array_equal(old[3], state[2]):
                    return refuse(f"storage changed between recorded launches {step} and {launch.index}")
                links.append({"from_launch": step, "to_launch": launch.index,
                              "from_parameter": old[4], "to_parameter": state[4],
                              "storage_id": identity, "bytes": state[1]})
            previous[identity] = (launch.index, state)
    return dict(base, status="captured storage sequence checked", launch_indices=indices,
                storages=len(previous), links=links,
                limitation="same recorder run required; intervening device work and task outputs are not certified")
