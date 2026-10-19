"""Device replay and fixed-rule emulation for an already bound captured region."""
import math

import dynamic_three_way as D
from kernel_spec.num import NotEstablished


def check(launches, directories, *, max_programs=2048):
    import torch
    from kernel_analyzer.reference_eval.capture import load_compiled, replay
    from kernel_analyzer.reference_eval import emulate as E

    if len(launches) != len(directories) or not launches:
        raise ValueError("one saved binary directory is required per launch")
    if type(max_programs) is not int or max_programs <= 0:
        raise ValueError("program budget must be a positive integer")
    if any(len(l.grid) != 3 or any(type(n) is not int or n <= 0 for n in l.grid) for l in launches):
        raise ValueError("complete positive program grids are required")
    if sum(math.prod(l.grid) for l in launches) > max_programs:
        raise NotEstablished("device correspondence exceeds the complete-grid program budget")
    report = {"status": "checked", "launches": [], "ungrouped": [], "swapped": [],
              "scope": "captured inputs only; replay bytes and IR write events, not a device memory access trace"}
    for launch, directory in zip(launches, directories):
        if launch.executed is not True:
            raise NotEstablished("captured launch execution is not established")
        expected = {a.storage_ptr: a.after for a in launch.args if a.kind == "tensor"}
        binary = load_compiled(directory)
        replays = []
        for _ in range(2):
            actual = replay(launch, binary)
            replays.append(set(actual) == set(expected) and
                           all(torch.equal(actual[p], expected[p]) for p in expected))
        result, _ = E.emulate(launch, ungrouped=(), swapped=())
        outputs = D.write_parameters(launch)
        writes = (D.inspect_reference_writes(launch, outputs, result) if outputs is not None
                  else {"outcome": "not_established", "note": "unknown IR write parameters"})
        comparisons = []
        for name, written, masks in E._verification_outputs(result, launch):
            comparisons.append({"storage": name, "written": int(written.sum()),
                                "emulated": int((written & masks["emulated"]).sum()),
                                "bit_equal": int((written & masks["bit_equal"]).sum())})
        passed = (all(replays) and writes["outcome"] == "checked" and bool(comparisons)
                  and all(c["written"] == c["emulated"] == c["bit_equal"] for c in comparisons))
        report["launches"].append({"launch": launch.index, "replays_equal_capture": replays,
                                   "IR_writes": writes, "outputs": comparisons, "checked": passed})
        if not passed:
            report["status"] = "not_established"
    return report
