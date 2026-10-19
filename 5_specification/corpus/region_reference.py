"""Complete K-versus-captured-graph comparison for an already mapped functional region.

Callers supply logical arrays from validated capture layouts. This does not
establish an independent task f, source independence, device effects, or semantic-error certification.
"""
import math

import dynamic_three_way as D
from kernel_spec import aten as A, lazy as L
from kernel_spec.num import Invalid, NotEstablished, Undefined, Unsupported


def compare_task(task, inputs, outputs, *, max_elements, max_evaluations):
    if any(type(n) is not int or n <= 0 for n in (max_elements, max_evaluations)):
        raise Invalid("positive integer output and scalar evaluation budgets are required")
    if any(not isinstance(x, str) for x in task["outputs"]):
        raise Unsupported("region comparison requires explicitly selected tensor output nodes")
    declared = set(task["outputs"])
    if not outputs or {o["node"] for o in outputs} != declared:
        raise Invalid("all declared task outputs must have observations")
    positions = [o["position"] for o in outputs]
    if any(type(p) is not int or p < 0 for p in positions) or len(set(positions)) != len(positions):
        raise Invalid("observed return positions must be distinct nonnegative integers")
    total = sum(math.prod(o["values"].shape) for o in outputs)
    report = {"complete": False, "outcome": "not_established", "total_elements": total,
              "max_elements": max_elements, "max_evaluations": max_evaluations,
              "evaluations_used": 0, "outputs": [], "reference_identity": "captured_compiler_graph",
              "semantic_verdict": "not_established",
              "scope": "all mapped graph outputs, K versus captured compiler graph semantics; no independent f, G-f or binding certification"}
    if total > max_elements:
        report["note"] = "complete region output exceeds element budget"
        return report
    failures = {Invalid: "invalid_input", Unsupported: "unsupported", Undefined: "domain_undefined",
                NotEstablished: "not_established"}
    with L.evaluation_budget(max_evaluations) as budget:
        try:
            env = L.evaluate_lazy(task, {name: L.from_array(v["values"], v["dtype"])
                                        for name, v in inputs.items()})
            for observed in outputs:
                result = env[observed["node"]]
                if not isinstance(result, Exception) and result.dtype != A.dt(observed["dtype"]):
                    comparison = {"complete": False, "outcome": "not_established", "elements": 0,
                                  "note": "reference and captured output dtypes differ"}
                else:
                    comparison = D.full_reference(result, observed["values"], max_elements)
                report["outputs"].append({"position": observed["position"], "node": observed["node"],
                                          "comparison": comparison})
            outcomes = {o["comparison"]["outcome"] for o in report["outputs"]}
            for state in ("invalid_input", "unsupported", "not_established", "domain_undefined", "value"):
                if state in outcomes:
                    report["outcome"] = state
                    break
            report["complete"] = all(o["comparison"].get("complete") is True for o in report["outputs"])
        except tuple(failures) as error:
            report.update(outcome=failures[type(error)], note=str(error))
        finally:
            report["evaluations_used"] = budget.used
    return report
