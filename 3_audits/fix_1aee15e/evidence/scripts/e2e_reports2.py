"""End-to-end measure.run reports for the follow-up batch: producer records (honest copy, ATen zeros accumulator,
value-equal arithmetic, underflow, compiled kernel after an ATen copy) and adaptive precision (met / not attainable)."""
import json, sys
from pathlib import Path
sys.path[:0] = ["2_tool/src"]
from kernel_analyzer import measure
HERE = Path("2_tool/tests").resolve()
X1D = {"sampler": {"uniform": [1, 2]}, "shape": [1024], "dtype": "float32"}
X2D = {"sampler": {"normal": [0, 1]}, "shape": [8, 512], "dtype": "float32"}
CANCEL = {"state": "audit_calls.py:cancelling_rows", "shape": [4, 256], "dtype": "float32"}
cases = {"honest_copy": (X1D, {}), "atomic_row_sum": (X2D, {}), "value_equal_upstream": (X1D, {}),
         "zero_upstream": (X1D, {}), "compiled_after_copy": (X1D, {}),
         "cumsum_rows": (CANCEL, {"resolution": {"ulp_fraction": 0.125}}),
         "cumsum_rows__unreachable": (CANCEL, {"resolution": {"ulp_fraction": 1e-12, "max_level": 2}})}
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
summary = {}
for name, (x, extra) in cases.items():
    call = name.split("__")[0]
    d = {"call": f"audit_calls.py:{call}", "inputs": {"x": x}, "compare": {"mode": "A", "measure": ["y"]},
         "budget": {"cpu_seconds": 600, "gpu_seconds": 600, "case_timeout": 300, "max_units": 6},
         "units": {"development": 2, "confirmation": 4}, "_base_dir": str(HERE), **extra}
    rep = measure.run(d, out=str(out / f"{name}.json"))
    lv = rep["levels"][0]
    y = lv.get("outputs", {}).get("y", {})
    summary[name] = {"status": lv["status"], "reference_scope": (y.get("reference") or {}).get("reference_scope"),
                     "mixed": y.get("mixed_non_triton_sources"),
                     "producer_records": y.get("upstream_with_producer_record"),
                     "resolution_met": (y.get("reference") or {}).get("resolution_met"),
                     "refinement": lv.get("refinement", {}).get("outcome"),
                     "levels": lv.get("refinement", {}).get("levels")}
(out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(summary, indent=1))
