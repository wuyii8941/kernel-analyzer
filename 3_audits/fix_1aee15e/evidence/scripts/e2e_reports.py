"""Write the end-to-end measure.run reports of the audit regressions (classic, Gluon, float atomic, F04 upstream
cases) for the evidence package."""
import json, sys
from pathlib import Path
sys.path[:0] = ["2_tool/src"]
from kernel_analyzer import measure
HERE = Path("2_tool/tests").resolve()
X1D = {"sampler": {"uniform": [1, 2]}, "shape": [1024], "dtype": "float32"}
cases = {"classic": X1D, "gluon_affine": X1D,
         "atomic_row_sum": {"sampler": {"normal": [0, 1]}, "shape": [8, 512], "dtype": "float32"},
         "value_equal_upstream": X1D, "zero_upstream": X1D, "honest_copy": X1D}
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
summary = {}
for call, x in cases.items():
    d = {"call": f"audit_calls.py:{call}", "inputs": {"x": x}, "compare": {"mode": "A", "measure": ["y"]},
         "budget": {"cpu_seconds": 600, "gpu_seconds": 600, "case_timeout": 300, "max_units": 6},
         "units": {"development": 2, "confirmation": 4}, "_base_dir": str(HERE)}
    rep = measure.run(d, out=str(out / f"{call}.json"))
    lv = rep["levels"][0]
    y = lv.get("outputs", {}).get("y", {})
    summary[call] = {"status": lv["status"], "reason": lv.get("reason"), "ir_kinds": (lv.get("notes") or {}).get("ir_kinds"),
                     "complete_rate": (y.get("reference") or {}).get("complete_rate"),
                     "reference_scope": (y.get("reference") or {}).get("reference_scope"),
                     "mixed": y.get("mixed_non_triton_sources"),
                     "execution": (y.get("execution") or {}).get("statistics"),
                     "declaration_sha256": rep["expanded_declaration"]["declaration_sha256"][:16],
                     "seconds": rep["seconds"]}
print(json.dumps(summary, indent=1))
(out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
