"""Evidence: run the F02 interpreter regression with the baseline (bb0a610) _scaled_dot plugged into the fixed tree
(parser support for one-sided scales kept), to show the test detects the dropped upper end itself."""
import ast, subprocess, sys, textwrap
import numpy as np
sys.path[:0] = ["2_tool/src", "2_tool/tests", "2_tool"]
from kernel_analyzer.reference_eval import ttir_eval as E
src = subprocess.run(["git", "show", "bb0a610:2_tool/src/kernel_analyzer/reference_eval/ttir_eval.py"],
                     capture_output=True, text=True, check=True).stdout
tree = ast.parse(src)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KernelReferenceEvaluator")
fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_scaled_dot")
ns = dict(vars(E))
exec(compile(ast.Module([fn], []), "baseline_scaled_dot", "exec"), ns)
E.KernelReferenceEvaluator._scaled_dot = ns["_scaled_dot"]
import pytest
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "2_tool/tests/test_audit_findings.py", "-k",
                      "computed_operand or upper_end"]))
