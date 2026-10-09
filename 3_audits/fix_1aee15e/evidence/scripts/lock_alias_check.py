import sys, pytest
sys.path[:0] = ["2_tool/src", "2_tool/tests", "2_tool"]
from kernel_analyzer.reference_eval import ttir_eval as E
if sys.argv[1] == "disable":
    E.KernelReferenceEvaluator._note_lock_write = lambda *a: None   # the writer record off
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "2_tool/tests/test_audit_findings.py", "-k", "aliased_lock"]))
