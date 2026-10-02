"""Composed reference evaluator (steps 1-2 of the automatic K_R plan).

See ``docs/stage_summary_20261002.md`` sections 2, 4, 5 and 11.

Names are loaded lazily so that ``reference_eval.capture`` can run in an
environment without the exact-arithmetic dependencies (gmpy2).
"""

from importlib import import_module

_EXPORTS = {
    "MAYBE": "evaluator",
    "UNDEFINED": "evaluator",
    "ComposedResult": "evaluator",
    "LocalNode": "evaluator",
    "Mode": "evaluator",
    "NotObservable": "evaluator",
    "OutputClass": "evaluator",
    "Ref": "evaluator",
    "ReferenceEvaluator": "evaluator",
    "Special": "evaluator",
    "UnsupportedOperation": "evaluator",
    "adjoint_residual": "evaluator",
    "classify": "evaluator",
    "coverage_report": "evaluator",
    "DomainError": "numbers",
    "Interval": "numbers",
    "round_interval": "numbers",
    "round_nearest_even": "numbers",
    "Op": "program",
    "Program": "program",
    "ProgramBuilder": "program",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
