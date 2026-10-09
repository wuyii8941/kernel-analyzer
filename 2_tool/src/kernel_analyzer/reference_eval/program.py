"""A small SSA program form that the reference evaluator executes.

TTIR is mapped onto this form in step 2.  Op names here are internal
reference operations, not MLIR names: a version change of Triton only changes
the mapping, not these definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Op:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, Any] = field(default_factory=dict)
    regions: tuple["Program", ...] = ()
    node_id: str = ""


@dataclass
class Program:
    """Straight-line op list; control flow lives in op regions.

    ``inputs`` are bound from the captured call.  A region's ``outputs`` are
    the values it yields to the enclosing op.
    """

    ops: list[Op]
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    def walk(self):
        for op in self.ops:
            yield op
            for region in op.regions:
                yield from region.walk()


class ProgramBuilder:
    """Convenience builder used by tests and by the TTIR mapper."""

    def __init__(self, inputs: tuple[str, ...] = ()):
        self.inputs = tuple(inputs)
        self.ops: list[Op] = []
        self._counter = 0

    def _fresh(self, hint: str) -> str:
        self._counter += 1
        return f"%{hint}{self._counter}"

    def op(self, name: str, *inputs: str, out: Optional[str] = None, n_out: int = 1,
           regions: tuple[Program, ...] = (), node_id: Optional[str] = None, **attrs) -> Any:
        if out is None:
            outputs = tuple(self._fresh(name) for _ in range(n_out))
        elif isinstance(out, str):
            outputs = (out,)
        else:
            outputs = tuple(out)
        node = Op(name, tuple(inputs), outputs, dict(attrs), tuple(regions),
                  node_id or f"{name}#{len(self.ops)}")
        self.ops.append(node)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def build(self, outputs: tuple[str, ...] = ()) -> Program:
        return Program(list(self.ops), self.inputs, tuple(outputs))
