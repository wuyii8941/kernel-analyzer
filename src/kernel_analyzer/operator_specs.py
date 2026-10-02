"""Small mathematical specifications reused across implementation backends.

Specifications are selected by the caller, not inferred from source or learned
from candidate outputs. The first reference is deliberately slow and limited:
it is for checking small sums, not timing production kernels.
"""

from dataclasses import dataclass, asdict
from fractions import Fraction
import math


@dataclass(frozen=True)
class SumSpec:
    """Exact sum of the stored finite input values along one declared axis.

    The reference returns FP64 only if that representation is exact. It raises
    otherwise; unsupported inputs never silently fall back to an inexact sum.
    No masks, weights, hidden padding or additional tensor operands are assumed.
    Output conversion error is included in the comparison with this real sum.
    """

    axis: int = -1
    keepdim: bool = False
    input_arg: int | str = 0
    max_elements: int = 16384

    def __post_init__(self):
        if type(self.axis) is not int or type(self.keepdim) is not bool:
            raise TypeError("axis must be an integer and keepdim a boolean")
        if not isinstance(self.input_arg, (int, str)) or isinstance(self.input_arg, bool):
            raise TypeError("input_arg must be a positional index or keyword name")
        if type(self.max_elements) is not int or self.max_elements < 1:
            raise ValueError("max_elements must be positive")

    def contract(self):
        return {
            **asdict(self),
            "name": "sum_of_stored_values_v1",
            "semantic_origin": "caller_declared_sum",
            "reference_source": "builtin_exact_rational_sum",
            "exactness": "rational_result_must_be_exactly_representable_in_float64",
            "includes_output_rounding": True,
            "backward_supported": False,
        }

    def reference(self, *args, **kwargs):
        import torch

        x = args[self.input_arg] if isinstance(self.input_arg, int) else kwargs[self.input_arg]
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("SumSpec requires one floating tensor at input_arg")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("unsupported input format for exact sum")
        if not x.ndim or not x.numel() or x.numel() > self.max_elements:
            raise ValueError("SumSpec requires a nonempty tensor within max_elements")
        if not -x.ndim <= self.axis < x.ndim:
            raise ValueError("sum axis is outside input rank")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("SumSpec only accepts finite inputs")
        axis = self.axis % x.ndim
        rows = x.detach().cpu().double().movedim(axis, -1).reshape(-1, x.shape[axis])
        totals = []
        for row in rows.tolist():
            exact = sum((Fraction.from_float(value) for value in row), Fraction(0))
            value = float(exact)
            if not math.isfinite(value) or Fraction.from_float(value) != exact:
                raise ValueError("exact sum is not representable in the FP64 reporting reference")
            totals.append(value)
        shape = tuple(size for index, size in enumerate(x.shape) if index != axis)
        result = torch.tensor(totals, dtype=torch.float64).reshape(shape)
        return result.unsqueeze(axis) if self.keepdim else result
