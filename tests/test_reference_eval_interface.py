"""Interface / compile-time constant inventory (CPU)."""

import math
from types import SimpleNamespace

import numpy as np

from kernel_analyzer.reference_eval.interface import constant_record, scalar_interface


def test_rounded_constants_are_read_back_to_their_simple_values():
    third = float(np.float32(1.0 / 3.0))
    rec = constant_record(third, "f32")
    assert rec["rounded_from"]["candidate"] == "1/3"
    assert rec["rounded_from"]["relative_rounding"] == 2.0 ** -25  # 3 * RN32(1/3) - 1, exactly
    assert "rounded_from" not in constant_record(0.125, "f32") and constant_record(0.125, "f32")["power_of_two"]
    assert "rounded_from" not in constant_record(4096.0, "f32")  # integers are exact
    assert constant_record(float(np.float32(math.sqrt(2.0))), "f32")["rounded_from"]["candidate"] == "sqrt(2)"
    assert constant_record(float(np.float32(-1 / math.sqrt(2.0))), "f32")["rounded_from"]["candidate"] == "-1/sqrt(2)"
    # the same value stored in f64 is the double nearest 1/3: still a rounded 1/3
    assert constant_record(1.0 / 3.0, "f64")["rounded_from"]["candidate"] == "1/3"


def test_runtime_scalars_record_the_value_the_kernel_receives():
    args = [SimpleNamespace(kind="float", name="t", value=1 / math.sqrt(2), constexpr=False, signature_type="fp32"),
            SimpleNamespace(kind="float", name="s", value=0.25, constexpr=False, signature_type="fp32"),
            SimpleNamespace(kind="int", name="D", value=4096, constexpr=True, signature_type="constexpr"),
            SimpleNamespace(kind="tensor", name="x", value=None, constexpr=False, signature_type="*fp32")]
    rows = {r["name"]: r for r in scalar_interface(SimpleNamespace(args=args))}
    assert rows["t"]["exact"] is False and float(rows["t"]["received"]) == float(np.float32(1 / math.sqrt(2)))
    assert rows["s"]["exact"] is True and "relative_rounding" not in rows["s"]
    assert rows["D"]["compile_time"] and "x" not in rows


def test_arbitrary_values_are_rarely_read_as_rounded_simple_values():
    rng = np.random.default_rng(1)
    vals = rng.uniform(0.01, 1000, 3000).astype(np.float32).astype(float)
    hits = sum("rounded_from" in constant_record(v, "f32") for v in vals)
    assert hits <= 9  # design: chance below about 1e-3 per constant
