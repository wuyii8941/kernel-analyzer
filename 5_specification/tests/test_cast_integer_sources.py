"""Integer and boolean sources for the CAST finite_rne draft (path A A2, 2026-10-20).

The draft pair contracts/draft/CAST_path_a_A2_20261020.{yaml} + CAST_adapter_path_a_A2_20261020.py is staged into
a temporary directory and loaded through kernel_spec.registry. Expected values use only Fraction and integer
arithmetic written here; they never call the adapter's rounding helpers or kernel_spec.ieee.
"""
import shutil
import sys
import tempfile
import unittest
from fractions import Fraction as Q
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernel_spec import registry
from kernel_spec.num import evaluate

DRAFT = ROOT / "contracts" / "draft"
PRECISION = {"f16": 11, "bf16": 8, "f32": 24, "f64": 53}
MAXIMUM = {"f16": 65504, "bf16": (2 ** 8 - 1) * 2 ** 120, "f32": (2 ** 24 - 1) * 2 ** 104,
           "f64": (2 ** 53 - 1) * 2 ** 971}
RANGE = {"i64": (-2 ** 63, 2 ** 63 - 1), "i32": (-2 ** 31, 2 ** 31 - 1), "i16": (-2 ** 15, 2 ** 15 - 1),
         "i8": (-128, 127), "u8": (0, 255)}


def rne_integer(n, target):
    """Round an integer to the nearest target value, ties to even significand (integers are never subnormal)."""
    if n == 0:
        return 0
    sign, m = (-1, -n) if n < 0 else (1, n)
    p = PRECISION[target]
    shift = max(m.bit_length() - p, 0)
    q, r = divmod(m, 1 << shift)
    half = (1 << shift) // 2 if shift else None
    if shift and (r > half or (r == half and q % 2 == 1)):
        q += 1
    value = q << shift
    if value > MAXIMUM[target]:
        raise OverflowError(target)
    return sign * value


class Staged:
    tmp = None

    @classmethod
    def stage(cls):
        if cls.tmp is None:
            cls.tmp = Path(tempfile.mkdtemp(prefix="cast_integer_"))
            (cls.tmp / "contracts").mkdir()
            (cls.tmp / "adapters").mkdir()
            shutil.copy(DRAFT / "CAST_path_a_A2_20261020.yaml", cls.tmp / "contracts" / "CAST.yaml")
            shutil.copy(DRAFT / "CAST_adapter_path_a_A2_20261020.py", cls.tmp / "adapters" / "CAST.py")
        return registry.load("CAST", contracts=cls.tmp / "contracts", adapters=cls.tmp / "adapters")


class CastIntegerSources(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = Staged.stage()
        cls.adapter = cls.contract.adapter

    def run_call(self, call, profile="finite_rne"):
        return evaluate(self.adapter.reference, profile, call)

    def convert(self, source, values, target, **extra):
        x = {"t": source, "v": values}
        x.update(extra)
        return self.run_call({"op": "prims.convert_element_type", "args": [x, target]})

    def values(self, result):
        self.assertEqual(result.outcome, "value", getattr(result, "detail", result))
        return [Q(v) for v in result.fields["out"]]

    def test_profile_keeps_task_choice(self):
        profile = self.contract.data["profiles"]["finite_rne"]
        self.assertIs(profile["task_choice_required"], True)
        self.assertEqual(list(self.contract.data["profiles"]), ["finite_rne"])
        self.assertEqual(self.contract.data["review"]["status"], "drafted")
        self.assertEqual(self.contract.data["activation"]["implementation_status"], "pending")

    def test_recorded_qwen_position_ids_structure(self):
        positions = [[[i] for i in range(128)]]
        call = {"op": "aten._to_copy", "args": [{"t": "i64", "v": positions, "shape": [1, 128, 1]}],
                "kwargs": {"dtype": {"dtype": "f32"}}}
        result = self.run_call(call)
        self.assertEqual(result.outcome, "value")
        self.assertEqual(result.fields["out"], positions)
        observed = {"out": {"t": "f32", "shape": [1, 128, 1], "v": positions}}
        self.assertTrue(self.adapter.check_output("finite_rne", call, observed))
        self.assertFalse(self.adapter.check_output(
            "finite_rne", call, {"out": {"t": "i64", "shape": [1, 128, 1], "v": positions}}))
        wrong = [[[i + (i == 5)] for i in range(128)]]
        self.assertFalse(self.adapter.check_output(
            "finite_rne", call, {"out": {"t": "f32", "shape": [1, 128, 1], "v": wrong}}))

    def test_precision_boundary_of_every_target(self):
        for target, p in PRECISION.items():
            base = 2 ** p
            points = [base - 1, base, base + 1, base + 2, base + 3, base + 5, 2 * base + 2, 2 * base + 6]
            points += [-x for x in points]
            if base + 8 > RANGE["i64"][1]:
                continue
            with self.subTest(target=target):
                self.assertEqual(self.values(self.convert("i64", points, target)),
                                 [rne_integer(n, target) for n in points])

    def test_range_limits_of_every_integer_dtype(self):
        for source, (lo, hi) in RANGE.items():
            points = [lo, lo + 1, -1 if lo < 0 else 0, 0, 1, hi - 1, hi]
            for target in PRECISION:
                expected = []
                try:
                    expected = [rne_integer(n, target) for n in points]
                except OverflowError:
                    self.assertEqual(self.convert(source, points, target).outcome, "unsupported")
                    continue
                with self.subTest(source=source, target=target):
                    self.assertEqual(self.values(self.convert(source, points, target)), expected)

    def test_f16_overflow_threshold(self):
        self.assertEqual(self.values(self.convert("i32", [65504, 65505, 65519, -65519], "f16")),
                         [65504, 65504, 65504, -65504])
        for n in (65520, -65520, 70000, 2 ** 31 - 1):
            with self.subTest(n=n):
                self.assertEqual(self.convert("i32", [n], "f16").outcome, "unsupported")

    def test_booleans(self):
        for target in PRECISION:
            with self.subTest(target=target):
                self.assertEqual(self.values(self.convert("b8", [True, False, True], target)), [1, 0, 1])

    def test_unsigned_bytes_are_never_negative(self):
        values = list(range(256))
        self.assertEqual(self.values(self.convert("u8", values, "f16")), values)
        self.assertEqual(self.values(self.convert("u8", values, "bf16")), [rne_integer(n, "bf16") for n in values])

    def test_malformed_integer_and_boolean_records(self):
        cases = (("u8", [256]), ("u8", [-1]), ("i8", [128]), ("i8", [-129]), ("i16", [2 ** 15]),
                 ("i32", [2 ** 31]), ("i64", [2 ** 63]), ("i64", ["1/2"]), ("i64", [0.5]), ("i64", [True]),
                 ("b8", [2]), ("b8", [1]), ("b8", ["1"]), ("i64", [float("inf")]))
        for source, values in cases:
            with self.subTest(source=source, values=values):
                self.assertEqual(self.convert(source, values, "f32").outcome, "invalid_input")

    def test_targets_other_than_the_four_floats_stay_unsupported(self):
        for source, value in (("i64", 1), ("b8", True), ("f32", "3/2")):
            for target in ("i64", "i32", "u8", "b8", "f8e4m3", "complex64"):
                with self.subTest(source=source, target=target):
                    self.assertEqual(self.convert(source, [value], target).outcome, "unsupported")

    def test_empty_and_multi_dimensional_shapes(self):
        result = self.convert("i16", [], "f64", shape=[2, 0])
        self.assertEqual(result.outcome, "value")
        self.assertEqual(result.fields["out"], [[], []])
        result = self.convert("i8", [[1, -2], [3, -4]], "bf16")
        self.assertEqual(result.fields["out"], [[1, -2], [3, -4]])

    def test_floating_scope_is_unchanged(self):
        self.assertEqual(self.values(self.convert("f32", ["257/256", "259/256"], "bf16")), [1, Q(65, 64)])
        self.assertEqual(self.convert("f16", ["1/3"], "f32").outcome, "invalid_input")
        self.assertEqual(self.convert("f32", ["inf"], "bf16").outcome, "unsupported")
        self.assertEqual(self.convert("f32", [float("nan")], "f16").outcome, "invalid_input")

    def test_to_copy_options_still_refused(self):
        x = {"t": "i64", "v": [1]}
        self.assertEqual(self.run_call({"op": "aten._to_copy", "args": [x]}).outcome, "unsupported")
        self.assertEqual(self.run_call({"op": "aten._to_copy", "args": [x],
                                        "kwargs": {"dtype": "f32", "device": "cuda"}}).outcome, "unsupported")


class CastIntegerMutants(unittest.TestCase):
    """Each integer misreading differs from the reference on its witness."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = Staged.stage().adapter

    def out(self, fn, source, values, target):
        r = evaluate(fn, "finite_rne", {"op": "prims.convert_element_type",
                                        "args": [{"t": source, "v": values}, target]})
        return r.outcome, r.fields.get("out") if r.outcome == "value" else None

    def test_mutants_are_rejected_on_their_witnesses(self):
        witnesses = {
            "CA.M5": ("i64", [16777219, 9223372036854775807], "f32"),
            "CA.M6": ("b8", [True, False], "bf16"),
            "CA.M7": ("u8", [128, 200, 255], "f16"),
            "CA.M8": ("i64", [1099511627777], "f64"),
        }
        for mid, (source, values, target) in witnesses.items():
            with self.subTest(mutant=mid):
                self.assertNotEqual(self.out(self.adapter.MUTANTS[mid], source, values, target),
                                    self.out(self.adapter.reference, source, values, target))
        self.assertEqual([Q(v) for v in self.out(self.adapter.MUTANTS["CA.M5"], "i64", [16777219], "f32")[1]],
                         [16777218])
        self.assertEqual(self.out(self.adapter.MUTANTS["CA.M6"], "b8", [True], "f32")[1], [-1])
        self.assertEqual(self.out(self.adapter.MUTANTS["CA.M7"], "u8", [200], "f32")[1], [-56])
        self.assertEqual(self.out(self.adapter.MUTANTS["CA.M8"], "i64", [1099511627777], "f64")[1],
                         [1099511627776])

    def test_mutants_agree_where_their_misreading_is_invisible(self):
        for mid in ("CA.M5", "CA.M6", "CA.M7", "CA.M8"):
            with self.subTest(mutant=mid):
                self.assertEqual(self.out(self.adapter.MUTANTS[mid], "i64", [0, 1, 127], "f32"),
                                 self.out(self.adapter.reference, "i64", [0, 1, 127], "f32"))


def tearDownModule():
    if Staged.tmp is not None:
        shutil.rmtree(Staged.tmp, ignore_errors=True)
        Staged.tmp = None


if __name__ == "__main__":
    unittest.main()
