"""Independent review tests for the CAST path A A2 draft (package F: integer and boolean sources), 2026-10-20.

The draft pair contracts/draft/CAST_path_a_A2_20261020.yaml + CAST_adapter_path_a_A2_20261020.py is copied into a
fresh stage /data1/tzh/kernel-analyzer/.cache/spec_a2_review/F and loaded with kernel_spec.registry.load. Every
expected value comes from the reviewer's own IEEE 754 roundTiesToEven encoder below (Python integers and Fraction
only; neither the adapter's helpers nor kernel_spec.ieee are used for the computation). Floating-to-floating
behaviour is compared with the top-level validated adapter on random inputs.
"""
import hashlib
import random
import shutil
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernel_spec import registry  # noqa: E402
from kernel_spec.num import evaluate  # noqa: E402

DRAFT_YAML = ROOT / "contracts" / "draft" / "CAST_path_a_A2_20261020.yaml"
DRAFT_PY = ROOT / "contracts" / "draft" / "CAST_adapter_path_a_A2_20261020.py"
STAGE = ROOT.parent / ".cache" / "spec_a2_review" / "F"

# IEEE 754-2019 3.6 binary formats (bf16 is the 8-bit-precision truncation of binary32): precision p, emax.
# emin = 1 - emax for every format.
FORMATS = {"f16": (11, 15), "bf16": (8, 127), "f32": (24, 127), "f64": (53, 1023)}
TARGETS = tuple(FORMATS)
INT_LIMITS = {"i64": 64, "i32": 32, "i16": 16, "i8": 8}


def int_range(dtype):
    if dtype == "u8":
        return 0, 255
    bits = INT_LIMITS[dtype]
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


class Overflow(Exception):
    pass


def largest_finite(fmt):
    p, emax = FORMATS[fmt]
    return Q((1 << p) - 1) * Q(2) ** (emax - p + 1)


def floor_log2(a):
    """floor(log2 a) for a positive Fraction."""
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Q(2) ** e > a:
        e -= 1
    while Q(2) ** (e + 1) <= a:
        e += 1
    return e


def rne(x, fmt):
    """IEEE 754 4.3.1 roundTiesToEven of an exact rational x into fmt; 7.4 overflow (rounded magnitude above the
    largest finite value, unbounded exponent) raises Overflow."""
    x = Q(x)
    if x == 0:
        return Q(0)
    p, emax = FORMATS[fmt]
    emin = 1 - emax
    a = abs(x)
    e = max(floor_log2(a), emin)
    quantum = Q(2) ** (e - p + 1)            # spacing of fmt in the binade of a (subnormals share emin's)
    k = a / quantum
    lower = k.numerator // k.denominator
    frac = k - lower
    if frac > Q(1, 2) or (frac == Q(1, 2) and lower % 2 == 1):
        lower += 1                            # integral significand of the chosen neighbour
    y = lower * quantum
    if y > largest_finite(fmt):
        raise Overflow(fmt)
    return y if x > 0 else -y


def rne_int_by_neighbours(n, fmt):
    """A second, structurally different integer encoder: list the two representable neighbours explicitly by
    walking the binade grid, then pick the nearer one, ties to the one with an even integral significand."""
    if n == 0:
        return 0
    p, _ = FORMATS[fmt]
    m = abs(n)
    spacing = 1
    while m >= (1 << p) * spacing:            # binade [2^(p-1) s, 2^p s) has spacing s
        spacing *= 2
    lo = m - m % spacing
    hi = lo + spacing
    if m - lo < hi - m:
        y = lo
    elif m - lo > hi - m:
        y = hi
    else:
        y = lo if (lo // spacing) % 2 == 0 else hi
    if y > largest_finite(fmt):
        raise Overflow(fmt)
    return y if n > 0 else -y


def random_float_value(rng, fmt):
    """A random finite value of fmt (normal or subnormal, any sign), built from its integral significand."""
    p, emax = FORMATS[fmt]
    emin = 1 - emax
    if rng.random() < 0.2:
        sig = rng.randrange(1, 1 << (p - 1))                  # subnormal
        e = emin
    else:
        sig = rng.randrange(1 << (p - 1), 1 << p)
        e = rng.randrange(emin, emax + 1)
    v = sig * Q(2) ** (e - p + 1)
    return -v if rng.random() < 0.5 else v


def enc(q):
    q = Q(q)
    return int(q) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage():
    (STAGE / "contracts").mkdir(parents=True, exist_ok=True)
    (STAGE / "adapters").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DRAFT_YAML, STAGE / "contracts" / "CAST.yaml")
    shutil.copyfile(DRAFT_PY, STAGE / "adapters" / "CAST.py")
    return registry.load("CAST", contracts=STAGE / "contracts", adapters=STAGE / "adapters")


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = stage()
        cls.ref = staticmethod(cls.contract.adapter.reference)

    def call(self, call, fn=None):
        return evaluate(fn or self.ref, "finite_rne", call)

    def conv(self, source, values, target, fn=None, **extra):
        x = {"t": source, "v": values}
        x.update(extra)
        return self.call({"op": "prims.convert_element_type", "args": [x, target]}, fn)

    def flat(self, v):
        if isinstance(v, list):
            out = []
            for x in v:
                out += self.flat(x)
            return out
        return [Q(v)]

    def assert_values(self, source, values, target, expected):
        r = self.conv(source, values, target)
        self.assertEqual(r.outcome, "value", (source, target, getattr(r, "detail", None)))
        got = self.flat(r.fields["out"])
        self.assertEqual(len(got), len(expected))
        for i, (g, e) in enumerate(zip(got, expected)):
            if g != e:
                self.fail(f"{source}->{target} coordinate {i}: input {values[i] if i < len(values) else '?'} "
                          f"got {g} expected {e}")

    def outcome(self, source, values, target, **extra):
        return self.conv(source, values, target, **extra).outcome


class Staging(Base):
    def test_stage_is_the_draft_pair(self):
        self.assertEqual(sha(STAGE / "contracts" / "CAST.yaml"), sha(DRAFT_YAML))
        self.assertEqual(sha(STAGE / "adapters" / "CAST.py"), sha(DRAFT_PY))
        self.assertTrue(Path(self.contract.adapter.__file__).resolve().is_relative_to(STAGE.resolve()))
        self.assertEqual(list(self.contract.data["profiles"]), ["finite_rne"])
        self.assertIs(self.contract.data["profiles"]["finite_rne"]["task_choice_required"], True)

    def test_encoders_agree_with_each_other_and_with_cpython_for_f64(self):
        rng = random.Random(7)
        pts = [rng.randrange(-(1 << 63), 1 << 63) for _ in range(3000)]
        pts += [rng.randrange(-(1 << 30), 1 << 30) for _ in range(1000)]
        for n in pts:
            for fmt in TARGETS:
                try:
                    a = rne(n, fmt)
                except Overflow:
                    with self.assertRaises(Overflow):
                        rne_int_by_neighbours(n, fmt)
                    continue
                self.assertEqual(a, rne_int_by_neighbours(n, fmt))
            # CPython int -> float is correctly rounded with ties to even
            self.assertEqual(rne(n, "f64"), Q(float(n)))


class IntegerSources(Base):
    def test_exhaustive_u8_i8_to_every_target(self):
        for source in ("u8", "i8"):
            lo, hi = int_range(source)
            values = list(range(lo, hi + 1))
            for target in TARGETS:
                with self.subTest(source=source, target=target):
                    self.assert_values(source, values, target, [rne(n, target) for n in values])

    def test_exhaustive_i16_to_f16_and_bf16(self):
        values = list(range(-(1 << 15), 1 << 15))
        for target in ("f16", "bf16"):
            with self.subTest(target=target):
                self.assert_values("i16", values, target, [rne(n, target) for n in values])

    def test_ties_and_neighbours_at_every_precision_boundary(self):
        for target, (p, _) in FORMATS.items():
            values = []
            for e in range(p - 2, p + 4):
                base = 1 << e
                values += [base + k for k in range(-9, 10)]
                spacing = 1 << max(e - p + 1, 0)
                values += [base + spacing * j + spacing // 2 for j in range(6)]       # exact midpoints
            values = sorted({v for v in values if v > 0})
            values += [-v for v in values]
            for source in ("i64", "i32"):
                lo, hi = int_range(source)
                pts = [v for v in values if lo <= v <= hi]
                try:
                    expected = [rne(n, target) for n in pts]
                except Overflow:
                    pts = [v for v in pts if abs(v) < 65520]
                    expected = [rne(n, target) for n in pts]
                with self.subTest(target=target, source=source):
                    self.assert_values(source, pts, target, expected)

    def test_named_points(self):
        cases = {
            ("i64", "f32"): {2 ** 24 + 1: 2 ** 24, 2 ** 24 + 3: 2 ** 24 + 4, -(2 ** 24 + 3): -(2 ** 24 + 4),
                             2 ** 63 - 1: 2 ** 63, 2 ** 63 - 2 ** 38: 2 ** 63, 2 ** 63 - 2 ** 38 - 1: 2 ** 63 - 2 ** 39,
                             -2 ** 63: -2 ** 63, -2 ** 63 + 1: -2 ** 63},
            ("i64", "f64"): {2 ** 53 + 1: 2 ** 53, 2 ** 53 + 3: 2 ** 53 + 4, 2 ** 63 - 1: 2 ** 63,
                             2 ** 63 - 2 ** 9: 2 ** 63, 2 ** 63 - 3 * 2 ** 9: 2 ** 63 - 2 ** 11,
                             2 ** 63 - 2 ** 9 - 1: 2 ** 63 - 2 ** 10, 2 ** 40 + 1: 2 ** 40 + 1},
            ("i64", "bf16"): {2 ** 63 - 2 ** 54: 2 ** 63, 2 ** 63 - 2 ** 54 - 1: 2 ** 63 - 2 ** 55, 257: 256,
                              259: 260},
            ("i32", "f16"): {65504: 65504, 65505: 65504, 65519: 65504, -65519: -65504, 65488: 65472,
                             65489: 65504, 2049: 2048, 2051: 2052, 4097: 4096, 4098: 4096, 4099: 4100},
            ("i16", "bf16"): {32767: 32768, -32768: -32768, 257: 256, 32640: 32640, 32704: 32768},
        }
        for (source, target), table in cases.items():
            pts = list(table)
            with self.subTest(source=source, target=target):
                self.assertEqual([rne(n, target) for n in pts], list(table.values()))   # reviewer hand check
                self.assert_values(source, pts, target, [Q(v) for v in table.values()])

    def test_random_wide_integers(self):
        rng = random.Random(20261020)
        for source in ("i64", "i32"):
            lo, hi = int_range(source)
            pts = [rng.randint(lo, hi) for _ in range(1500)]
            pts += [rng.choice([-1, 1]) * rng.randrange(1 << 20, 1 << 26) for _ in range(500)]
            pts = [v for v in pts if lo <= v <= hi]
            for target in ("bf16", "f32", "f64"):
                with self.subTest(source=source, target=target):
                    self.assert_values(source, pts, target, [rne(n, target) for n in pts])

    def test_range_limits_of_every_integer_dtype(self):
        for source in ("i64", "i32", "i16", "i8", "u8"):
            lo, hi = int_range(source)
            pts = sorted({lo, lo + 1, hi - 1, hi, 0, 1})
            for target in TARGETS:
                with self.subTest(source=source, target=target):
                    try:
                        expected = [rne(n, target) for n in pts]
                    except Overflow:
                        self.assertEqual(self.outcome(source, pts, target), "unsupported")
                        continue
                    self.assert_values(source, pts, target, expected)

    def test_single_direct_rounding_not_through_f32(self):
        # exact in f64 but not in f32: a double rounding through f32 would change them
        pts = [2 ** 40 + 1, 2 ** 30 + 1, -(2 ** 52 + 1), 2 ** 62 + 2 ** 10]
        self.assert_values("i64", pts, "f64", [Q(n) for n in pts])
        # bf16 via f32 double rounding witness: 2^31 + 2^23 + 1 -> f32 gives 2^31 + 2^23 (tie to even? no:
        # f32 spacing at 2^31 is 2^8, so f32 keeps 2^31+2^23 exactly after dropping +1), bf16 spacing 2^24:
        # direct: 2^31+2^23+1 is above the bf16 midpoint 2^31+2^23 -> 2^31+2^24; via f32: tie -> 2^31 (even).
        n = 2 ** 31 + 2 ** 23 + 1
        self.assertEqual(rne(n, "bf16"), 2 ** 31 + 2 ** 24)
        self.assertEqual(rne(rne(n, "f32"), "bf16"), 2 ** 31)
        self.assert_values("i64", [n], "bf16", [Q(2 ** 31 + 2 ** 24)])

    def test_unsigned_bytes_are_unsigned(self):
        self.assert_values("u8", [128, 200, 255], "f32", [Q(128), Q(200), Q(255)])

    def test_f16_overflow_boundary(self):
        self.assert_values("i32", list(range(65000, 65520)), "f16", [rne(n, "f16") for n in range(65000, 65520)])
        self.assert_values("i32", [-n for n in range(65500, 65520)], "f16",
                           [rne(-n, "f16") for n in range(65500, 65520)])
        for n in (65520, 65521, 65535, 65536, 70000, -65520, 2 ** 31 - 1, -2 ** 31):
            with self.subTest(n=n):
                with self.assertRaises(Overflow):
                    rne(n, "f16")
                self.assertEqual(self.outcome("i32", [n], "f16"), "unsupported")
                self.assertEqual(self.outcome("i64", [1, n], "f16"), "unsupported")

    def test_booleans(self):
        for target in TARGETS:
            with self.subTest(target=target):
                self.assert_values("b8", [[True, False], [False, True]], target, [Q(1), Q(0), Q(0), Q(1)])

    def test_recorded_position_ids_call(self):
        ids = [[[i] for i in range(128)]]
        for dtype in ({"dtype": "f32"}, "f32", "torch.float32"):
            with self.subTest(dtype=dtype):
                r = self.call({"op": "aten._to_copy", "args": [{"t": "i64", "v": ids, "shape": [1, 128, 1]}],
                               "kwargs": {"dtype": dtype}})
                self.assertEqual(r.outcome, "value")
                self.assertEqual(r.fields["out"], [[[Q(i)] for i in range(128)]])

    def test_empty_shapes(self):
        for source in ("i64", "u8", "b8"):
            r = self.conv(source, [], "f16", shape=[0, 3])
            self.assertEqual((r.outcome, r.fields.get("out")), ("value", []))


class Refusals(Base):
    def test_invalid_records(self):
        cases = []
        for source in ("i64", "i32", "i16", "i8", "u8"):
            lo, hi = int_range(source)
            cases += [(source, [lo - 1]), (source, [hi + 1]), (source, [0, hi + 1])]
            cases += [(source, ["1/2"]), (source, [0.5]), (source, [True]), (source, [None]),
                      (source, [float("nan")]), (source, ["inf"]), (source, [float("-inf")])]
        cases += [("b8", [0]), ("b8", [1]), ("b8", ["true"]), ("b8", [None]), ("b8", [1.0])]
        for source, values in cases:
            for target in ("f16", "f64"):
                with self.subTest(source=source, values=values, target=target):
                    self.assertEqual(self.outcome(source, values, target), "invalid_input")

    def test_integral_spellings_are_accepted(self):
        self.assert_values("i64", [3.0, "7", "-12/3"], "f32", [Q(3), Q(7), Q(-4)])

    def test_unsupported_targets_sources_and_options(self):
        for source, v in (("i64", 1), ("u8", 1), ("b8", True), ("f32", 1)):
            for target in ("i64", "i32", "i16", "i8", "u8", "b8", "f8e4m3", "f8e5m2", "complex64"):
                with self.subTest(source=source, target=target):
                    self.assertEqual(self.outcome(source, [v], target), "unsupported")
        for source in ("f8e4m3", "f8e5m2", "complex64", "u16", "u32"):
            with self.subTest(source=source):
                self.assertEqual(self.outcome(source, [1], "f32"), "unsupported")
        x = {"t": "i64", "v": [1]}
        self.assertEqual(self.call({"op": "aten._to_copy", "args": [x]}).outcome, "unsupported")
        for kw in ({"device": "cuda"}, {"layout": "strided"}, {"pin_memory": True}, {"non_blocking": True},
                   {"memory_format": "contiguous_format"}):
            with self.subTest(kw=kw):
                self.assertEqual(self.call({"op": "aten._to_copy", "args": [x],
                                            "kwargs": {"dtype": "f32", **kw}}).outcome, "unsupported")


class FloatingUnchanged(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.draft = staticmethod(stage().adapter.reference)
        cls.validated = staticmethod(registry.load("CAST").adapter.reference)

    def both(self, source, values, target):
        call = {"op": "prims.convert_element_type", "args": [{"t": source, "v": values}, target]}
        return evaluate(self.draft, "finite_rne", call), evaluate(self.validated, "finite_rne", call)

    def test_random_float_values_match_validated_and_encoder(self):
        rng = random.Random(5)
        for source in TARGETS:
            for target in TARGETS:
                vals = [random_float_value(rng, source) for _ in range(150)]
                keep, expected = [], []
                for v in vals:
                    try:
                        expected.append(rne(v, target))
                        keep.append(v)
                    except Overflow:
                        pass
                d, v = self.both(source, [enc(x) for x in keep], target)
                with self.subTest(source=source, target=target):
                    self.assertEqual((d.outcome, v.outcome), ("value", "value"))
                    self.assertEqual(d.fields["out"], v.fields["out"])
                    self.assertEqual([Q(x) for x in d.fields["out"]], expected)

    def test_random_float_refusals_match_validated(self):
        rng = random.Random(11)
        for source in TARGETS:
            for target in TARGETS:
                for _ in range(20):
                    kind = rng.choice(["inf", "nan", "unrep", "overflow", "mixed"])
                    v = random_float_value(rng, source)
                    if kind == "inf":
                        values = [enc(v), rng.choice(["inf", "-inf"])]
                    elif kind == "nan":
                        values = [enc(v), float("nan")]
                    elif kind == "unrep":
                        p, _ = FORMATS[source]
                        values = [enc(v), enc(1 + Q(1, 2 ** (p + 1)))]
                    elif kind == "overflow":
                        values = [enc(largest_finite(source))]
                    else:
                        values = [enc(v), enc(-v)]
                    d, w = self.both(source, values, target)
                    with self.subTest(source=source, target=target, values=values):
                        self.assertEqual(d.outcome, w.outcome)
                        if d.outcome == "value":
                            self.assertEqual(d.fields["out"], w.fields["out"])


if __name__ == "__main__":
    unittest.main()
