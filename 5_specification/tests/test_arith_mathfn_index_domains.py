"""Path A package D (2026-10-20): domain extensions of the ARITH, MATHFN and INDEX draft candidates.

The drafts live in contracts/draft/<ID>_path_a_A2_20261020.yaml and contracts/draft/<ID>_adapter_path_a_A2_20261020.py.
Each test class stages them into a temporary <stage>/contracts, <stage>/adapters layout and loads them through
kernel_spec.registry.load. Expected values are computed here independently: Fraction arithmetic for exact results,
python-flint Arb at 1000 bits for transcendental ones, and NumPy coordinate construction for lookups.
"""
from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import flint
import numpy as np

from kernel_spec import registry
from kernel_spec.num import Invalid, Undefined, Unsupported, ends, evaluate

SPEC = Path(__file__).resolve().parents[1]
DRAFT = SPEC / "contracts" / "draft"
STAMP = "path_a_A2_20261020"


def stage(*cids):
    root = Path(tempfile.mkdtemp(prefix="arith_mathfn_index_"))
    (root / "contracts").mkdir()
    (root / "adapters").mkdir()
    for cid in cids:
        shutil.copy(DRAFT / f"{cid}_{STAMP}.yaml", root / "contracts" / f"{cid}.yaml")
        shutil.copy(DRAFT / f"{cid}_adapter_{STAMP}.py", root / "adapters" / f"{cid}.py")
    return root, {cid: registry.load(cid, contracts=root / "contracts", adapters=root / "adapters").adapter
                  for cid in cids}


def flatten(x):
    if isinstance(x, list):
        for y in x:
            yield from flatten(y)
    else:
        yield x


def f32(values):
    """Exact rationals of float32-representable test values (NumPy does the storage, Fraction reads it)."""
    return [Fraction(float(v)) for v in np.asarray(values, dtype=np.float32).reshape(-1)]


class Staged(unittest.TestCase):
    CIDS = ()

    @classmethod
    def setUpClass(cls):
        cls.root, cls.adapters = stage(*cls.CIDS)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def call(self, cid, op, args, kwargs=None):
        inp = {"op": op, "args": args}
        if kwargs:
            inp["kwargs"] = kwargs
        return self.adapters[cid].reference("real", inp)


class ArithScalarsPowInfinitiesIntegers(Staged):
    CIDS = ("ARITH",)

    def test_adapter_reads_its_own_staged_contract(self):
        self.assertIn("aten.pow", self.adapters["ARITH"].COVERS)

    def test_recorded_python_floats_are_their_exact_binary64_values(self):
        rng = np.random.default_rng(20261020)
        values = rng.standard_normal(16).astype(np.float32) * 3
        for op, scalar in (("aten.mul", 0.044715), ("aten.mul", 0.7978845608028654),
                           ("aten.mul", 0.3535533905932738), ("aten.add", 1e-06)):
            with self.subTest(op=op, scalar=scalar):
                out = self.call("ARITH", op, [{"t": "f32", "v": values.tolist()}, scalar]).fields["out"]
                q = Fraction(scalar)          # the exact value of the binary64 number received
                expected = [x * q if op == "aten.mul" else x + q for x in f32(values)]
                self.assertEqual(out, expected)
        # bf16 tensor: the scalar is still exact, not 183/4096
        out = self.call("ARITH", "aten.mul", [{"t": "bf16", "v": [1]}, 0.044715]).fields["out"]
        self.assertEqual(out, [Fraction(6444110634811895, 2 ** 57)])
        self.assertNotEqual(out, [Fraction(183, 4096)])

    def test_scalars_no_aten_scalar_can_carry_are_invalid(self):
        with self.assertRaises(Invalid):
            self.call("ARITH", "aten.add", [{"t": "f32", "v": [1]}, "1/3"])
        # legal Python scalars outside the profile are unsupported, not invalid (review A2)
        for scalar in (True, 2 ** 64, -2 ** 63 - 1):
            with self.subTest(scalar=scalar), self.assertRaises(Unsupported):
                self.call("ARITH", "aten.add", [{"t": "f64", "v": [1]}, scalar])
        self.assertEqual(self.call("ARITH", "aten.add", [{"t": "f64", "v": [0]}, 2 ** 64 - 1]).fields["out"],
                         [2 ** 64 - 1])

    def test_floor_division_with_an_infinite_operand_is_unsupported(self):
        for op, args, kwargs in (("aten.div", [{"t": "f32", "v": ["inf"]}, 2], {"rounding_mode": "floor"}),
                                 ("aten.div", [{"t": "f32", "v": [-1]}, "inf"], {"rounding_mode": "floor"}),
                                 ("aten.floor_divide", [{"t": "f32", "v": [-1]}, {"t": "f32", "v": ["inf"]}], None)):
            with self.subTest(op=op, args=args), self.assertRaises(Unsupported):
                self.call("ARITH", op, args, kwargs)
        out = self.call("ARITH", "aten.div", [{"t": "f32", "v": ["inf", 7]}, 2], {"rounding_mode": "trunc"})
        self.assertEqual(out.fields["out"], ["+inf", 3])
        out = self.call("ARITH", "aten.div", [{"t": "f32", "v": [7, -7]}, 2], {"rounding_mode": "floor"})
        self.assertEqual(out.fields["out"], [3, -4])
        with self.assertRaises(Invalid):
            self.call("ARITH", "aten.add", [{"t": "f32", "v": [1]}, float("nan")])

    def test_readings_round_the_scalar_only_as_diagnostics(self):
        readings = self.adapters["ARITH"].READINGS
        inp = {"op": "aten.mul", "args": [{"t": "bf16", "v": [1]}, 0.044715]}
        self.assertEqual(readings["scalar_rounded_to_dtype"]("real", inp).fields["out"], [Fraction(183, 4096)])
        self.assertEqual(readings["scalar_rounded_to_opmath"]("real", inp).fields["out"],
                         [Fraction(12003091, 2 ** 28)])

    def test_integral_pow_is_exact_for_every_recorded_exponent(self):
        values = np.asarray([-3, -2.5, -0.5, 0, 0.75, 2, 1e-3], dtype=np.float32).tolist()
        for exponent in (3.0, 2.0, 2, 3, 1.0, 0, -2):
            with self.subTest(exponent=exponent):
                res = evaluate(self.adapters["ARITH"].reference, "real",
                               {"op": "aten.pow", "args": [{"t": "f32", "v": values}, exponent]})
                n = int(exponent)
                if n < 0:
                    self.assertEqual(res.outcome, "domain_undefined")   # the zero base
                    continue
                self.assertEqual(res.fields["out"], [x ** n if n else Fraction(1) for x in f32(values)])

    def test_pow_refusals(self):
        for exponent in (2.5, float("inf"), 65, {"t": "f32", "v": [2]}):
            with self.subTest(exponent=exponent), self.assertRaises(Unsupported):
                self.call("ARITH", "aten.pow", [{"t": "f32", "v": [4]}, exponent])
        with self.assertRaises(Invalid):          # 1/3 cannot reach an ATen Scalar
            self.call("ARITH", "aten.pow", [{"t": "f32", "v": [4]}, "1/3"])
        with self.assertRaises(Undefined):
            self.call("ARITH", "aten.pow", [{"t": "f32", "v": [0]}, -1])
        with self.assertRaises(Unsupported):
            self.call("ARITH", "aten.pow", [{"t": "i64", "v": [2]}, 2])

    def test_additive_mask_with_negative_infinity_broadcasts(self):
        a = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4) / 8
        mask = np.triu(np.full((4, 4), -np.inf, dtype=np.float32), 1)
        out = self.call("ARITH", "aten.add", [{"t": "f32", "v": a.tolist()},
                                              {"t": "f32", "v": mask.tolist()}]).fields["out"]
        self.assertEqual(np.asarray(out, dtype=object).shape, (2, 3, 4, 4))
        for index in np.ndindex(2, 3, 4, 4):
            got = np.asarray(out, dtype=object)[index]
            if index[3] > index[2]:
                self.assertEqual(got, "-inf")
            else:
                self.assertEqual(got, Fraction(float(a[index])))

    def test_where_passes_scalar_tensor_infinities_through(self):
        cond = [[True, False, True], [False, False, True]]
        out = self.call("ARITH", "aten.where", [{"t": "b8", "v": cond},
                                                {"t": "f32", "v": float("-inf"), "shape": []},
                                                {"t": "f32", "v": 0.0, "shape": []}]).fields["out"]
        self.assertEqual(out, [["-inf" if c else 0 for c in row] for row in cond])

    def test_undefined_extended_real_points(self):
        cases = (("aten.add", [{"t": "f32", "v": [float("inf")]}, {"t": "f32", "v": [float("-inf")]}], None),
                 ("aten.sub", [{"t": "f32", "v": ["inf"]}, {"t": "f32", "v": ["inf"]}], None),
                 ("aten.mul", [{"t": "f32", "v": ["inf"]}, 0], None),
                 ("aten.add", [{"t": "f32", "v": [1]}, {"t": "f32", "v": ["inf"]}], {"alpha": 0}))
        for op, args, kwargs in cases:
            with self.subTest(op=op), self.assertRaises(Undefined):
                self.call("ARITH", op, args, kwargs)

    def test_integer_arithmetic_is_exact_and_overflow_is_undefined(self):
        out = self.call("ARITH", "aten.add", [{"t": "i64", "v": list(range(128))}, 0]).fields["out"]
        self.assertEqual(out, list(range(128)))
        big = 2 ** 62 + 1                      # not a float32 or float64 value
        self.assertEqual(self.call("ARITH", "aten.add", [{"t": "i64", "v": [big]}, 3]).fields["out"], [big + 3])
        for op, dtype, value, scalar in (("aten.add", "i64", 2 ** 63 - 1, 1), ("aten.mul", "i32", 2 ** 30, 2),
                                         ("aten.sub", "u8", 0, 1)):
            with self.subTest(op=op, dtype=dtype), self.assertRaises(Undefined):
                self.call("ARITH", op, [{"t": dtype, "v": [value]}, scalar])
        with self.assertRaises(Undefined):          # exact 1 + 1000 leaves int8 (eager wraps to -23)
            self.call("ARITH", "aten.add", [{"t": "i8", "v": [1]}, 1000])
        self.assertEqual(self.call("ARITH", "aten.clamp", [{"t": "i8", "v": [-100, 5]}],
                                   {"min": -1000, "max": 3}).fields["out"], [-100, 3])
        for args in ([{"t": "i64", "v": [1]}, 0.5], [{"t": "i64", "v": [7]}, {"t": "i64", "v": [2]}]):
            with self.subTest(args=args), self.assertRaises(Unsupported):
                self.call("ARITH", "aten.add" if args[1] == 0.5 else "aten.div", args)

    def test_autograd_profiles_keep_the_finite_scope(self):
        adapter = self.adapters["ARITH"]
        with self.assertRaises(Invalid):
            adapter.reference("torch210_clamp_scalar_autograd", {
                "op": "torch.clamp", "input": {"t": "f32", "shape": [1], "v": [1]}, "min": 0.1, "max": None,
                "upstream": {"t": "f32", "shape": [1], "v": [1]}})
        with self.assertRaises(Unsupported):
            adapter.reference("torch210_where_tensor_autograd", {
                "op": "torch.where", "condition": {"t": "b8", "shape": [1], "v": [True]},
                "input": {"t": "f32", "shape": [1], "v": ["-inf"]}, "other": {"t": "f32", "shape": [1], "v": [0]},
                "upstream": {"t": "f32", "shape": [1], "v": [1]}})


def arb_value(op, q):
    a = flint.arb(flint.fmpq(q.numerator, q.denominator))
    return {"aten.sin": a.sin, "aten.cos": a.cos, "aten.tanh": a.tanh, "aten.exp": a.exp,
            "aten.expm1": a.expm1, "aten.log1p": a.log1p, "aten.log": a.log,
            "aten.sigmoid": lambda: 1 / (1 + (-a).exp())}[op]()


def exact(arf):
    m, e = arf.man_exp()
    return Fraction(int(m)) * Fraction(2) ** int(e)


class MathfnMagnitudeAndEvaluation(Staged):
    CIDS = ("MATHFN",)

    def assert_encloses(self, op, dtype, q):
        out = self.call("MATHFN", op, [{"t": dtype, "v": [f"{q.numerator}/{q.denominator}"]}]).fields["out"][0]
        lo, hi = ends(out)
        saved = flint.ctx.prec
        flint.ctx.prec = 1000
        try:
            ball = arb_value(op, q)
            ilo, ihi = exact(ball.lower()), exact(ball.upper())
        finally:
            flint.ctx.prec = saved
        if op in ("aten.sin", "aten.cos", "aten.tanh"):     # the true value lies in [-1, 1]
            ilo, ihi = max(ilo, Fraction(-1)), min(ihi, Fraction(1))
        self.assertLessEqual(lo, ilo)
        self.assertGreaterEqual(hi, ihi)
        self.assertLess(hi - lo, Fraction(1, 2 ** 200) * max(abs(lo), Fraction(1, 2 ** 1100)) + Fraction(1, 2 ** 240))

    def test_sin_cos_tanh_have_no_magnitude_bound(self):
        f32max = Fraction(float(np.finfo(np.float32).max))
        f64max = Fraction(float(np.finfo(np.float64).max))
        points = [(Fraction(127), "f32"), (Fraction(255, 2), "f32"), (Fraction(-128), "f32"),
                  (Fraction(101), "f32"), (f32max, "f32"), (-f32max, "f32"), (f64max, "f64"),
                  (Fraction(65504), "f16"), (Fraction(2) ** -149, "f32"), (Fraction(2) ** -1074, "f64"),
                  (Fraction(-3, 2 ** 100), "f64"), (Fraction(250), "f32")]
        for op in ("aten.sin", "aten.cos", "aten.tanh"):
            for q, dtype in points:
                with self.subTest(op=op, q=float(q)):
                    self.assert_encloses(op, dtype, q)

    def test_small_arguments_where_mpmath_interval_exp_fails(self):
        # mpmath 1.3.0 iv.exp returns a point excluding exp(x) for |x| in about [2^-271, 2^-55]
        for k in (55, 60, 100, 149, 200, 256):
            for sign in (1, -1):
                q = sign * Fraction(1, 2 ** k)
                for op in ("aten.exp", "aten.expm1", "aten.tanh", "aten.sigmoid", "aten.log1p", "aten.sin"):
                    with self.subTest(op=op, k=k, sign=sign):
                        self.assert_encloses(op, "f64", q)

    def test_bounds_kept_elsewhere_and_specials_refused(self):
        for op in ("aten.exp", "aten.expm1", "aten.sigmoid", "aten.log", "aten.sqrt", "aten.erf"):
            with self.subTest(op=op), self.assertRaises(Unsupported):
                self.call("MATHFN", op, [{"t": "f32", "v": [101]}])
        for op in ("aten.sin", "aten.cos", "aten.tanh"):
            with self.subTest(op=op), self.assertRaises(Unsupported):
                self.call("MATHFN", op, [{"t": "f32", "v": ["inf"]}])
            with self.subTest(op=op), self.assertRaises(Invalid):
                self.call("MATHFN", op, [{"t": "f32", "v": [float("nan")]}])

    def test_exact_points_stay_exact(self):
        self.assertEqual(self.call("MATHFN", "aten.cos", [{"t": "f32", "v": [0]}]).fields["out"], [1])
        self.assertEqual(self.call("MATHFN", "aten.tanh", [{"t": "f32", "v": [0]}]).fields["out"], [0])
        self.assertEqual(self.call("MATHFN", "aten.sigmoid", [{"t": "f32", "v": [0]}]).fields["out"], [Fraction(1, 2)])


class IndexIntegerDataAndPadding(Staged):
    CIDS = ("INDEX",)

    def test_integer_gather_and_index_select_are_exact_copies(self):
        rng = np.random.default_rng(7)
        buffer = (np.arange(512, dtype=np.int64) + 2 ** 53 + 1).reshape(1, 512)
        idx = rng.integers(0, 512, size=(1, 128))
        out = self.call("INDEX", "aten.gather", [{"t": "i64", "v": buffer.tolist()}, 1,
                                                  {"t": "i64", "v": idx.tolist()}]).fields["out"]
        self.assertEqual(out, [[int(buffer[0, k]) for k in idx[0]]])
        src = rng.integers(-128, 128, size=(3, 4))
        sel = [3, 0, 3]
        out = self.call("INDEX", "aten.index_select", [{"t": "i8", "v": src.tolist()}, 1,
                                                        {"t": "i64", "v": sel}]).fields["out"]
        self.assertEqual(out, [[int(src[r, c]) for c in sel] for r in range(3)])

    def test_embedding_forward_ignores_a_valid_padding_idx(self):
        rng = np.random.default_rng(11)
        weight = rng.integers(-50, 50, size=(6, 3)).astype(np.float32) / 4
        ids = np.array([[0, 5, 0, 2]])
        expected = [[[Fraction(float(v)) for v in weight[k]] for k in ids[0]]]
        for padding in (-1, 0, 5):
            with self.subTest(padding=padding):
                out = self.call("INDEX", "aten.embedding", [{"t": "f32", "v": weight.tolist()},
                                                            {"t": "i64", "v": ids.tolist()}, padding])
                self.assertEqual(out.fields["out"], expected)
        for padding in (None, 0, -6, 5):
            with self.subTest(public=padding):
                out = self.adapters["INDEX"].reference("real", {
                    "op": "torch.nn.functional.embedding", "args": [{"t": "i64", "v": ids.tolist()},
                                                                    {"t": "f32", "v": weight.tolist()}],
                    "kwargs": {"padding_idx": padding}})
                self.assertEqual(out.fields["out"], expected)

    def test_embedding_refusals(self):
        w = {"t": "f32", "v": [[1, 2], [3, 4]]}
        with self.assertRaises(Invalid):
            self.adapters["INDEX"].reference("real", {"op": "torch.nn.functional.embedding",
                                                      "args": [{"t": "i64", "v": [0]}, w], "kwargs": {"padding_idx": 2}})
        with self.assertRaises(Unsupported):
            self.call("INDEX", "aten.embedding", [w, {"t": "i64", "v": [0]}, 2])
        with self.assertRaises(Unsupported):
            self.call("INDEX", "aten.embedding", [{"t": "i64", "v": [[1, 2]]}, {"t": "i64", "v": [0]}])
        with self.assertRaises(Unsupported):
            self.adapters["INDEX"].reference("real", {"op": "torch.nn.functional.embedding",
                                                      "args": [{"t": "i64", "v": [0]}, w], "kwargs": {"max_norm": 1}})


class TorchCrossCheck(Staged):
    """Supporting evidence only: PyTorch 2.10 eager (CPU) agrees on dtype, shape and the values that are exact."""
    CIDS = ("ARITH", "INDEX")

    def test_eager_results_on_recorded_structures(self):
        import torch
        x = torch.tensor([[-2.0, 0.5, 3.0]])
        for e in (2, 3, 1.0, 2.0, 3.0):
            out = self.call("ARITH", "aten.pow", [{"t": "f32", "v": x.tolist()}, e]).fields["out"]
            self.assertEqual([float(v) for v in flatten(out)], torch.pow(x, e).reshape(-1).tolist())
        pos = torch.arange(128)
        self.assertEqual(self.call("ARITH", "aten.add", [{"t": "i64", "v": pos.tolist()}, 0]).fields["out"],
                         torch.ops.aten.add.Tensor(pos, 0).tolist())
        cond = torch.tensor([[True, False], [False, True]])
        eager = torch.ops.aten.where.self(cond, torch.tensor(float("-inf")), torch.tensor(0.0))
        ours = self.call("ARITH", "aten.where", [{"t": "b8", "v": cond.tolist()},
                                                 {"t": "f32", "v": float("-inf"), "shape": []},
                                                 {"t": "f32", "v": 0.0, "shape": []}]).fields["out"]
        self.assertEqual([-math.inf if v == "-inf" else float(v) for v in flatten(ours)], eager.reshape(-1).tolist())
        w = torch.randn(5, 3)
        ids = torch.tensor([[0, 4, 0]])
        eager = torch.ops.aten.embedding.default(w, ids, 0)
        ours = self.call("INDEX", "aten.embedding", [{"t": "f32", "v": w.double().tolist()},
                                                     {"t": "i64", "v": ids.tolist()}, 0]).fields["out"]
        self.assertEqual([float(v) for v in flatten(ours)], eager.reshape(-1).tolist())


if __name__ == "__main__":
    unittest.main()
