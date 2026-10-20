"""kernel_spec.bulk against the contract adapters: same outcome, and the same real-number result.

Exact entries (MATMUL, ARITH, INDEX): bulk lo / hi must equal the directed binary64 roundings of the adapter's exact
value at every coordinate. Enclosure entries (SOFTMAX, ACT): both are rigorous enclosures of one real number, so they
must intersect at every coordinate, and the bulk width must stay within a few binary64 ulps. Refusals: where bulk
declines (None) nothing is claimed; where it raises, the adapter must give the same outcome.
"""
import math
import random
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

SPEC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SPEC)]
from kernel_spec import bulk, registry  # noqa: E402
from kernel_spec.num import IV, Invalid, Unsupported, ends, evaluate  # noqa: E402
from kernel_spec.real import R  # noqa: E402

TORCH = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32, "f64": torch.float64}


def record(arr, dtype):
    t = torch.as_tensor(np.asarray(arr, dtype=np.float64)).to(TORCH[dtype])
    return {"t": dtype, "v": t.double().tolist(), "shape": list(t.shape)}


def index_record(arr):
    a = np.asarray(arr, dtype=np.int64)
    return {"t": "i64", "v": a.tolist(), "shape": list(a.shape)}


def flat(value):
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(flat(v))
        return out
    return [value.v if isinstance(value, R) else value]


def directed(q):
    f = float(q)
    fq = Fraction(f)
    if fq == q:
        return f, f
    return (f, math.nextafter(f, math.inf)) if fq < q else (math.nextafter(f, -math.inf), f)


class Bulk(unittest.TestCase):
    def compare(self, cid, profile, inp, exact):
        adapter = registry.load(cid).adapter
        ref = evaluate(adapter.reference, profile, inp)
        try:
            got = bulk.evaluate(cid, profile, inp, force=True)
        except (Invalid, Unsupported) as error:
            expected = "invalid_input" if isinstance(error, Invalid) else "unsupported"
            self.assertEqual(ref.outcome, expected, f"bulk raised {error!r}; adapter {ref.outcome}: {ref.note}")
            return "raised"
        if got is None:
            return "declined"
        self.assertEqual(ref.outcome, "value", f"bulk evaluated a call the adapter answers with {ref.outcome}: {ref.note}")
        b = got.fields["out"]
        values = flat(ref.fields["out"])
        lo, hi = b.lo.ravel(), b.hi.ravel()
        self.assertEqual(len(values), lo.size)
        self.assertEqual(b.exact, exact)
        for i, v in enumerate(values):
            if exact:
                self.assertIsInstance(v, (int, Fraction))
                self.assertEqual((lo[i], hi[i]), directed(Fraction(v)), f"coordinate {i}")
            else:
                a_lo, a_hi = (Fraction(v), Fraction(v)) if isinstance(v, (int, Fraction)) else ends(v)
                self.assertLessEqual(Fraction(lo[i]), a_hi, f"coordinate {i}")
                self.assertGreaterEqual(Fraction(hi[i]), a_lo, f"coordinate {i}")
                width_ulps = (hi[i] - lo[i]) / max(math.ulp(max(abs(lo[i]), abs(hi[i]))), 2.0 ** -1074)
                self.assertLessEqual(width_ulps, 8, f"coordinate {i}: bulk enclosure {width_ulps} ulps wide")
        return "evaluated"

    def rng(self, seed):
        return np.random.default_rng(seed)

    def test_matmul(self):
        seen = set()
        for seed in range(24):
            g = self.rng(seed)
            dtype = ["f32", "bf16", "f16", "f64"][seed % 4]
            m, k, n = (int(x) for x in g.integers(1, 7, 3))
            scale = [1, 1e-3, 1e3, 1e-20][seed % 4] if dtype in ("f32", "bf16", "f64") else 1
            a = g.standard_normal((m, k)) * scale
            b = g.standard_normal((k, n))
            seen.add(self.compare("MATMUL", "real", {"op": "aten.mm", "args": [record(a, dtype), record(b, dtype)]}, True))
            c = g.standard_normal((n,) if seed % 2 else (m, n))
            kwargs = [{}, {"beta": 0}, {"beta": 0.5, "alpha": 3}, {"alpha": -2}][seed % 4]
            seen.add(self.compare("MATMUL", "real", {"op": "aten.addmm", "args": [record(c, dtype), record(a, dtype),
                                                                                  record(b, dtype)], "kwargs": kwargs}, True))
            bb = g.integers(1, 4)
            seen.add(self.compare("MATMUL", "real", {"op": "aten.bmm", "args": [
                record(g.standard_normal((bb, m, k)), dtype), record(g.standard_normal((bb, k, n)), dtype)]}, True))
        self.assertIn("evaluated", seen)

    def test_matmul_refusals(self):
        g = self.rng(100)
        a, b = g.standard_normal((2, 3)), g.standard_normal((3, 2))
        inf_a = a.copy()
        inf_a[0, 0] = math.inf
        cases = [{"op": "aten.mm", "args": [record(inf_a, "f32"), record(b, "f32")]},
                 {"op": "aten.mm", "args": [record(a, "f32"), record(b, "bf16")]},
                 {"op": "aten.mm", "args": [record(a, "f32"), record(a, "f32")]},
                 {"op": "aten.addmm", "args": [record(g.standard_normal((3, 3)), "f32"), record(a, "f32"),
                                               record(b, "f32")]},
                 {"op": "aten.addmm", "args": [record(inf_a[:, :2], "f32"), record(a, "f32"), record(b, "f32")],
                  "kwargs": {"beta": 0}},
                 {"op": "aten.mm", "args": [{"t": "i64", "v": [[1, 2]], "shape": [1, 2]}, record(b[:2], "f32")]}]
        for inp in cases:
            self.assertIn(self.compare("MATMUL", "real", inp, True), ("declined", "raised"))

    def test_arith(self):
        for seed in range(18):
            g = self.rng(200 + seed)
            dtype = ["f32", "bf16", "f16"][seed % 3]
            shape = tuple(int(x) for x in g.integers(1, 5, 2))
            a = g.standard_normal(shape) * [1, 1e8, 1e-8][seed % 3]
            b = g.standard_normal(shape)
            for op in ("aten.add", "aten.sub", "aten.mul"):
                self.assertEqual(self.compare("ARITH", "real", {"op": op, "args": [record(a, dtype), record(b, dtype)]},
                                              True), "evaluated")
        g = self.rng(250)
        a = g.standard_normal((2, 2))
        self.assertEqual(self.compare("ARITH", "real", {"op": "aten.add", "args": [record(a, "f32"), record(a, "f32")],
                                                        "kwargs": {"alpha": 1}}, True), "evaluated")
        for inp in ({"op": "aten.add", "args": [record(a, "f32"), record(a[0], "f32")]},
                    {"op": "aten.add", "args": [record(a, "f32"), record(a, "f32")], "kwargs": {"alpha": 2}},
                    {"op": "aten.add", "args": [record(a, "f32"), 1.5]},
                    {"op": "aten.mul", "args": [record(a, "f64"), record(a, "f64")]}):
            self.assertIn(self.compare("ARITH", "real", inp, True), ("declined", "raised"))

    def test_index(self):
        for seed in range(8):
            g = self.rng(300 + seed)
            dtype = ["f32", "bf16"][seed % 2]
            w = g.standard_normal((int(g.integers(2, 9)), int(g.integers(1, 5))))
            idx = g.integers(0, w.shape[0], (int(g.integers(1, 4)), int(g.integers(1, 4))))
            self.assertEqual(self.compare("INDEX", "real", {"op": "aten.embedding",
                                                            "args": [record(w, dtype), index_record(idx)]}, True),
                             "evaluated")
        w = np.array([[1.0, -0.0], [2.0, 3.0]])
        self.assertEqual(self.compare("INDEX", "real", {"op": "aten.embedding",
                                                        "args": [record(w, "f32"), index_record([[0, 1]])]}, True),
                         "raised")
        self.assertIn(self.compare("INDEX", "real", {"op": "aten.embedding", "args": [
            record(np.ones((2, 2)), "f32"), index_record([[0, 2]])]}, True), ("declined", "raised"))

    def test_softmax(self):
        for seed in range(10):
            g = self.rng(400 + seed)
            dtype = ["f32", "bf16"][seed % 2]
            shape = tuple(int(x) for x in g.integers(1, 5, 2))
            x = g.standard_normal(shape) * [1, 30, 1e-3, 80, 5][seed % 5]
            dim = int(g.integers(-2, 2))
            self.assertEqual(self.compare("SOFTMAX", "real", {"op": "aten._log_softmax",
                                                              "args": [record(x, dtype), dim, False]}, False),
                             "evaluated")
            y = torch.log_softmax(torch.tensor(x), dim).numpy()
            gr = g.standard_normal(shape)
            self.assertEqual(self.compare("SOFTMAX", "real", {"op": "aten._log_softmax_backward_data", "args": [
                record(gr, dtype), record(y, dtype), dim, {"dtype": dtype}]}, False), "evaluated")

    def test_act(self):
        for seed in range(8):
            g = self.rng(500 + seed)
            dtype = ["f32", "bf16"][seed % 2]
            shape = tuple(int(x) for x in g.integers(1, 6, 2))
            x = g.standard_normal(shape) * [1, 10, 1e-4, 40][seed % 4]
            x.flat[0] = -1.2784645  # near the zero of silu'
            self.assertEqual(self.compare("ACT", "torch", {"op": "aten.silu_backward", "args": [
                record(g.standard_normal(shape), dtype), record(x, dtype)]}, False), "evaluated")
        self.assertIn(self.compare("ACT", "torch", {"op": "aten.silu_backward", "args": [
            record(np.ones((2, 2)), "f32"), record(np.ones((2,)), "f32")]}, False), ("declined", "raised"))

    def test_cast(self):
        statement = "roundTiesToEven conversion (test)"
        adapter = registry.load("CAST").adapter
        for seed in range(12):
            g = self.rng(600 + seed)
            src = ["f32", "f64", "bf16", "f16"][seed % 4]
            dst = ["bf16", "f16", "f32", "bf16", "f64"][seed % 5]
            x = g.standard_normal((3, 5)) * [1, 1e3, 1e-3, 1e-6, 6e4][seed % 5]
            if dst == "f16":
                x = np.clip(x, -6e4, 6e4)
            x.flat[0] = 1 + 2.0 ** -8 if src in ("f32", "f64") else x.flat[0]   # a bf16 tie
            x.flat[1] = 2.0 ** -133 if src in ("f32", "f64") else x.flat[1]     # bf16 subnormal range
            inp = {"op": "aten._to_copy", "args": [record(x, src)], "kwargs": {"dtype": {"dtype": dst}}}
            ref = evaluate(adapter.reference, "finite_rne", inp)
            got = bulk.evaluate("CAST", "finite_rne", inp, force=True)
            if got is None:
                continue
            self.assertEqual(ref.outcome, "value", ref.note)
            for i, v in enumerate(flat(ref.fields["out"])):
                self.assertEqual((got.fields["out"].lo.ravel()[i], got.fields["out"].hi.ravel()[i]),
                                 directed(Fraction(v)), (src, dst, i))
        big = {"op": "aten._to_copy", "args": [record(np.array([3e38]), "f32")], "kwargs": {"dtype": {"dtype": "f16"}}}
        self.assertIsNone(bulk.evaluate("CAST", "finite_rne", big, force=True))
        del statement

    def test_nll(self):
        adapter = registry.load("NLL").adapter
        for seed in range(12):
            g = self.rng(700 + seed)
            n, c = int(g.integers(1, 6)), int(g.integers(2, 7))
            dtype = ["f32", "bf16"][seed % 2]
            x = -np.abs(g.standard_normal((n, c))) * 3
            t = g.integers(0, c, n)
            if seed % 3 == 0:
                t[0] = -100
            if seed == 6:
                t[:] = -100
            red = seed % 3
            fwd = {"op": "aten.nll_loss_forward", "args": [record(x, dtype), index_record(t), None, red, -100]}
            ref = evaluate(adapter.reference, "aten_real", fwd)
            got = bulk.evaluate("NLL", "aten_real", fwd, force=True)
            if got is None:
                self.assertNotEqual(ref.outcome, "value") if seed == 6 and red == 1 else None
            else:
                self.assertEqual(ref.outcome, "value", ref.note)
                self.assertEqual(flat(ref.fields["out"]), flat(got.fields["out"]))
            go = record(g.standard_normal((n,)) if red == 0 else g.standard_normal(()), "f32")
            tw = record(np.array(float(max(1, (t != -100).sum()))), "f32")
            bwd = {"op": "aten.nll_loss_backward", "args": [go, record(x, dtype), index_record(t), None, red, -100, tw]}
            self.assertEqual(self.compare("NLL", "aten_real", bwd, True), "evaluated")

    def test_threshold_and_unlisted(self):
        small = {"op": "aten.mm", "args": [record(np.ones((2, 2)), "f32"), record(np.ones((2, 2)), "f32")]}
        self.assertIsNone(bulk.evaluate("MATMUL", "real", small))
        self.assertIsNone(bulk.evaluate("NORM", "torch", small, force=True))


if __name__ == "__main__":
    unittest.main()
