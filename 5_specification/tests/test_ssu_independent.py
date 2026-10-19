"""Independent high-precision and public PyTorch checks for selective state update."""
import copy
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

import flint
import torch
import torch.nn.functional as torch_f

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import NotEstablished, ends, evaluate


PROFILE = "mamba232_real"


def _nested(shape, fn, prefix=()):
    if not shape:
        return fn(prefix)
    return [_nested(shape[1:], fn, prefix + (index,)) for index in range(shape[0])]


def _case(seed, batch=2, heads=4, dim=2, dstate=2, groups=2, *, softplus=False,
          with_d=True, with_z=True, with_bias=True, slots=None, indices=None):
    values = (-1, "-1/2", 0, "1/2", 1)
    pick = lambda index, shift=0: values[(sum((axis + 1) * value for axis, value in enumerate(index)) + seed + shift) % len(values)]
    slots = batch if slots is None else slots
    return {
        "state": _nested((slots, heads, dim, dstate), lambda index: pick(index)),
        "x": _nested((batch, heads, dim), lambda index: pick(index, 1)),
        "dt": _nested((batch, heads, dim), lambda index: pick(index, 3)),
        "A": _nested((heads, dim, dstate), lambda index: ("-1/2", "-1/4", 0)[(sum(index) + seed) % 3]),
        "B": _nested((batch, groups, dstate), lambda index: pick(index, 2)),
        "C": _nested((batch, groups, dstate), lambda index: pick(index, 4)),
        "D": _nested((heads, dim), lambda index: pick(index, 2)) if with_d else None,
        "z": _nested((batch, heads, dim), lambda index: pick(index, 4)) if with_z else None,
        "dt_bias": _nested((heads, dim), lambda index: pick(index, 1)) if with_bias else None,
        "dt_softplus": softplus,
        "state_batch_indices": indices,
        "dtype": "f32",
    }


def _headless(inp):
    out = copy.deepcopy(inp)
    out["state"] = [slot[0] for slot in out["state"]]
    for name in ("x", "dt", "z"):
        if out[name] is not None:
            out[name] = [row[0] for row in out[name]]
    out["A"] = out["A"][0]
    out["B"] = [row[0] for row in out["B"]]
    out["C"] = [row[0] for row in out["C"]]
    for name in ("D", "dt_bias"):
        if out[name] is not None:
            out[name] = out[name][0]
    return out


THRESHOLD = {
    "state": [[[[0], [0]]]], "x": [[[1, 1]]], "dt": [[[20, 21]]],
    "A": [[[0], [0]]], "B": [[[1]]], "C": [[[1]]], "D": [[0, 0]],
    "z": None, "dt_bias": [[0, 0]], "dt_softplus": True,
    "state_batch_indices": None, "dtype": "f32",
}

CASES = (
    _case(0, with_z=False),
    _case(1, softplus=True),
    _case(2, heads=2, groups=1, dim=3, dstate=3, with_z=False),
    _case(3, slots=3, indices=[2, 0]),
    _case(4, slots=3, indices=[1, -1]),
    _headless(_case(5, heads=1, groups=1, with_z=False)),
    THRESHOLD,
)


def _arb(value):
    value = F(value)
    return flint.arb(value.numerator) / flint.arb(value.denominator)


def _fraction(endpoint):
    mantissa, exponent = endpoint.man_exp()
    return F(int(mantissa)) * F(2) ** int(exponent)


def _as_heads(inp):
    heads = len(_shape(inp["state"])) == 4
    if heads:
        return copy.deepcopy(inp), True
    out = copy.deepcopy(inp)
    out["state"] = [[slot] for slot in out["state"]]
    for name in ("x", "dt", "z"):
        if out[name] is not None:
            out[name] = [[row] for row in out[name]]
    out["A"] = [out["A"]]
    out["B"] = [[row] for row in out["B"]]
    out["C"] = [[row] for row in out["C"]]
    for name in ("D", "dt_bias"):
        if out[name] is not None:
            out[name] = [out[name]]
    return out, False


def _shape(value):
    return () if not isinstance(value, list) else (len(value),) + _shape(value[0])


def _independent(inp):
    source, had_heads = _as_heads(inp)
    state = _nested(_shape(source["state"]), lambda index: _arb(source["state"][index[0]][index[1]][index[2]][index[3]]))
    batch, heads, dim = _shape(source["x"])
    groups, dstate = len(source["B"][0]), len(source["B"][0][0])
    repeats = heads // groups
    y = _nested((batch, heads, dim), lambda _: flint.arb(0))
    for batch_index in range(batch):
        slot = batch_index if source["state_batch_indices"] is None else source["state_batch_indices"][batch_index]
        if slot < 0:
            continue
        for head in range(heads):
            group = head // repeats
            for channel in range(dim):
                delta = _arb(source["dt"][batch_index][head][channel])
                if source["dt_bias"] is not None:
                    delta += _arb(source["dt_bias"][head][channel])
                # All test deltas are exact, so this comparison directly checks the documented strict threshold.
                if source["dt_softplus"] and delta.mid() <= 20:
                    delta = (flint.arb(1) + delta.exp()).log()
                value = flint.arb(0)
                for coordinate in range(dstate):
                    decay = (delta * _arb(source["A"][head][channel][coordinate])).exp()
                    incoming = delta * _arb(source["B"][batch_index][group][coordinate])
                    state[slot][head][channel][coordinate] = (
                        state[slot][head][channel][coordinate] * decay
                        + incoming * _arb(source["x"][batch_index][head][channel])
                    )
                    value += state[slot][head][channel][coordinate] * _arb(source["C"][batch_index][group][coordinate])
                if source["D"] is not None:
                    value += _arb(source["x"][batch_index][head][channel]) * _arb(source["D"][head][channel])
                if source["z"] is not None:
                    gate = _arb(source["z"][batch_index][head][channel])
                    value *= gate / (flint.arb(1) + (-gate).exp())
                y[batch_index][head][channel] = value
    if not had_heads:
        y = [row[0] for row in y]
        state = [slot[0] for slot in state]
    return {"out": {"y": y, "state": state}}


def _visit_enclosure(test, actual, independent):
    if isinstance(independent, dict):
        test.assertEqual(set(actual), set(independent))
        for key in independent:
            _visit_enclosure(test, actual[key], independent[key])
        return
    if isinstance(independent, list):
        test.assertEqual(len(actual), len(independent))
        for left, right in zip(actual, independent):
            _visit_enclosure(test, left, right)
        return
    lo, hi = ends(actual)
    if lo == hi:
        test.assertTrue(independent.contains(_arb(lo)))
    else:
        test.assertLessEqual(lo, _fraction(independent.lower()))
        test.assertGreaterEqual(hi, _fraction(independent.upper()))


def _midpoint(value):
    if isinstance(value, dict):
        return {key: _midpoint(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_midpoint(item) for item in value]
    lo, hi = ends(value)
    return float((lo + hi) / 2)


def _torch_formula(inp):
    source, had_heads = _as_heads(inp)
    def tensor(value):
        if value is None:
            return None
        def convert(item):
            if isinstance(item, list):
                return [convert(child) for child in item]
            return float(F(item))
        return torch.tensor(convert(value), dtype=torch.float32)
    state = tensor(source["state"])
    x, dt, A, B, C = (tensor(source[name]) for name in ("x", "dt", "A", "B", "C"))
    D, z, bias = (tensor(source[name]) for name in ("D", "z", "dt_bias"))
    if bias is not None:
        dt = dt + bias
    if source["dt_softplus"]:
        dt = torch_f.softplus(dt)
    batch, heads, dim = x.shape
    repeats = heads // B.shape[1]
    expanded_b = B.repeat_interleave(repeats, dim=1)
    expanded_c = C.repeat_interleave(repeats, dim=1)
    y = torch.zeros_like(x)
    for batch_index in range(batch):
        slot = batch_index if source["state_batch_indices"] is None else source["state_batch_indices"][batch_index]
        if slot < 0:
            continue
        next_state = (
            state[slot] * torch.exp(dt[batch_index].unsqueeze(-1) * A)
            + dt[batch_index].unsqueeze(-1) * expanded_b[batch_index].unsqueeze(1) * x[batch_index].unsqueeze(-1)
        )
        state[slot] = next_state
        out = torch.einsum("hdn,hn->hd", next_state, expanded_c[batch_index])
        if D is not None:
            out = out + x[batch_index] * D
        if z is not None:
            out = out * torch_f.silu(z[batch_index])
        y[batch_index] = out
    if not had_heads:
        y, state = y[:, 0], state[:, 0]
    return y, state


class IndependentSelectiveStateUpdate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = registry.load("SSU").adapter

    def test_reference_encloses_independent_768_bit_element_loop(self):
        saved = flint.ctx.prec
        self.addCleanup(setattr, flint.ctx, "prec", saved)
        flint.ctx.prec = 768
        for index, inp in enumerate(CASES):
            with self.subTest(case=index):
                before = copy.deepcopy(inp)
                reference = self.adapter.reference(PROFILE, inp).fields
                _visit_enclosure(self, reference, _independent(inp))
                self.assertEqual(inp, before)

    def test_public_pytorch_formula_matches_all_value_and_state_coordinates(self):
        for index, inp in enumerate(CASES):
            with self.subTest(case=index):
                y, state = _torch_formula(inp)
                reference = _midpoint(self.adapter.reference(PROFILE, inp).fields["out"])
                torch.testing.assert_close(y.double(), torch.tensor(reference["y"], dtype=torch.float64), rtol=2e-5, atol=2e-5)
                torch.testing.assert_close(state.double(), torch.tensor(reference["state"], dtype=torch.float64), rtol=2e-5, atol=2e-5)

    def test_softplus_uses_the_linear_branch_only_above_twenty(self):
        reference = _midpoint(self.adapter.reference(PROFILE, THRESHOLD).fields["out"])
        self.assertAlmostEqual(reference["state"][0][0][0][0], torch_f.softplus(torch.tensor(20.0)).item(), places=5)
        self.assertEqual(reference["state"][0][0][1][0], 21.0)

    def test_complete_output_checker_requires_both_value_and_post_state(self):
        inp = _case(0, batch=1, heads=1, dim=1, dstate=1, groups=1, with_z=False)
        inp["A"] = [[[0]]]
        exact = _midpoint(self.adapter.reference(PROFILE, inp).fields["out"])
        candidate = {"out": {
            "y": {"t": "f32", "shape": [1, 1, 1], "v": exact["y"]},
            "state": {"t": "f32", "shape": [1, 1, 1, 1], "v": exact["state"]},
        }}
        self.assertTrue(self.adapter.check_output(PROFILE, inp, candidate))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {"out": {"y": candidate["out"]["y"]}}))
        self.assertFalse(self.adapter.check_output(PROFILE, inp, {**candidate, "extra": []}))
        wrong_dtype = copy.deepcopy(candidate)
        wrong_dtype["out"]["state"]["t"] = "f16"
        self.assertFalse(self.adapter.check_output(PROFILE, inp, wrong_dtype))
        wrong_shape = copy.deepcopy(candidate)
        wrong_shape["out"]["y"]["shape"] = [1]
        self.assertFalse(self.adapter.check_output(PROFILE, inp, wrong_shape))
        wrong_value = copy.deepcopy(candidate)
        wrong_value["out"]["state"]["v"][0][0][0][0] += 1
        self.assertFalse(self.adapter.check_output(PROFILE, inp, wrong_value))

    def test_collisions_and_inputs_outside_the_reviewed_profile_are_refused(self):
        base = _case(0, slots=2, indices=[0, 1])
        self.assertEqual(evaluate(
            self.adapter.reference, PROFILE, dict(base, state_batch_indices=[0, 0])).outcome,
            "not_established",
        )
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, dtype="f16")).outcome, "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, D=None)).outcome, "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, dt_bias=None)).outcome, "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, x=[[[float("inf")]]])).outcome, "unsupported")
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, dict(base, state_batch_indices=[0, 2])).outcome, "invalid_input")
        missing = copy.deepcopy(base)
        del missing["D"]
        self.assertEqual(evaluate(self.adapter.reference, PROFILE, missing).outcome, "invalid_input")


if __name__ == "__main__":
    unittest.main()
