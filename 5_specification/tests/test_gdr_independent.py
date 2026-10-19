"""Independent affine expansion of GDN Eq. 10, in the transposed FLA layout.

Source: sources/GDN_PAPER.txt.gz, Eq. 10; the FLA interface choices are in
sources/FLA_GDR_NAIVE.py.gz. This tests the declared recurrence, not all FLA kernels.
"""
import copy
import ast
import gzip
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import d2_linear as L, registry
from kernel_spec.real import R


def mm(a, b):
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), F(0))
             for j in range(len(b[0]))] for i in range(len(a))]


def expanded(q, k, v, alpha, beta, scale, initial):
    """H_t = (A_t ... A_0) H_initial + sum_s (A_t ... A_{s+1}) B_s.

    Each prefix is recomputed from matrix products; no recurrent state or
    triangular solve from the implementation under test is used.
    """
    K, V = len(initial), len(initial[0])
    ident = [[F(i == j) for j in range(K)] for i in range(K)]
    A = [[[alpha[t] * (F(i == j) - beta[t] * k[t][i] * k[t][j])
           for j in range(K)] for i in range(K)] for t in range(len(q))]
    B = [[[beta[t] * k[t][i] * v[t][j] for j in range(V)] for i in range(K)]
         for t in range(len(q))]
    out, state = [], copy.deepcopy(initial)
    for t in range(len(q)):
        product = ident
        terms = []
        for s in range(t, -1, -1):
            terms.append(mm(product, B[s]))
            product = mm(product, A[s])
        base = mm(product, initial)
        state = [[base[i][j] + sum((term[i][j] for term in terms), F(0))
                  for j in range(V)] for i in range(K)]
        out.append([scale * sum((q[t][i] * state[i][j] for i in range(K)), F(0)) for j in range(V)])
    return out, state


def wrap(x):
    return [wrap(y) for y in x] if isinstance(x, list) else R(x)


def plain(x):
    return [plain(y) for y in x] if isinstance(x, (tuple, list)) else x.v


class IndependentGDR(unittest.TestCase):
    def test_inactive_gate_parameters_match_archived_fla_interface(self):
        # Execute only the retained public Python wrapper, with its GPU backend
        # replaced by an argument recorder. This checks API acceptance, not GPU arithmetic.
        source = gzip.open(Path(__file__).resolve().parents[1] / "sources/FLA_GDR_CHUNK.py.gz", "rt").read()
        wrapper = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                       and n.name == "chunk_gated_delta_rule")
        wrapper.decorator_list = []
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                  wrapper], type_ignores=[])
        env = {"ChunkGatedDeltaRuleFunction": SimpleNamespace(apply=lambda *args: (args, None))}
        exec(compile(ast.fix_missing_locations(module), "retained_fla_wrapper", "exec"), env)
        tensor = SimpleNamespace(shape=(1, 1, 1, 1))
        adapter = registry.load("GDR").adapter
        inp = dict(q=[[[[1]]]], k=[[[[1]]]], v=[[[[3]]]], g=[[[0]]], beta=[[[1]]],
                   output_final_state=True)
        for options in ({"A_log": None, "dt_bias": None}, {"A_log": [17], "dt_bias": [2]}):
            call, _ = env["chunk_gated_delta_rule"](tensor, tensor, tensor, tensor, tensor, **options)
            self.assertIs(call[13], False)
            self.assertEqual(call[14:16], (options["A_log"], options["dt_bias"]))
            result = adapter.reference("fla_chunk", dict(inp, **options))
            self.assertEqual(result.fields, {"out": {"o": [[[[F(3)]]]], "final_state": [[[[F(3)]]]]}})

    def fixture(self, T=17):
        q = [[F((t + i) % 5 - 2, 4) for i in range(4)] for t in range(T)]
        k = [[F((2*t + i) % 7 - 3, 8) for i in range(4)] for t in range(T)]
        v = [[F((t + j) % 5 - 2, 3) for j in range(3)] for t in range(T)]
        h = [[F(i - j + 1, 5) for j in range(3)] for i in range(4)]
        return q, k, v, h

    def test_recurrence_and_chunk_boundary_against_matrix_expansion(self):
        q, k, v, h = self.fixture()
        alpha = [F(1, 2) if t % 2 else F(1) for t in range(len(q))]
        beta = [F(t % 4, 3) for t in range(len(q))]
        expected = list(expanded(q, k, v, alpha, beta, F(1, 2), h))
        args = [wrap(x) for x in (q, k, v, alpha, beta, F(1, 2), h)]
        self.assertEqual(plain(L.recurrent(*args)), expected)
        for chunk in (1, 4, 16, 32):
            with self.subTest(chunk=chunk):
                self.assertEqual(plain(L.chunked(*args, chunk)), expected)

    def test_adapter_default_scale_rectangular_state_and_layout(self):
        adapter = registry.load("GDR").adapter
        q, k, v, h = self.fixture(3)
        beta = [F(1, 2)] * 3
        expected_o, expected_h = expanded(q, k, v, [F(1)] * 3, beta, F(1, 2), h)
        inp = dict(q=[[ [row] for row in q]], k=[[ [row] for row in k]],
                   v=[[ [row] for row in v]], g=[[[0]] * 3], beta=[[[b] for b in beta]],
                   initial_state=[[h]], output_final_state=True)
        original = copy.deepcopy(inp)
        for fn in (adapter.reference, adapter.READINGS["chunked"]):
            out = fn("fla_chunk", inp).fields["out"]
            self.assertEqual(out, {"o": [[[row] for row in expected_o]], "final_state": [[expected_h]]})
        self.assertEqual(inp, original)
        inp["state_v_first"] = True
        inp["initial_state"] = [[list(map(list, zip(*h)))]]
        out = adapter.reference("fla_chunk", inp).fields["out"]
        self.assertEqual(out["final_state"], [[list(map(list, zip(*expected_h)))]] )

    def test_empty_sequence_keeps_its_own_pre_state(self):
        adapter = registry.load("GDR").adapter
        inp = dict(q=[[[[1]], [[1]]]], k=[[[[1]], [[1]]]], v=[[[[3]], [[4]]]],
                   g=[[[0], [0]]], beta=[[[0], [0]]], cu_seqlens=[0, 1, 1, 2],
                   initial_state=[[[[2]]], [[[7]]], [[[5]]]], output_final_state=True)
        for fn in (adapter.reference, adapter.READINGS["chunked"]):
            out = fn("fla_chunk", inp).fields["out"]
            self.assertEqual(out["o"], [[[[2]], [[5]]]])
            self.assertEqual(out["final_state"], inp["initial_state"])


if __name__ == "__main__":
    unittest.main()
