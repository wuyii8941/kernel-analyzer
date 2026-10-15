import os, sys, unittest, random
from fractions import Fraction as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, '..', 'phase2'), os.path.join(HERE, '..', '..')]
import spec_ema as ema, spec_kvcache as kv, spec_kvcomp as kc, spec_sparse as sp, spec_state_io as io
import spec_optimizers as so
from derive import enclosure_zero

def rq(rng, lo=-3, hi=3, den=5): return F(rng.randint(lo * den, hi * den), den)
class T(unittest.TestCase):
    # ---- EMA / SWA
    def test_first_update_copies(self):
        st = ema.AveragedState(); ema.ema_update(st, [F(3), F(-1)], F(9, 10)); self.assertEqual(st.params, [3, -1]); self.assertEqual(st.n_averaged, 1)
    def test_ema_second_update(self):
        st = ema.AveragedState(); ema.ema_update(st, [F(0)], F(9, 10)); ema.ema_update(st, [F(10)], F(9, 10)); self.assertEqual(st.params, [1])
    def test_swa_mean_and_ema_closed_form(self):
        rng = random.Random(1)
        for _ in range(50):
            hist = [[rq(rng) for _ in range(3)] for _ in range(rng.randint(1, 6))]
            a, b = ema.prop_swa_is_mean(hist); self.assertEqual(a, b)
            a, b = ema.prop_ema_closed_form(hist, F(rng.randint(0, 10), 10)); self.assertEqual(a, b)
    def test_ema_length_and_decay_contract(self):
        st = ema.AveragedState(); ema.ema_update(st, [F(1)], F(1, 2))
        with self.assertRaises(ValueError): ema.ema_update(st, [F(1), F(2)], F(1, 2))
        with self.assertRaises(ValueError): ema.ema_update(ema.AveragedState(), [F(1)], F(3, 2))
    # ---- KV cache
    def _qkv(self, rng, L, d):
        return ([[rq(rng) for _ in range(d)] for _ in range(L)], [[rq(rng) for _ in range(d)] for _ in range(L)], [[rq(rng) for _ in range(2)] for _ in range(L)])
    def test_decode_equals_prefill_contiguous_paged_rope(self):
        rng = random.Random(2)
        for _ in range(8):
            L, d = rng.randint(1, 5), rng.choice([2, 4]); Q, K, V = self._qkv(rng, L, d)
            for rope in (False, True):
                pre = kv.prefill(Q, K, V, rope=rope)
                for cache in (kv.ContiguousCache(), kv.PagedCache(2, [3, 0, 2, 1])):
                    dec = kv.decode(Q, K, V, cache, rope=rope)
                    self.assertTrue(all(enclosure_zero(a - b) for ra, rb in zip(pre, dec) for a, b in zip(ra, rb)))
    def test_decode_mutation_wrong_rope_position(self):     # caching keys rotated at position 0 must break the relation
        rng = random.Random(3); Q, K, V = self._qkv(rng, 4, 2)
        pre = kv.prefill(Q, K, V, rope=True)
        import spec_attention as at
        cache = kv.ContiguousCache(); outs = []
        for t in range(4):
            cache.append(at.rope_apply(K[t], 0, 2), V[t]); ks, vs = cache.read(); o, _ = at.attention([at.rope_apply(Q[t], t, 2)], ks, vs); outs.append(o[0])
        self.assertFalse(all(enclosure_zero(a - b) for ra, rb in zip(pre, outs) for a, b in zip(ra, rb)))
    # ---- KV compression
    def test_windows(self):
        self.assertEqual(kc.windows(10, 4, 4), [(0, [0, 1, 2, 3]), (1, [4, 5, 6, 7])]); self.assertEqual(len(kc.windows(10, 4, 2)), 4)
    def test_shift_by_stride_equivariance(self):
        rng = random.Random(4)
        for mode in ('uniform', 'gated'):
            for per in (False, True):
                W = S = 3; d = 2; H = [[rq(rng) for _ in range(d)] for _ in range(9)]
                C = [[[rq(rng) for _ in range(d)] for _ in range(d)] for _ in range(W)] if per else [[rq(rng) for _ in range(d)] for _ in range(d)]
                Z = [[rq(rng) for _ in range(d)] for _ in range(W)] if per else [rq(rng) for _ in range(d)]
                b = [rq(rng) for _ in range(W)] if per else rq(rng)
                pad = [[rq(rng) for _ in range(d)] for _ in range(S)]
                a = kc.compress(H, W, S, C, mode, Z, b, per); s = kc.compress(pad + H, W, S, C, mode, Z, b, per)
                ok = all((x == y) if mode == 'uniform' else enclosure_zero(x - y) for ra, rb in zip(a, s[1:]) for x, y in zip(ra, rb))
                self.assertTrue(ok, (mode, per))
    def test_shift_by_one_breaks_membership(self):
        H = [[F(i)] for i in range(8)]; C = [[F(1)]]
        a = kc.compress(H, 4, 4, C); s = kc.compress([[F(100)]] + H, 4, 4, C)
        self.assertNotEqual(a[0], s[0]); self.assertNotEqual(a[0], s[1])
    def test_within_window_permutation(self):
        H = [[F(1)], [F(2)], [F(3)], [F(4)]]; Hp = [H[1], H[0], H[3], H[2]]
        self.assertEqual(kc.compress(H, 4, 4, [[F(2)]]), kc.compress(Hp, 4, 4, [[F(2)]]))               # shared: invariant
        C = [[[F(1)]], [[F(2)]], [[F(3)]], [[F(4)]]]
        self.assertNotEqual(kc.compress(H, 4, 4, C, per_offset=True), kc.compress(Hp, 4, 4, C, per_offset=True))  # per offset: not
    def test_causal_visibility(self):
        self.assertEqual(kc.visible_windows(4, 4, 4, 12), [0]); self.assertEqual(kc.visible_windows(3, 4, 4, 12), [])
    # ---- sparse
    def test_sparse_paths(self):
        rng = random.Random(5)
        for _ in range(100):
            r, c, m = rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 3); n = rng.randint(0, 8)
            idx = [[rng.randrange(r), rng.randrange(c)] for _ in range(n)]; val = [rq(rng) for _ in range(n)]; B = [[rq(rng) for _ in range(m)] for _ in range(c)]
            D = sp.to_dense(idx, val, (r, c))
            self.assertEqual(sp.spmm(idx, val, (r, c), B), [[sum((D[i][k] * B[k][j] for k in range(c)), F(0)) for j in range(m)] for i in range(r)])
            ci, cv = sp.coalesce(idx, val, (r, c)); self.assertEqual(sp.to_dense(ci, cv, (r, c)), D)
            perm = list(range(n)); rng.shuffle(perm); self.assertEqual(sp.to_dense([idx[p] for p in perm], [val[p] for p in perm], (r, c)), D)
    def test_sparse_contract(self):
        with self.assertRaises(ValueError): sp.to_dense([[5, 0]], [F(1)], (2, 2))
        with self.assertRaises(ValueError): sp.to_dense([[0, 0]], [F(1), F(2)], (2, 2))
    # ---- save / load
    def test_adam_resume_equals_continue_and_mutation(self):
        th0, g1, g2 = [F(1), F(-2)], [F(1, 10), F(3, 10)], [F(-1, 5), F(1, 7)]
        st = so.AdamState(2); th1 = so.adam_step(th0, g1, st, F(1, 100)); blob = io.save(st)
        cont = so.adam_step(th1, g2, st, F(1, 100))
        res = so.adam_step(th1, g2, io.load(blob, so.AdamState), F(1, 100))
        self.assertTrue(all(enclosure_zero(a - b) for a, b in zip(cont, res)))
        bad = so.adam_step(th1, g2, io.load(blob, so.AdamState, drop=('t',)), F(1, 100))   # forgetting the step counter
        self.assertFalse(all(enclosure_zero(a - b) for a, b in zip(cont, bad)))
    def test_ema_resume(self):
        st = ema.AveragedState(); ema.ema_update(st, [F(1)], F(9, 10)); blob = io.save(st)
        a = ema.ema_update(st, [F(5)], F(9, 10)); b = ema.ema_update(io.load(blob, ema.AveragedState), [F(5)], F(9, 10)); self.assertEqual(a, b)
        c = ema.ema_update(io.load(blob, ema.AveragedState, drop=('n_averaged',)), [F(5)], F(9, 10)); self.assertNotEqual(a, c)
if __name__ == '__main__': unittest.main()
