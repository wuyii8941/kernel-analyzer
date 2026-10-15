"""KV cache semantics. Source: the task definition of causal decoding: the output at position t computed incrementally
from a cache of keys/values for positions 0..t must equal the output of a causal prefill over the same sequence at t.
Covers a contiguous cache, a paged cache (block table), and RoPE applied at absolute positions before caching."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase2'))
import spec_attention as at

def _rot(xs, d, rope):
    return [at.rope_apply(x, t, d) for t, x in enumerate(xs)] if rope else xs

def prefill(Q, K, V, rope=False):
    d = len(Q[0]); Qr, Kr = _rot(Q, d, rope), _rot(K, d, rope)
    out, _ = at.attention(Qr, Kr, V, is_causal=True, alignment='upper_left'); return out

class ContiguousCache:
    def __init__(self): self.k = []; self.v = []
    def append(self, k, v): self.k.append(k); self.v.append(v)
    def read(self): return list(self.k), list(self.v)

class PagedCache:
    """Physical storage of fixed-size blocks; a block table maps logical block b -> physical block id."""
    def __init__(self, block_size, physical_order):
        self.B = block_size; self.order = list(physical_order); self.storage = {}; self.table = []; self.length = 0
    def append(self, k, v):
        b, off = divmod(self.length, self.B)
        if off == 0:
            if b >= len(self.order): raise ValueError('out of physical blocks')
            self.table.append(self.order[b]); self.storage[self.order[b]] = [None] * self.B
        self.storage[self.table[b]][off] = (k, v); self.length += 1
    def read(self):
        ks, vs = [], []
        for i in range(self.length):
            b, off = divmod(i, self.B); k, v = self.storage[self.table[b]][off]; ks.append(k); vs.append(v)
        return ks, vs

def decode(Q, K, V, cache, rope=False):
    d = len(Q[0]); outs = []
    for t in range(len(Q)):
        k = at.rope_apply(K[t], t, d) if rope else K[t]
        cache.append(k, V[t]); ks, vs = cache.read()
        q = at.rope_apply(Q[t], t, d) if rope else Q[t]
        o, _ = at.attention([q], ks, vs); outs.append(o[0])
    return outs
