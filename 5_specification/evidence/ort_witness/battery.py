"""Step 2, run 1: our executable specs against witnesses that share no code with PyTorch:
ONNX Runtime 1.24.4 (Microsoft, C++, semantics defined by the ONNX operator specification) and SciPy / NumPy.
Rule: no witness is the arbiter. Each disagreement is recorded with both values and triaged against the documents."""
import sys, json, random, traceback
from fractions import Fraction as F
import numpy as np, scipy.special as ss, mpmath as mp
import os
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..', '..', 'specification_v2')
sys.path[:0] = [R, R + '/algorithm_specs/phase1', R + '/algorithm_specs/phase2', HERE]
from onnx_min import node, run, T
import spec_cross_entropy as ce, spec_base_ops as bo, spec_normalization as sn, spec_attention as at, spec_index_scatter as ix
import oracle_rounding as orc
from derive import _lo, _hi, _frac

rng = np.random.default_rng(20261015)
ROWS, UNAVAILABLE = [], {}

def bounds(s):
    if isinstance(s, F): return s, s
    if isinstance(s, (int,)): return F(s), F(s)
    return _frac(_lo(s)), _frac(_hi(s))
def rec(role, witness, case, w, s, note=''):
    lo, hi = bounds(s)
    w = float(w)
    if not np.isfinite(w):
        ROWS.append(dict(role=role, witness=witness, case=case, w=w, lo=float(lo), hi=float(hi), gap=float('inf'), note=note)); return
    wf = F(w)
    g = 0.0 if lo <= wf <= hi else float(min(abs(wf - lo), abs(wf - hi)) / max(F(1), abs(wf)))
    ROWS.append(dict(role=role, witness=witness, case=case, w=w, lo=float(lo), hi=float(hi), gap=g, note=note))
def ort(nodes, feeds, outs, opset, ir=None):
    ir = ir or (11 if opset >= 23 else 10 if opset >= 21 else 8)
    return run(nodes, feeds, outs, opset=opset, ir=ir)
def try_witness(key, fn):
    try: fn()
    except Exception as e: UNAVAILABLE[key] = repr(e)[:200]

# ------------------------------------------------------------- SOFTMAX / LOG_SOFTMAX / LOGSUMEXP
cases = [rng.normal(size=C) * s for C in (1, 2, 3, 7, 33, 128) for s in (1, 10, 100)]
cases += [np.array(v, dtype=float) for v in ([5, 5, 5], [1000, 0, -1000], [-800, -801, -802], [709, 709.5], [-1e4] * 4, [0, 1e-300, -1e-300])]
for ci, x in enumerate(cases):
    xf = [F(float(v)) for v in x]; C = len(x)
    sp, slp, slse = bo.softmax(xf), bo.log_softmax(xf), bo.logsumexp(xf)
    X = x.reshape(1, -1)
    def go():
        p = ort([node('Softmax', ['x'], ['y'], axis=-1)], {'x': X}, [('y', T['DOUBLE'], [1, C])], 13)[0][0]
        lp = ort([node('LogSoftmax', ['x'], ['y'], axis=-1)], {'x': X}, [('y', T['DOUBLE'], [1, C])], 13)[0][0]
        l = ort([node('ReduceLogSumExp', ['x'], ['y'], axes=[1], keepdims=0)], {'x': X}, [('y', T['DOUBLE'], [1])], 13)[0][0]
        for k in range(C): rec('SOFTMAX', 'ORT', ci, p[k], sp[k]); rec('LOG_SOFTMAX', 'ORT', ci, lp[k], slp[k])
        rec('LOGSUMEXP', 'ORT', ci, l, slse)
    try_witness('ORT softmax family', go)
    p, lp, l = ss.softmax(x), ss.log_softmax(x), ss.logsumexp(x)
    for k in range(C): rec('SOFTMAX', 'SciPy', ci, p[k], sp[k]); rec('LOG_SOFTMAX', 'SciPy', ci, lp[k], slp[k])
    rec('LOGSUMEXP', 'SciPy', ci, l, slse)

# ------------------------------------------------------------- CE (ONNX SoftmaxCrossEntropyLoss; no label smoothing in ONNX)
for ci in range(120):
    N, C = int(rng.integers(1, 6)), int(rng.integers(2, 9))
    X = rng.normal(size=(N, C)) * rng.choice([1, 5, 20])
    y = rng.integers(0, C, size=N); y[rng.random(N) < 0.3] = -100
    if ci % 15 == 0: y[:] = -100
    w = rng.uniform(0.1, 2, size=C) if rng.random() < 0.5 else None
    red = str(rng.choice(['none', 'sum', 'mean']))
    spec = ce.cross_entropy([[F(float(v)) for v in r] for r in X], [int(t) for t in y],
                            weights=None if w is None else [F(float(v)) for v in w], ignore_index=-100, reduction=red)
    def go():
        feeds = {'s': X, 'l': y.astype(np.int64)}; ins = ['s', 'l']
        if w is not None: feeds['w'] = w; ins.append('w')
        shape = [N] if red == 'none' else []
        out = ort([node('SoftmaxCrossEntropyLoss', ins, ['o'], reduction=red, ignore_index=-100)], feeds, [('o', T['DOUBLE'], shape)], 13)[0]
        cfg = f'N={N} C={C} red={red} w={w is not None} ignored={int((y == -100).sum())}'
        if spec['nan']:
            ROWS.append(dict(role='CE', witness='ORT', case=ci, w=float(out), lo=float('nan'), hi=float('nan'), gap=0.0 if np.isnan(out) else float('inf'),
                             note='spec: undefined (all ignored, mean); ' + cfg)); return
        if red == 'none':
            for n in range(N): rec('CE', 'ORT', ci, out[n], spec['loss'][n], cfg)
        else: rec('CE', 'ORT', ci, out, spec['loss'], cfg)
    try_witness('ORT SoftmaxCrossEntropyLoss', go)

# ------------------------------------------------------------- LayerNorm / RMSNorm / GroupNorm
for ci in range(18):
    C = [2, 7, 64][ci % 3]; eps = [1e-5, 1e-6, 0.0][(ci // 3) % 3]
    x, g, b = rng.normal(size=C) * rng.choice([1, 100]), rng.normal(size=C), rng.normal(size=C)
    eps32 = F(float(np.float32(eps)))                      # ONNX stores epsilon as float32: the spec must see the same scalar
    sl = sn.layer_norm([F(v) for v in x], [F(v) for v in g], [F(v) for v in b], eps32)
    sr = sn.rms_norm([F(v) for v in x], [F(v) for v in g], eps32)
    def go_ln():
        o = ort([node('LayerNormalization', ['x', 'g', 'b'], ['y'], axis=-1, epsilon=float(eps))], {'x': x.reshape(1, C), 'g': g, 'b': b},
                [('y', T['DOUBLE'], [1, C])], 17)[0][0]
        for k in range(C): rec('NORM_LN', 'ORT', ci, o[k], sl[k], f'C={C} eps={eps}')
    def go_rms():
        o = ort([node('RMSNormalization', ['x', 'g'], ['y'], axis=-1, epsilon=float(eps))], {'x': x.reshape(1, C), 'g': g},
                [('y', T['DOUBLE'], [1, C])], 23)[0][0]
        for k in range(C): rec('NORM_RMS', 'ORT', ci, o[k], sr[k], f'C={C} eps={eps}')
    try_witness('ORT LayerNormalization', go_ln); try_witness('ORT RMSNormalization', go_rms)
for ci in range(6):
    Nb, C, G, Hs = 2, 6, [1, 2, 3, 6, 2, 3][ci], 4
    x, g, b = rng.normal(size=(Nb, C, Hs)), rng.normal(size=C), rng.normal(size=C)
    eps32 = F(float(np.float32(1e-5)))
    def go():
        o = ort([node('GroupNormalization', ['x', 'g', 'b'], ['y'], num_groups=G, epsilon=1e-5)], {'x': x, 'g': g, 'b': b},
                [('y', T['DOUBLE'], [Nb, C, Hs])], 21)[0]
        for n in range(Nb):
            s = sn.group_norm([[F(v) for v in x[n, c]] for c in range(C)], G, [F(v) for v in g], [F(v) for v in b], eps32)
            for c in range(C):
                for h in range(Hs): rec('GROUPNORM', 'ORT', ci, o[n, c, h], s[c][h], f'G={G}')
    try_witness('ORT GroupNormalization', go)

# ------------------------------------------------------------- BatchNorm, training mode, running statistics
for ci in range(6):
    N, C, Hs = [2, 4, 8][ci % 3], 3, 2
    x = rng.normal(size=(N, C, Hs)); g, b = rng.normal(size=C), rng.normal(size=C)
    rm0, rv0 = rng.normal(size=C), rng.uniform(0.5, 2, size=C)
    mom_onnx = [0.9, 0.7][ci % 2]
    def go():
        Y, RM, RV = ort([node('BatchNormalization', ['x', 'g', 'b', 'm', 'v'], ['y', 'rm', 'rv'], epsilon=1e-5, momentum=mom_onnx, training_mode=1)],
                        {'x': x, 'g': g, 'b': b, 'm': rm0, 'v': rv0},
                        [('y', T['DOUBLE'], [N, C, Hs]), ('rm', T['DOUBLE'], [C]), ('rv', T['DOUBLE'], [C])], 15)
        st = sn.BNState(C); st.running_mean = [F(v) for v in rm0]; st.running_var = [F(v) for v in rv0]
        mom_spec = 1 - F(float(np.float32(mom_onnx)))      # ONNX: running = running*momentum + batch*(1-momentum)
        out = sn.batch_norm([[[F(v) for v in x[n, c]] for c in range(C)] for n in range(N)], st, [F(v) for v in g], [F(v) for v in b],
                            F(float(np.float32(1e-5))), momentum=mom_spec, training=True)
        for n in range(N):
            for c in range(C):
                for h in range(Hs): rec('BN-output', 'ORT', ci, Y[n, c, h], out[n][c][h])
        for c in range(C):
            rec('BN-running_mean', 'ORT', ci, RM[c], st.running_mean[c], f'momentum_onnx={mom_onnx}')
            vals = [F(v) for v in x[:, c, :].ravel()]; m = sum(vals) / len(vals)
            var_b = sum((v - m) ** 2 for v in vals) / len(vals)
            alt = (1 - mom_spec) * F(rv0[c]) + mom_spec * var_b     # same update with the BIASED batch variance
            rec('BN-running_var', 'ORT', ci, RV[c], st.running_var[c], f'N*H={N*Hs}; biased-variance alternative={float(alt):.17g}')
    try_witness('ORT BatchNormalization training', go)

# ------------------------------------------------------------- GELU
xs = np.concatenate([np.linspace(-8, 8, 33), [0.0, 1e-8, -1e-8, 3.0001, -0.7]])
for approx in ('none', 'tanh'):
    def go():
        dt, et = (np.float32, 'FLOAT') if approx == 'none' else (np.float64, 'DOUBLE')
        o = ort([node('Gelu', ['x'], ['y'], approximate=approx)], {'x': xs.astype(dt)}, [('y', T[et], [len(xs)])], 20)[0]
        for i, v in enumerate(xs):
            v = float(dt(v))
            s = bo.gelu_erf(F(v)) if approx == 'none' else bo.gelu_tanh(F(v))
            rec('ACT-gelu-' + approx, 'ORT-' + ('f32' if approx == 'none' else 'f64'), i, o[i], s)
    try_witness('ORT Gelu ' + approx, go)
    for i, v in enumerate(xs):
        s = bo.gelu_erf(F(float(v))) if approx == 'none' else bo.gelu_tanh(F(float(v)))
        wv = 0.5 * v * (1 + ss.erf(v / np.sqrt(2))) if approx == 'none' else 0.5 * v * (1 + np.tanh(np.sqrt(2 / np.pi) * (v + 0.044715 * v ** 3)))
        rec('ACT-gelu-' + approx, 'SciPy', i, wv, s)

# ------------------------------------------------------------- CAST (exact comparison; ONNX Cast with saturate, opset 21)
def grid_values(fmt):
    p, emin, emax, maxv, has_inf = orc.FORMATS[fmt]
    vals = set()
    for e in range(emin - p, emax + 1):                     # representable values and the midpoints between them
        for q in range(0, 2 ** (p - 1), max(1, 2 ** (p - 1) // 8)):
            for frac in (F(0), F(1, 2), F(1, 3)):
                v = (1 + F(q, 2 ** (p - 1)) + frac / 2 ** (p - 1)) * F(2) ** e
                vals.add(float(np.float32(float(v))))
    for v in (float(maxv), float(maxv) * 1.0625, float(maxv) * 1.5, float(maxv) * 4, 1e30):
        vals.add(float(np.float32(v)))
    vals |= {float(np.float32(v)) for v in rng.normal(size=400) * 10.0 ** rng.integers(-9, 6, size=400)}
    vals = sorted(v for v in vals if np.isfinite(v))
    return np.array(vals + [-v for v in vals], dtype=np.float32)
for fmt, onnx_t, sat_opts in (('f8e4m3fn', 'FLOAT8E4M3FN', (0, 1)), ('f8e5m2', 'FLOAT8E5M2', (0, 1)), ('bf16', 'BFLOAT16', (None,)), ('f16', 'FLOAT16', (None,))):
    vals = grid_values(fmt)
    for sat in sat_opts:
        def go():
            kw = dict(to=T[onnx_t]) if sat is None else dict(to=T[onnx_t], saturate=sat)
            o = ort([node('Cast', ['x'], ['c'], **kw), node('Cast', ['c'], ['y'], to=T['FLOAT'])], {'x': vals}, [('y', T['FLOAT'], [len(vals)])], 21)[0]
            for i, v in enumerate(vals):
                s = orc.round_to(F(float(v)), fmt, 'rne', saturate=bool(sat))
                w = float(o[i])
                if isinstance(s, str):
                    ok = (s == 'nan' and np.isnan(w)) or (s == 'inf' and w == np.inf) or (s == '-inf' and w == -np.inf)
                else:
                    ok = np.isfinite(w) and F(w) == s
                ROWS.append(dict(role=f'CAST-{fmt}' + ('' if sat is None else f'-sat{sat}'), witness='ORT', case=i, w=w,
                                 lo=float('nan') if isinstance(s, str) else float(s), hi=s if isinstance(s, str) else float(s),
                                 gap=0.0 if ok else float('inf'), note=f'x={float(v)!r}'))
        try_witness(f'ORT Cast {onnx_t} sat={sat}', go)
    if fmt == 'f16':                                         # NumPy float16 as a second, unrelated witness
        o = vals.astype(np.float16).astype(np.float32)
        for i, v in enumerate(vals):
            s = orc.round_to(F(float(v)), fmt, 'rne'); w = float(o[i])
            ok = (isinstance(s, str) and ((s == 'inf' and w == np.inf) or (s == '-inf' and w == -np.inf))) or (not isinstance(s, str) and np.isfinite(w) and F(w) == s)
            ROWS.append(dict(role='CAST-f16', witness='NumPy', case=i, w=w, lo=0, hi=0, gap=0.0 if ok else float('inf'), note=f'x={float(v)!r}'))

# ------------------------------------------------------------- Attention (ONNX Attention, opset 23): causal alignment, softcap, GQA
for ci, (Lq, Lk, causal, softcap, scale) in enumerate([(3, 3, 1, None, None), (2, 5, 1, None, None), (5, 2, 1, None, None), (2, 5, 0, None, None),
                                                      (3, 4, 0, 5.0, None), (3, 4, 1, 5.0, 0.5), (4, 4, 1, None, 0.25)]):
    Hq, Hkv, D = 2, 1, 4
    Q, K, V = rng.normal(size=(1, Hq, Lq, D)), rng.normal(size=(1, Hkv, Lk, D)), rng.normal(size=(1, Hkv, Lk, D))
    def go():
        kw = dict(is_causal=causal)
        if softcap is not None: kw['softcap'] = float(softcap)
        if scale is not None: kw['scale'] = float(scale)
        o = ort([node('Attention', ['q', 'k', 'v'], ['y'], **kw)], {'q': Q, 'k': K, 'v': V}, [('y', T['DOUBLE'], [1, Hq, Lq, D])], 23)[0]
        for h in range(Hq):
            hk = at.gqa_kv_head(h, Hq, Hkv)
            q = [[F(v) for v in r] for r in Q[0, h]]; k = [[F(v) for v in r] for r in K[0, hk]]; v = [[F(x) for x in r] for r in V[0, hk]]
            for align in ('upper_left', 'bottom_right'):
                try:
                    out, _ = at.attention(q, k, v, is_causal=bool(causal), scale=None if scale is None else F(float(np.float32(scale))),
                                          alignment=align, softcap=None if softcap is None else F(float(np.float32(softcap))), allow_empty_row='zero')
                except Exception as e:
                    continue
                for i in range(Lq):
                    for d in range(D): rec(f'ATTN-{align}', 'ORT', ci, o[0, h, i, d], out[i][d], f'Lq={Lq} Lk={Lk} causal={causal} softcap={softcap} scale={scale} h={h}')
    try_witness(f'ORT Attention case {ci}', go)

# ------------------------------------------------------------- RoPE (ONNX RotaryEmbedding, opset 23)
for ci, (interleaved, S, D) in enumerate([(0, 4, 8), (1, 4, 8), (0, 6, 16), (1, 6, 16)]):
    H = 1; X = rng.normal(size=(1, H, S, D)); pos = np.arange(S, dtype=np.int64).reshape(1, S)
    mp.mp.dps = 50
    cosc = np.array([[float(mp.cos(p * mp.power(10000, -mp.mpf(2 * i) / D))) for i in range(D // 2)] for p in range(S)])
    sinc = np.array([[float(mp.sin(p * mp.power(10000, -mp.mpf(2 * i) / D))) for i in range(D // 2)] for p in range(S)])
    def go():
        o = ort([node('RotaryEmbedding', ['x', 'c', 's', 'p'], ['y'], interleaved=interleaved)],
                {'x': X.astype(np.float32), 'c': cosc.astype(np.float32), 's': sinc.astype(np.float32), 'p': pos},
                [('y', T['FLOAT'], [1, H, S, D])], 23)[0]
        for layout in ('rotate_half', 'interleaved'):
            for s_ in range(S):
                sp = at.rope_apply([F(float(np.float32(v))) for v in X[0, 0, s_]], s_, D, layout=layout)
                for d in range(D): rec(f'ROPE-{layout}', 'ORT-f32', ci, o[0, 0, s_, d], sp[d], f'onnx interleaved={interleaved} S={S} D={D}')
    try_witness(f'ORT RotaryEmbedding case {ci}', go)

# ------------------------------------------------------------- scatter with reduction, gather, cumsum, matmul, variance
for ci in range(24):
    red_onnx, red_spec = [('add', 'sum'), ('mul', 'prod'), ('max', 'amax'), ('min', 'amin')][ci % 4]
    data = rng.integers(-4, 5, size=(4, 3)).astype(np.float64); idx = rng.integers(0, 4, size=(5, 3)).astype(np.int64)
    upd = rng.integers(-4, 5, size=(5, 3)).astype(np.float64)
    def go():
        o = ort([node('ScatterElements', ['d', 'i', 'u'], ['y'], axis=0, reduction=red_onnx)], {'d': data, 'i': idx, 'u': upd}, [('y', T['DOUBLE'], [4, 3])], 18)[0]
        s = ix.scatter_reduce([[F(v) for v in r] for r in data], 0, idx.tolist(), [[F(v) for v in r] for r in upd], red_spec, include_self=True)
        for r in range(4):
            for c in range(3): rec('INDEXREDUCE-scatter', 'ORT', ci, o[r, c], s[r][c], f'reduce={red_spec} include_self=True')
    try_witness('ORT ScatterElements', go)
for ci in range(8):
    inp = rng.normal(size=(3, 4)); idx = rng.integers(0, 4, size=(3, 2)).astype(np.int64)
    def go():
        o = ort([node('GatherElements', ['x', 'i'], ['y'], axis=1)], {'x': inp, 'i': idx}, [('y', T['DOUBLE'], [3, 2])], 13)[0]
        s = bo.gather([[F(v) for v in r] for r in inp], 1, idx.tolist())
        for r in range(3):
            for c in range(2): rec('GATHER', 'ORT', ci, o[r, c], s[r][c])
    try_witness('ORT GatherElements', go)
for ci in range(8):
    x = rng.normal(size=50) * 10.0 ** int(rng.integers(-3, 4))
    def go():
        o = ort([node('CumSum', ['x', 'a'], ['y'])], {'x': x, 'a': np.array(0, dtype=np.int64)}, [('y', T['DOUBLE'], [50])], 14)[0]
        s = bo.cumsum([F(v) for v in x])
        for i in range(50): rec('CUMSUM', 'ORT', ci, o[i], s[i])
    try_witness('ORT CumSum', go)
for ci in range(6):
    A, B = rng.normal(size=(3, 5)), rng.normal(size=(5, 2))
    def go():
        o = ort([node('MatMul', ['a', 'b'], ['y'])], {'a': A, 'b': B}, [('y', T['DOUBLE'], [3, 2])], 13)[0]
        s = bo.matmul2([[F(v) for v in r] for r in A], [[F(v) for v in r] for r in B])
        for r in range(3):
            for c in range(2): rec('MATMUL', 'ORT', ci, o[r, c], s[r][c])
    try_witness('ORT MatMul', go)
for ci in range(12):
    x = rng.normal(size=int(rng.integers(2, 30))); corr = ci % 3
    if corr >= len(x): continue
    rec('REDUCE-var', 'NumPy', ci, np.var(x, ddof=corr), bo.rvar([F(v) for v in x], correction=corr), f'correction={corr}')

json.dump(dict(rows=ROWS, unavailable=UNAVAILABLE), open(os.path.join(HERE, 'battery_rows.json'), 'w'), default=str)
print('rows', len(ROWS)); print('unavailable:', json.dumps(UNAVAILABLE, indent=1))
