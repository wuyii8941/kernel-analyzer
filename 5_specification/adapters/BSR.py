"""Adapter for contracts/BSR.yaml: torch.sparse._triton_ops bsr_dense_mm, bsr_softmax and sampled_addmm on a BSR matrix.

Input {op, bsr: {shape [M, N], blocksize [R, C], crow [M/R + 1], col [nnz], values [nnz][R][C]}, dense | mat1, mat2,
alpha, beta}. Output `out`: bsr_dense_mm -> the dense product [M][n]; bsr_softmax and sampled_addmm -> the values of the
result, [nnz][R][C], on the input's pattern. Unspecified elements are 0 for products and absent (-inf) for softmax.
READINGS['nonzero_pattern']: sampled_addmm with spy(input) read literally as the numerically non-zero entries.
MUTANTS: independent code below.
"""
from __future__ import annotations

from kernel_spec import d2_common as U
from kernel_spec import real as rl
from kernel_spec.num import Invalid, Result, Unsupported, integer, rational
from kernel_spec.real import R

KEYS = {"op", "bsr", "dense", "mat1", "mat2", "alpha", "beta"}
OPS = ("bsr_dense_mm", "bsr_softmax", "sampled_addmm")


def parse_bsr(b):
    if not isinstance(b, dict) or set(b) - {"shape", "blocksize", "crow", "col", "values"}:
        raise Invalid("bsr: {shape, blocksize, crow, col, values}")
    M, N = [integer(x, "shape", lo=0) for x in b["shape"]]
    Rb, Cb = [integer(x, "blocksize", lo=1) for x in b["blocksize"]]
    if M % Rb or N % Cb:
        raise Invalid("block size must divide the matrix shape")
    crow = [integer(x, "crow") for x in b["crow"]]
    col = [integer(x, "col") for x in b["col"]]
    nnz = len(col)
    if len(crow) != M // Rb + 1 or crow[0] != 0 or crow[-1] != nnz or any(y < x for x, y in zip(crow, crow[1:])):
        raise Invalid("crow must have nrowblocks + 1 non-decreasing entries from 0 to nnz")
    for br in range(M // Rb):
        cs = col[crow[br]:crow[br + 1]]
        if any(c < 0 or c >= N // Cb for c in cs) or any(y <= x for x, y in zip(cs, cs[1:])):
            raise Invalid("column block indices must be in range and increasing within a block row")
    if nnz:
        vals, _ = U.tensor(b["values"], 3, "values", shape=(nnz, Rb, Cb))
    else:
        if b["values"] not in ([], None):
            raise Invalid("values must be empty when nnz = 0")
        vals = []
    rows = [br for br in range(M // Rb) for _ in range(crow[br + 1] - crow[br])]
    return dict(M=M, N=N, R=Rb, C=Cb, crow=crow, col=col, vals=vals, rowblk=rows, nnz=nnz)


def parse(profile, inp):
    if profile != "torch_sparse":
        raise Unsupported(f"profile {profile!r}")
    U.check_keys(inp, KEYS)
    if inp.get("op") not in OPS:
        raise Invalid(f"op must be one of {OPS}")
    a = dict(op=inp["op"], b=parse_bsr(inp.get("bsr")))
    if a["op"] == "bsr_dense_mm":
        a["d"], (K, n) = U.tensor(inp["dense"], 2, "dense")
        if K != a["b"]["N"]:
            raise Invalid("dense.size(0) must equal the BSR column count")
        a["n"] = n
    elif a["op"] == "sampled_addmm":
        a["m1"], (m, k) = U.tensor(inp["mat1"], 2, "mat1")
        a["m2"], (k2, n) = U.tensor(inp["mat2"], 2, "mat2")
        if m != a["b"]["M"] or n != a["b"]["N"] or k != k2:
            raise Invalid("mat1 (m, k) @ mat2 (k, n) must match the input's shape (m, n)")
        a["k"] = k
        a["alpha"] = R(rational(inp.get("alpha", 1), "alpha"))
        a["beta"] = R(rational(inp.get("beta", 1), "beta"))
    return a


def to_dense(b):
    D = U.zeros(b["M"], b["N"])
    for e in range(b["nnz"]):
        r0, c0 = b["rowblk"][e] * b["R"], b["col"][e] * b["C"]
        for i in range(b["R"]):
            for j in range(b["C"]):
                D[r0 + i][c0 + j] = b["vals"][e][i][j]
    return D


def _mm_entry(A, B, i, j, k):
    return U.rsum(A[i][t] * B[t][j] for t in range(k))


def _evaluate(profile, inp, spy_nonzero=False):
    a = parse(profile, inp)
    b = a["b"]
    if a["op"] == "bsr_dense_mm":
        D = to_dense(b)
        out = [[_mm_entry(D, a["d"], i, j, b["N"]) for j in range(a["n"])] for i in range(b["M"])]
        return Result("value", {"out": U.plain(out)})
    if a["op"] == "bsr_softmax":
        out = [[[None] * b["C"] for _ in range(b["R"])] for _ in range(b["nnz"])]
        for br in range(b["M"] // b["R"]):
            es = list(range(b["crow"][br], b["crow"][br + 1]))
            for i in range(b["R"]):
                xs = [b["vals"][e][i][j] for e in es for j in range(b["C"])]
                if not xs:
                    continue
                m = max(xs, key=lambda x: x.v)
                ex = [rl.exp(x - m) for x in xs]
                z = U.rsum(ex)
                for t, (e, j) in enumerate([(e, j) for e in es for j in range(b["C"])]):
                    out[e][i][j] = ex[t] / z
        return Result("value", {"out": U.plain(out)})
    out = []
    for e in range(b["nnz"]):
        r0, c0 = b["rowblk"][e] * b["R"], b["col"][e] * b["C"]
        blk = []
        for i in range(b["R"]):
            row = []
            for j in range(b["C"]):
                v = b["vals"][e][i][j]
                prod = _mm_entry(a["m1"], a["m2"], r0 + i, c0 + j, a["k"]) if a["k"] else R(0)
                if spy_nonzero and v.sign() == 0:
                    prod = R(0)
                row.append(a["alpha"] * prod + a["beta"] * v)
            blk.append(row)
        out.append(blk)
    return Result("value", {"out": U.plain(out)})


def reference(profile, inp):
    return _evaluate(profile, inp)


READINGS = {"nonzero_pattern": lambda profile, inp: _evaluate(profile, inp, spy_nonzero=True)}


# ------------------------------------------------------------------ mutants (independent code)
def _mut(kind):
    def run(profile, inp):
        a = parse(profile, inp)
        b = a["b"]
        if a["op"] == "bsr_dense_mm":
            out = [[R(0)] * a["n"] for _ in range(b["M"])]
            for e in range(b["nnz"]):
                cb = (e - b["crow"][b["rowblk"][e]]) if kind == "col_ignored" else b["col"][e]
                for i in range(b["R"]):
                    for j in range(b["C"]):
                        for c in range(a["n"]):
                            out[b["rowblk"][e] * b["R"] + i][c] = (out[b["rowblk"][e] * b["R"] + i][c]
                                                                   + b["vals"][e][i][j] * a["d"][cb * b["C"] + j][c])
            return Result("value", {"out": U.plain(out)})
        if a["op"] == "bsr_softmax":
            D = None
            out = [[[None] * b["C"] for _ in range(b["R"])] for _ in range(b["nnz"])]
            if kind == "unspecified_as_zero":
                D = to_dense(b)
            for e in range(b["nnz"]):
                br = b["rowblk"][e]
                for i in range(b["R"]):
                    if kind == "per_block":
                        pool = [b["vals"][e][i][j] for j in range(b["C"])]
                    elif kind == "unspecified_as_zero":
                        pool = D[br * b["R"] + i]
                    else:
                        es = range(b["crow"][br], b["crow"][br + 1])
                        pool = [b["vals"][f][i][j] for f in es for j in range(b["C"])
                                if not (kind == "zeros_excluded" and b["vals"][f][i][j].sign() == 0)]
                    z = U.rsum(rl.exp(x) for x in pool)
                    for j in range(b["C"]):
                        x = b["vals"][e][i][j]
                        out[e][i][j] = R(0) if (kind == "zeros_excluded" and x.sign() == 0) else rl.exp(x) / z
            return Result("value", {"out": U.plain(out)})
        out = []
        for e in range(b["nnz"]):
            blk = []
            for i in range(b["R"]):
                row = []
                for j in range(b["C"]):
                    r, c = b["rowblk"][e] * b["R"] + i, b["col"][e] * b["C"] + j
                    if kind == "transposed":
                        r, c = c, r
                    p = U.rsum(a["m1"][r][t] * a["m2"][t][c] for t in range(a["k"]))
                    al = R(1) if kind == "alpha_ignored" else a["alpha"]
                    be = R(0) if kind == "beta_ignored" else a["beta"]
                    row.append(al * p + be * b["vals"][e][i][j])
                blk.append(row)
            out.append(blk)
        return Result("value", {"out": U.plain(out)})
    return run


MUTANTS = {
    "BS.M1": _mut("col_ignored"),
    "BS.M2": _mut("per_block"),
    "BS.M3": _mut("unspecified_as_zero"),
    "BS.M4": _mut("zeros_excluded"),
    "BS.M5": _mut("beta_ignored"),
    "BS.M6": _mut("alpha_ignored"),
    "BS.M7": _mut("transposed"),
}
