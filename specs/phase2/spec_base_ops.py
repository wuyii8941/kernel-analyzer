"""Independent specification f for basic operators.  (v0.1)

Documentation (fetched 2026-10-08 unless marked):
  gelu    : https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html
            approximate='none': GELU(x) = x * Phi(x); approximate='tanh': 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715 x^3)))
  silu    : SiLU(x) = x * sigmoid(x)              (torch.nn.SiLU)
  matmul  : torch.matmul broadcasting rules (batch dims broadcast, 1-D promotion)      [doc text not re-fetched]
  linear  : y = x A^T + b                          (torch.nn.functional.linear)
  var/std : correction (default 1): divide by max(0, N - correction)   (torch.var)       [doc text not re-fetched]
  softmax : softmax(x)_i = exp(x_i) / sum_j exp(x_j); log_softmax; logsumexp
  cumsum  : y_i = sum_{j<=i} x_j
  gather  : out[i][j] = input[index[i][j]][j] for dim=0 (2-D); index_select picks slices along dim;
            take_along_dim = gather with broadcasting of index (1-D/2-D here)

Semantics are the documented real-valued formulas. Exact where only +,-,*,/,max,min occur; rigorous intervals for exp,
log, sqrt, tanh (via exp); erf-GELU uses a declared high-precision approximation (see rigorous_mp.erf_approx).

Ambiguities (ambiguities_phase2.md):
  BASE-A1 var/std with N - correction <= 0: undefined (torch documents NaN/inf results for such cases in some versions;
          the spec declines). amax/amin/prod/sum of an empty reduction: sum=0, prod=1 (identities, documented); amax/amin
          undefined (declines).
  BASE-A2 softmax of a row whose entries are all -inf: undefined (declines); logsumexp of such a row = -inf.
  BASE-A3 GELU with approximate='tanh' is a DIFFERENT declared formula from approximate='none'; a candidate declaring one
          is compared with that one only. The 0.044715 and sqrt(2/pi) constants are part of the documented formula.
  BASE-A4 matmul: 1-D operands are promoted per the documented rules; this spec supports 2-D x 2-D and batched 3-D.
"""
from fractions import Fraction as F
import math
import rigorous_mp as R
from rigorous_mp import I, iv, SpecInputError, SpecNotEstablished


# ------------------------------------------------------------------ matmul / linear (exact rationals)
def matmul2(A, B):
    if len(A[0]) != len(B):
        raise SpecInputError("inner dimensions differ")
    return [[sum((F(A[i][k]) * F(B[k][j]) for k in range(len(B))), F(0)) for j in range(len(B[0]))] for i in range(len(A))]


def transpose(A):
    return [list(r) for r in zip(*A)]


def linear(x, W, b=None):
    """y = x W^T + b ; x: [N][in], W: [out][in], b: [out]."""
    y = matmul2(x, transpose(W))
    if b is not None:
        if len(b) != len(W):
            raise SpecInputError("bias length must equal out_features")
        y = [[v + F(b[j]) for j, v in enumerate(row)] for row in y]
    return y


def linear_backward(x, W, grad_y, has_bias=True):
    """exact gradients: dX = gY W ; dW = gY^T x ; db = sum_rows gY."""
    dX = matmul2(grad_y, W)
    dW = matmul2(transpose(grad_y), x)
    db = [sum((F(grad_y[n][j]) for n in range(len(grad_y))), F(0)) for j in range(len(W))] if has_bias else None
    return dX, dW, db


def bmm(A, B):
    if len(A) != len(B):
        raise SpecInputError("batch sizes differ (this spec does not broadcast batch dims)")
    return [matmul2(a, b) for a, b in zip(A, B)]


# ------------------------------------------------------------------ reductions
def rsum(xs): return sum((F(v) for v in xs), F(0))
def rmean(xs):
    if not xs:
        raise SpecNotEstablished("mean of an empty reduction is undefined")
    return rsum(xs) / len(xs)
def rprod(xs): return math.prod((F(v) for v in xs), start=F(1))
def ramax(xs):
    if not xs:
        raise SpecNotEstablished("amax of an empty reduction is undefined")
    return max(F(v) for v in xs)
def ramin(xs):
    if not xs:
        raise SpecNotEstablished("amin of an empty reduction is undefined")
    return min(F(v) for v in xs)


def rvar(xs, correction=1):
    n = len(xs)
    if n - correction <= 0:
        raise SpecNotEstablished("variance with non-positive degrees of freedom is undefined (BASE-A1)")
    m = rmean(xs)
    return sum(((F(v) - m) ** 2 for v in xs), F(0)) / (n - correction)


def rstd(xs, correction=1):
    return iv.sqrt(I(rvar(xs, correction)))


def softmax(xs):
    finite = [v for v in xs if not (isinstance(v, float) and v == float("-inf"))]
    if not finite:
        raise SpecNotEstablished("softmax of an all -inf row is undefined (BASE-A2)")
    if len(finite) != len(xs):
        # -inf entries contribute exactly 0 probability
        p, _ = R.softmax_row(finite)
        out, k = [], 0
        for v in xs:
            if isinstance(v, float) and v == float("-inf"):
                out.append(iv.mpf(0))
            else:
                out.append(p[k]); k += 1
        return out
    return R.softmax_row(xs)[0]


def log_softmax(xs):
    return R.softmax_row(xs)[1]


def logsumexp(xs):
    finite = [v for v in xs if not (isinstance(v, float) and v == float("-inf"))]
    if not finite:
        return float("-inf")
    xs = [I(v) for v in finite]
    m = max(v.b for v in xs)
    return m + iv.log(R.isum(iv.exp(v - m) for v in xs))


def cumsum(xs):
    out, acc = [], F(0)
    for v in xs:
        acc += F(v); out.append(acc)
    return out


# ------------------------------------------------------------------ activations
def relu(x): return max(F(x), F(0))
def silu(x): return I(x) * R.sigmoid(x)
def gelu_tanh(x):
    x = F(x)
    inner = I(x + F(44715, 1000000) * x ** 3) * iv.sqrt(iv.mpf(2) / iv.pi)
    return I(x) * (1 + R.tanh(inner)) / 2
def gelu_erf(x):
    """declared approximation for erf (not a rigorous enclosure); Phi(x) = (1 + erf(x/sqrt2))/2."""
    x = I(x)
    return x * (1 + R.erf_approx(x / iv.sqrt(iv.mpf(2)))) / 2
def swiglu(a, b): return silu(a) * I(b)
def geglu(a, b, approximate="none"): return (gelu_tanh(a) if approximate == "tanh" else gelu_erf(a)) * I(b)


def relu_grad(x, upstream):
    """subgradient convention at 0 is NOT documented: the spec returns the set {0, upstream} at x == 0."""
    x = F(x)
    if x > 0: return {F(upstream)}
    if x < 0: return {F(0)}
    return {F(0), F(upstream)}


# ------------------------------------------------------------------ gather / index_select / take_along_dim
def gather(inp, dim, index):
    """2-D: out[i][j] = inp[index[i][j]][j] (dim 0) or inp[i][index[i][j]] (dim 1)."""
    R0, C0 = len(inp), len(inp[0])
    out = []
    for i, row in enumerate(index):
        o = []
        for j, k in enumerate(row):
            if dim == 0:
                if not (0 <= k < R0) or j >= C0:
                    raise SpecInputError("gather index out of range")
                o.append(F(inp[k][j]))
            elif dim == 1:
                if not (0 <= k < C0) or i >= R0:
                    raise SpecInputError("gather index out of range")
                o.append(F(inp[i][k]))
            else:
                raise SpecInputError("dim must be 0 or 1")
        out.append(o)
    return out


def index_select(inp, dim, index):
    if dim == 0:
        for k in index:
            if not (0 <= k < len(inp)):
                raise SpecInputError("index out of range")
        return [[F(v) for v in inp[k]] for k in index]
    if dim == 1:
        for k in index:
            if not (0 <= k < len(inp[0])):
                raise SpecInputError("index out of range")
        return [[F(row[k]) for k in index] for row in inp]
    raise SpecInputError("dim must be 0 or 1")


def gather_backward(inp_shape, dim, index, grad_out):
    """exact scatter-add of upstream into the gathered positions (duplicates accumulate)."""
    R0, C0 = inp_shape
    g = [[F(0)] * C0 for _ in range(R0)]
    for i, row in enumerate(index):
        for j, k in enumerate(row):
            if dim == 0:
                g[k][j] += F(grad_out[i][j])
            else:
                g[i][k] += F(grad_out[i][j])
    return g


# ------------------------------------------------------------------ reference-free properties
def prop_softmax_shift(xs, c):
    """softmax(x + c) == softmax(x) (intervals must overlap entrywise)."""
    return softmax(xs), softmax([F(v) + F(c) for v in xs])


def prop_linear_adjoint(x, W, grad_y, u_x, u_W):
    """<dX, u_x> + <dW, u_W> == <gY, J_x u_x + J_W u_W>, where J_x u_x = u_x W^T and J_W u_W = x u_W^T (exact)."""
    dX, dW, _ = linear_backward(x, W, grad_y, has_bias=False)
    lhs = sum(F(a) * F(b) for ra, rb in zip(dX, u_x) for a, b in zip(ra, rb)) + \
          sum(F(a) * F(b) for ra, rb in zip(dW, u_W) for a, b in zip(ra, rb))
    Ju = [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(matmul2(u_x, transpose(W)), matmul2(x, transpose(u_W)))]
    rhs = sum(F(a) * F(b) for ra, rb in zip(grad_y, Ju) for a, b in zip(ra, rb))
    return lhs, rhs


def prop_gather_adjoint(inp, dim, index, grad_out, u):
    """<gather(inp), g> == <inp, gather_backward(g)> is bilinear in (inp, g); check <gather(u), g> == <u, gbw(g)>."""
    lhs = sum(F(a) * F(b) for ra, rb in zip(gather(u, dim, index), grad_out) for a, b in zip(ra, rb))
    gb = gather_backward((len(inp), len(inp[0])), dim, index, grad_out)
    rhs = sum(F(a) * F(b) for ra, rb in zip(u, gb) for a, b in zip(ra, rb))
    return lhs, rhs
