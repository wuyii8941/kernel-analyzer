import numpy as np, sympy as sp
from fractions import Fraction as F

print("== 1. GPT counterexample: f(x+y)=f(x)+f(y)+2xy, f(-x)=f(x) => f=x^2")
x, y = sp.symbols('x y', real=True)
f = sp.Function('f')
# step a: x=y=0 -> f(0)=2f(0)+0 -> f(0)=0
f0 = sp.symbols('f0')
print(" a) f(0) solves", sp.solve(sp.Eq(f0, 2*f0), f0))
# step b: y=-x: f(0)=f(x)+f(-x)-2x^2, with f(-x)=f(x), f(0)=0 -> f(x)=x^2
fx = sp.symbols('fx')
print(" b) f(x) =", sp.solve(sp.Eq(0, 2*fx - 2*x**2), fx))
# x^2 satisfies both relations
g = lambda t: t**2
print(" c) x^2 satisfies:", sp.simplify(g(x+y) - g(x) - g(y) - 2*x*y) == 0, sp.simplify(g(-x) - g(x)) == 0)

print("== 2. equivariance relations are closed under scaling: lambda*f passes every check f(Tx)=T'f(x)")
def softmax(v, beta=1.0):
    z = np.exp(beta*(v - v.max())); return z/z.sum()
rng = np.random.default_rng(0)
v = rng.normal(size=5); P = np.eye(5)[rng.permutation(5)]
K2 = lambda u: 2*softmax(u)
print(" 2*softmax: shift residual", np.abs(K2(v+3.7)-K2(v)).max(), " perm residual", np.abs(K2(P@v)-P@K2(v)).max(), " sum-to-one residual", K2(v).sum()-1)
Kb = lambda u: softmax(u, beta=2.0)
print(" softmax(2x): shift residual", np.abs(Kb(v+3.7)-Kb(v)).max(), " perm residual", np.abs(Kb(P@v)-P@Kb(v)).max(), " sum-to-one residual", Kb(v).sum()-1)

print("== 3. group averaging: mean bias along w under an exchangeable vs a skewed input distribution")
n, N = 5, 400000
X = rng.normal(size=(N, n))
def sm_rows(M, beta):
    Z = np.exp(beta*(M - M.max(1, keepdims=True))); return Z/Z.sum(1, keepdims=True)
E = sm_rows(X, 2.0) - sm_rows(X, 1.0)          # temperature error, lies in the blind set
w = np.array([1.0, 0, 0, 0, 0])
se = E[:,0].std()/np.sqrt(N)
print(f" exchangeable inputs: E<w,e> = {E[:,0].mean():+.5f}  (se {se:.5f}); prediction from relations alone: (1/n)E[1'e] + residual term = {E.sum(1).mean()/n:+.5f} + 0")
Xs = X.copy(); Xs[:,0] += 2.0                    # class 0 systematically larger, like a true label
Es = sm_rows(Xs, 2.0) - sm_rows(Xs, 1.0)
print(f" skewed inputs:       E<w,e> = {Es[:,0].mean():+.5f}  (se {Es[:,0].std()/np.sqrt(N):.5f}); relations still report zero residuals: sum {np.abs(Es.sum(1)).max():.1e}")

print("== 4. decomposition with K=3,G=2,f_alg=1,f_task=0: all three terms nonzero")
K,G,fa,ft = 3,2,1,0; print(" e_num, e_sem, e_design =", K-G, G-fa, fa-ft, " total", K-ft)

print("== 5. mean test vs elementwise tolerance: +-0.02 errors, tau=0.01")
e = np.array([0.02,-0.02]*500); print(" all |e_i|>tau:", np.all(np.abs(e)>0.01), " mean:", e.mean())
