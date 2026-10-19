"""Expected values written in contracts, parsed without eval.

An expected scalar is an integer, a 'p/q' string, or a formula over integers with + - * / **, unary minus and the
functions below. A formula without functions and with integer exponents is exact (a Fraction); anything else becomes
a rigorous enclosure computed with mpmath interval arithmetic at 256 bits. Booleans are discrete values. Any other
syntax is rejected.
"""
from __future__ import annotations

import ast
from fractions import Fraction

from mpmath import iv

from .num import IV, to_iv

FUNCTIONS = {
    "exp": iv.exp, "log": iv.log, "sqrt": iv.sqrt, "sin": iv.sin, "cos": iv.cos,
    "tanh": lambda x: 1 - 2 / (iv.exp(2 * x) + 1),     # x occurs once and the map is monotone: a tight enclosure
    "log1p": lambda x: iv.log(1 + x), "expm1": lambda x: iv.exp(x) - 1,
}
CONSTANTS = {"pi": lambda: iv.mpf(1) * iv.pi, "e": lambda: iv.exp(1)}


class ExpressionError(ValueError):
    pass


def iv_erf(x: IV) -> IV:
    """Rigorous enclosure of erf on an interval. erf is increasing, so it is enclosed at both endpoints; each endpoint
    value is computed by Arb (python-flint) ball arithmetic, which bounds its own error, and the ball's endpoints are
    read out exactly and re-enclosed outward. Endpoints of the input interval are exact binary numbers, converted to
    Arb exactly."""
    import flint
    from fractions import Fraction
    from .num import ends, to_iv
    saved = flint.ctx.prec
    flint.ctx.prec = iv.prec + 64
    try:
        lo_q, hi_q = ends(x)
        out = []
        for q, side in ((lo_q, 0), (hi_q, 1)):
            # int(): with gmpy2 installed mpmath (and so ends()) yields gmpy2.mpz, which Arb does not accept
            a = flint.arb(int(q.numerator)) / flint.arb(int(q.denominator))   # Arb encloses the quotient rigorously
            ball = a.erf()
            m, e = (ball.lower() if side == 0 else ball.upper()).man_exp()
            v = Fraction(int(m)) * (Fraction(2) ** int(e))
            out.append(v)
    finally:
        flint.ctx.prec = saved
    lo, hi = to_iv(max(out[0], Fraction(-1))), to_iv(min(out[1], Fraction(1)))
    return iv.mpf([lo.a, hi.b])


FUNCTIONS["erf"] = iv_erf


EXPECTED_PREC = 512


def parse(text):
    """Return a Fraction (exact), a bool, an infinity ("+inf" / "-inf"), or an IV enclosure. Formulas are enclosed at 512 bits, twice the precision of
    the references, so that a correct reference enclosure contains the expected enclosure."""
    saved = iv.prec
    iv.prec = EXPECTED_PREC
    try:
        return _parse(text)
    finally:
        iv.prec = saved


INFINITIES = {"inf": "+inf", "+inf": "+inf", "-inf": "-inf"}


def _parse(text):
    if isinstance(text, bool):
        return text
    if isinstance(text, str) and text.strip() in INFINITIES:
        return INFINITIES[text.strip()]
    if isinstance(text, int):
        return Fraction(text)
    if not isinstance(text, str):
        raise ExpressionError(f"expected value {text!r} is neither an integer, a boolean nor a formula string")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"cannot parse {text!r}: {e}") from None
    return _ev(tree.body, text)


def _ev(n, text):
    if isinstance(n, ast.Constant) and type(n.value) is int:
        return Fraction(n.value)
    if isinstance(n, ast.Name) and n.id in CONSTANTS:
        return CONSTANTS[n.id]()
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
        v = _ev(n.operand, text)
        return v if isinstance(n.op, ast.UAdd) else -v
    if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        a, b = _ev(n.left, text), _ev(n.right, text)
        if isinstance(n.op, ast.Pow):
            if isinstance(b, Fraction) and b.denominator == 1 and isinstance(a, Fraction):
                if a == 0 and b < 0:
                    raise ExpressionError(f"{text!r}: zero to a negative power")
                return a ** int(b)
            return to_iv(a) ** to_iv(b)
        if isinstance(a, Fraction) and isinstance(b, Fraction):
            if isinstance(n.op, ast.Div) and b == 0:
                raise ExpressionError(f"{text!r}: division by zero")
            return {ast.Add: a.__add__, ast.Sub: a.__sub__, ast.Mult: a.__mul__, ast.Div: a.__truediv__}[type(n.op)](b)
        a, b = to_iv(a), to_iv(b)
        return {ast.Add: a.__add__, ast.Sub: a.__sub__, ast.Mult: a.__mul__, ast.Div: a.__truediv__}[type(n.op)](b)
    if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCTIONS and len(n.args) == 1
            and not n.keywords):
        return FUNCTIONS[n.func.id](to_iv(_ev(n.args[0], text)))
    raise ExpressionError(f"{text!r}: unsupported syntax {ast.dump(n)[:80]}")
