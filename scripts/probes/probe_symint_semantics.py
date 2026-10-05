#!/usr/bin/env python3
"""Probe: Python integer / float semantics on symbolic sizes.  Under dynamic shapes x.shape[0] is a SymInt; every
Python operation on it must give what the same operation gives on a Python int.  Each expression is compiled once
(dynamic=True) and called with several sizes; the value it produces is compared with plain Python.  Known instance:
pytorch#198064 (round(n, -k) is a no-op on SymInt).

    python scripts/probes/probe_symint_semantics.py
"""
import math

import torch

EXPRS = {
    "n // 3": lambda n: n // 3,
    "-n // 3": lambda n: -n // 3,
    "n // -3": lambda n: n // -3,
    "(n - 7) // 4": lambda n: (n - 7) // 4,
    "n % 3": lambda n: n % 3,
    "-n % 3": lambda n: -n % 3,
    "n % -3": lambda n: n % -3,
    "(n - 7) % 4": lambda n: (n - 7) % 4,
    "divmod(n - 7, 4)[0]": lambda n: divmod(n - 7, 4)[0],
    "divmod(n - 7, 4)[1]": lambda n: divmod(n - 7, 4)[1],
    "round(n / 2)": lambda n: round(n / 2),
    "round(n / 3)": lambda n: round(n / 3),
    "round(n, -1)  (known #198064)": lambda n: round(n, -1),
    "round(n * 1.5)": lambda n: round(n * 1.5),
    "math.floor(n / 3)": lambda n: math.floor(n / 3),
    "math.ceil(n / 3)": lambda n: math.ceil(n / 3),
    "math.trunc(-n / 3)": lambda n: math.trunc(-n / 3),
    "int(-n / 3)": lambda n: int(-n / 3),
    "int(n ** 0.5)": lambda n: int(n ** 0.5),
    "math.isqrt(n)": lambda n: math.isqrt(n),
    "int(math.sqrt(n))": lambda n: int(math.sqrt(n)),
    "int(math.log2(n))": lambda n: int(math.log2(n)),
    "n.bit_length()": lambda n: n.bit_length(),
    "n & 7": lambda n: n & 7,
    "n | 1": lambda n: n | 1,
    "n ^ 3": lambda n: n ^ 3,
    "n << 2": lambda n: n << 2,
    "n >> 1": lambda n: n >> 1,
    "-n >> 1": lambda n: -n >> 1,
    "abs(n - 10)": lambda n: abs(n - 10),
    "max(n, 7)": lambda n: max(n, 7),
    "min(n, 7) * 2": lambda n: min(n, 7) * 2,
    "pow(n, 2, 5)": lambda n: pow(n, 2, 5),
    "n ** 2 % 7": lambda n: n ** 2 % 7,
    "(n + 2) // 3 * 3": lambda n: (n + 2) // 3 * 3,
    "math.gcd(n, 12)": lambda n: math.gcd(n, 12),
    "len(range(0, n, 3))": lambda n: len(range(0, n, 3)),
    "float(n) / 7 (float)": lambda n: n / 7,
    "math.exp(n / 10) (float)": lambda n: math.exp(n / 10),
    "n * 0.1 // 0.3 (float floordiv)": lambda n: n * 0.1 // 0.3,
    "(-n) * 0.5 % 1.5 (float mod)": lambda n: (-n) * 0.5 % 1.5,
}
SIZES = [2, 3, 5, 7, 8, 13, 17, 40, 63, 100]


def main():
    bad = 0
    for name, expr in EXPRS.items():
        torch._dynamo.reset()

        def fn(x, expr=expr):
            return x.new_full((1,), float(expr(x.shape[0])), dtype=torch.float64)

        cf = torch.compile(fn, dynamic=True, backend="eager")
        wrong = []
        for n in SIZES:
            x = torch.zeros(n)
            try:
                got = cf(x).item()
            except Exception as e:  # noqa: BLE001
                wrong.append(f"n={n}: {type(e).__name__}: {str(e).splitlines()[0][:60]}")
                continue
            want = float(expr(n))
            if not (got == want or (math.isnan(got) and math.isnan(want)) or abs(got - want) <= 1e-12 * abs(want)):
                wrong.append(f"n={n}: {got} vs {want}")
        bad += bool(wrong)
        print(f"{name:34s} {'ok' if not wrong else 'WRONG: ' + '; '.join(wrong[:4])}", flush=True)
    print(f"{len(EXPRS)} expressions, {bad} wrong")


if __name__ == "__main__":
    main()
