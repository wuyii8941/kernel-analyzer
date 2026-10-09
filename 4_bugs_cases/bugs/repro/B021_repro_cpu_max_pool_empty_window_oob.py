"""B021: eager CPU max_pool{1,2,3}d - for a legal window that samples only padding (dilation > 1, padding), the forward
returns an index past the end of the input plane and the CPU backward writes the window's gradient there: into the
next channel / sample, and past the end of the gradient buffer for the last ones (heap out-of-bounds write).

Independent check (no tool code): enumerate the window positions in plain Python; no window touches any input, so the
true gradient is exactly zero everywhere.  Each case runs in a child process because the out-of-bounds write can
corrupt the heap (set MALLOC_CHECK_=3 to make glibc report it).

    python bugs/repro/B021_repro_cpu_max_pool_empty_window_oob.py
"""
import subprocess
import sys

CHILD = r"""
import sys, torch, torch.nn.functional as F
nd, n, c, dev = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
pool = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}[nd]
x = torch.arange(1.0, n * c + 1).reshape((n, c) + (1,) * nd).to(dev).requires_grad_(True)
y, i = pool(x, 2, 1, 1, dilation=2, return_indices=True)
up = torch.arange(10.0, 10.0 + n * c).reshape(y.shape).to(dev)
y.backward(up)
print(f"torch {torch.__version__} {dev} max_pool{nd}d N={n} C={c}: out {y.flatten().tolist()} "
      f"index {i.flatten().tolist()} upstream {up.flatten().tolist()} grad {x.grad.flatten().tolist()}", flush=True)
"""


def windows_touch_input():
    """kernel 2, stride 1, padding 1, dilation 2 on a length-1 axis: one window, positions -1 and +1."""
    L, k, s, p, d = 1, 2, 1, 1, 2
    out = (L + 2 * p - d * (k - 1) - 1) // s + 1
    return [[o * s - p + d * m for m in range(k)] for o in range(out)], L


wins, L = windows_touch_input()
print("windows (positions sampled):", wins, "input positions: 0 ..", L - 1,
      "-> any window touching the input:", any(0 <= q < L for w in wins for q in w), "=> true gradient is 0")
devices = ["cpu"]
try:
    import torch
    if torch.cuda.is_available():
        devices.append("cuda")
except Exception:
    pass
for dev in devices:
    for nd, n, c in ((1, 2, 1), (1, 1, 4), (2, 1, 4), (3, 1, 4)):
        r = subprocess.run([sys.executable, "-c", CHILD, str(nd), str(n), str(c), dev], capture_output=True, text=True,
                           env={**__import__("os").environ, "MALLOC_CHECK_": "3"})
        lines = [l for l in (r.stdout + r.stderr).splitlines() if "Warning" not in l and l.strip()]
        print("\n".join(lines[-2:]), f"(exit {r.returncode})")
