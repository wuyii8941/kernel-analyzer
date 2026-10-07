"""B022: Inductor (nightly / main) - the global scatter_add decomposition of max_pool2d_with_indices_backward scatters
each window's gradient to the index the forward stored; Inductor's own forward stores -1 for a window that has no real
maximum (it samples only padding), so the compiled backward hits the indirect-indexing bounds assert
("index out of bounds") and the CUDA context is lost.  2.10 (no such decomposition) gives the correct zero gradient.

Independent check (no tool code): the only window samples positions -1 and +1 of a length-1 axis, both padding, so no
input is in any window and the true gradient is 0.  Run each case in a child process (the assert kills the context).

    python bugs/repro/B022_repro_inductor_max_pool_backward_assert.py
"""
import subprocess
import sys

CHILD = r"""
import sys, torch, torch.nn.functional as F
ri = sys.argv[1] == "1"
x = torch.tensor([[[7.0]], [[8.0]]], device="cuda", requires_grad=True)      # batch 2, 1 channel, length 1
def f(x):
    r = F.max_pool1d(x, 2, 1, 1, dilation=2, return_indices=ri)
    return r[0] if ri else r
y = torch.compile(f, fullgraph=True)(x)
torch.cuda.synchronize()
print(f"torch {torch.__version__} return_indices={ri}: out {y.flatten().tolist()}", flush=True)
y.backward(torch.tensor([[[5.0]], [[7.0]]], device="cuda"))
torch.cuda.synchronize()
print(f"   grad {x.grad.flatten().tolist()} (expected [0, 0])", flush=True)
"""

for ri in ("1", "0"):
    r = subprocess.run([sys.executable, "-c", CHILD, ri], capture_output=True, text=True,
                       env={**__import__("os").environ, "CUDA_LAUNCH_BLOCKING": "1"})
    out = [l for l in (r.stdout + r.stderr).splitlines()
           if l.startswith(("torch", "   grad")) or "Assertion" in l or "Error" in l]
    seen, lines = set(), []
    for l in out:
        key = l.split("thread:")[0] if "Assertion" in l else l
        if key not in seen:
            seen.add(key)
            lines.append(l[:220])
    print("\n".join(lines[:4]), f"(exit {r.returncode})")
