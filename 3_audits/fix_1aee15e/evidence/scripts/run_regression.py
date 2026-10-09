"""Rerun of the v1.1 program regression (2_tool/scripts/dsl_v2/regression_v11.py, one worker process per program)
after the audit fixes.  The four race programs (T5: prog_01, prog_13, prog_22, prog_23) are not launched on the GPU
(task book: refused race kernels are never run around the permission system); their earlier results stay referenced."""
import os, subprocess, sys, time
from pathlib import Path
R = Path("/data1/tzh/kernel-analyzer")
run_id, workers = sys.argv[1], int(sys.argv[2])
RACE = {"prog_01", "prog_13", "prog_22", "prog_23"}
sys.path[:0] = [str(R / "2_tool/scripts/acceptance"), str(R / "2_tool/src"), str(R / "2_tool/scripts/general")]
import structure_v11 as S
man = S.manifest()
todo = [p for p, e in sorted(man.items()) if e["execution_lane"] == "measurement" and p not in RACE]
todo.sort(key=lambda p: -int(man[p].get("execution_repeats", 1)))
out = R / "1_experiments/dsl_v2/regression_v11" / run_id
logs = R / ".cache/audit_work/regression_logs" / run_id
logs.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
running, slot = {}, 0
print(len(todo), "programs", flush=True)
while todo or running:
    while todo and len(running) < workers:
        pid = todo.pop(0)
        if (out / f"{pid}.json").exists():
            continue
        e2 = dict(env, CUDA_VISIBLE_DEVICES=str(slot % 4))
        slot += 1
        log = open(logs / f"{pid}.log", "w")
        p = subprocess.Popen(["/data1/tzh/envs/ka_main/bin/python", str(R / "2_tool/scripts/dsl_v2/regression_v11.py"),
                              "--run-id", run_id, "--program", pid], cwd=R / "2_tool", env=e2, stdout=log,
                             stderr=subprocess.STDOUT)
        running[p.pid] = (p, pid, log, time.time())
    time.sleep(5)
    for k, (p, pid, log, t) in list(running.items()):
        if p.poll() is not None:
            log.close()
            print(f"{time.strftime('%H:%M:%S')} {pid} exit={p.returncode} {time.time() - t:.0f}s", flush=True)
            del running[k]
print("REGRESSION_DONE", flush=True)
