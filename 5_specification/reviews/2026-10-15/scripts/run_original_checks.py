from pathlib import Path
import subprocess,json,time,hashlib,sys
root=Path(__file__).resolve().parent
p=root/'input/kernel_spec_package_2026-10-15'
runs=[('relations',p/'specification_v2',[sys.executable,'test_relations.py']),('rounding',p/'specification_v2',[sys.executable,'test_oracle.py']),('phase1',p/'specification_v2/algorithm_specs/phase1',[sys.executable,'test_specs.py']),('phase2',p/'specification_v2/algorithm_specs/phase2',[sys.executable,'test_specs_phase2.py']),('phase3',p/'specification_v2/algorithm_specs/phase3',[sys.executable,'test_specs_phase3.py']),('contracts',p,[sys.executable,'contracts/contract_check.py']),('historical',p,[sys.executable,'evidence/historical_misreadings/meaning_relations.py']),('withdrawn_theory_checks',p,[sys.executable,'evidence/theory_checks/checks.py'])]
results=[]
for name,cwd,cmd in runs:
 t=time.monotonic()
 try:
  r=subprocess.run(cmd,cwd=cwd,text=True,capture_output=True,timeout=60)
  out=r.stdout+'\nSTDERR:\n'+r.stderr
  code=r.returncode
 except subprocess.TimeoutExpired as e:
  out=str(e);code='timeout'
 (root/'evidence'/f'original_{name}.txt').write_text(out)
 rec={'name':name,'command':cmd,'cwd':str(cwd.relative_to(p)),'returncode':code,'seconds':time.monotonic()-t}
 results.append(rec);print(json.dumps(rec),flush=True);print(out[-800:],flush=True)
(root/'evidence/original_checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
