from pathlib import Path
import sys, json, ast, contextlib,io,runpy,subprocess,tempfile,shutil,os
from fractions import Fraction as F
import yaml, mpmath as mp
root=Path(__file__).resolve().parent
p=root/'input/kernel_spec_package_2026-10-15'
sys.path.insert(0,str(p/'contracts/adapters'))
import common, ADAMW, CE, RANKMEAN
code=(p/'contracts/contract_check.py').read_text()
prefix=code.split('report, summary = [], []')[0]
ns={'__file__':str(p/'contracts/contract_check.py')}
exec(compile(prefix,str(p/'contracts/contract_check.py'),'exec'),ns)
judge=ns['judge']
mp.mp.dps=80
out={}
out['wrong_exact_value']=judge((mp.mpf(1)+mp.mpf('1e-51'),)*2,mp.mpf(1))
out['reversed_interval']=judge((mp.mpf(1)+mp.mpf('1e-52'),mp.mpf(1)-mp.mpf('1e-52')),mp.mpf(1))
rankcases=[{'S':[6,4],'n':[3.9,1]},{'S':[6,4],'n':[3,1,1]},{'S':[6,4],'n':[-1,2]},{'S':[6,4],'n':[0,1]}]
out['rank_domain']=[{'input':inp,'result':str(RANKMEAN.reference(inp))} for inp in rankcases]
base={'theta':1,'g':[2],'lr':'1/10','beta1':'9/10','beta2':'99/100','eps':0,'wd':'1/10'}
warm=dict(base,m='1/2',v=1,t=5)
out['adam_ignored_prestate']={'cold':str(ADAMW.reference(base)),'warm':str(ADAMW.reference(warm)),'same':ADAMW.reference(base)==ADAMW.reference(warm)}
ceis=[{'logits':[[0,1]],'target':[1],'reduction':'none'}, {'logits':[[0,1]],'target':[[F(1,4),F(3,4)]],'reduction':'sum'}]
res=[]
for c in ceis:
 try: r=CE.reference(c);res.append({'input':str(c),'result':str(r)})
 except Exception as e:res.append({'input':str(c),'exception':f'{type(e).__name__}: {e}'})
out['ce_declared_interface']=res
# CE gradient is in the declared signature but absent from all contract checks.
orig=CE.ce.cross_entropy
n_calls=0
def wrong_grad(*a,**kw):
 global n_calls
 r=orig(*a,**kw)
 if r.get('grad') is not None:
  n_calls+=1;r['grad']=[[type(v).exact(0) for v in row] for row in r['grad']]
 return r
CE.ce.cross_entropy=wrong_grad
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
 runpy.run_path(str(p/'contracts/contract_check.py'),run_name='__main__')
CE.ce.cross_entropy=orig
(root/'evidence/gradient_mutation_checker.txt').write_text(buf.getvalue())
out['zero_ce_gradient_still_passes']={'mutated_calls':n_calls,'ce_all_five_established':sum('| CE |' in l and l.endswith('| established |') for l in buf.getvalue().splitlines())==5}
# Work only on temporary copies for the checker/runner error propagation probes.
with tempfile.TemporaryDirectory() as td:
 cp=Path(td)/'p';shutil.copytree(p,cp)
 yp=cp/'contracts/CE.yaml'; c=yaml.safe_load(yp.read_text()); c['examples']['E1']['expected']='999';yp.write_text(yaml.safe_dump(c,sort_keys=False))
 r=subprocess.run([sys.executable,'contracts/contract_check.py'],cwd=cp,capture_output=True,text=True,timeout=30)
 out['contradiction_exit']={'returncode':r.returncode,'has_contradiction':'contradiction' in r.stdout}
 yp.write_text((p/'contracts/CE.yaml').read_text())
 # Unresolvable source id is accepted by original checker.
 c=yaml.safe_load(yp.read_text());c['clauses']['CE.C1']['basis']='MISSING.D0';yp.write_text(yaml.safe_dump(c,sort_keys=False))
 r=subprocess.run([sys.executable,'contracts/contract_check.py'],cwd=cp,capture_output=True,text=True,timeout=30)
 out['unresolved_source']={'returncode':r.returncode,'accepted':any('MISSING.D0' in line and line.endswith('| established |') for line in r.stdout.splitlines())}
 yp.write_text((p/'contracts/CE.yaml').read_text())
 c=yaml.safe_load(yp.read_text());c['examples']['E5']['readings']['R_B']='999';yp.write_text(yaml.safe_dump(c,sort_keys=False))
 r=subprocess.run([sys.executable,'contracts/contract_check.py'],cwd=cp,capture_output=True,text=True,timeout=30)
 out['wrong_reading']={'returncode':r.returncode,'reading_failed':"'E5[R_B]': 'contradiction'" in r.stdout,'ce_c4_still_established':any('CE.C4' in line and line.endswith('| established |') for line in r.stdout.splitlines())}
 yp.write_text((p/'contracts/CE.yaml').read_text())
 (cp/'specification_v2/test_relations.py').write_text('import sys\nprint("INTENTIONAL AUDIT TEST FAILURE",file=sys.stderr)\nsys.exit(9)\n')
 r=subprocess.run(['sh','run_all.sh'],cwd=cp,capture_output=True,text=True,timeout=40)
 out['masked_test_failure']={'returncode':r.returncode,'saw_injected_failure':'INTENTIONAL AUDIT TEST FAILURE' in r.stdout+r.stderr}
 (root/'evidence/injected_test_failure_runner.txt').write_text(r.stdout+'\nSTDERR\n'+r.stderr)
(root/'evidence/original_probes.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False,indent=2))
