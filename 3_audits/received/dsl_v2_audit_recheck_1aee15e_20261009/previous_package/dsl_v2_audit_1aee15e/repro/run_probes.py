#!/usr/bin/env python3
"""CPU audit of isolated production rules and explicit mathematical counterexamples.

Not an end-to-end Triton/GPU execution. Default: run transcribed source excerpts
with minimal dependency fixtures. --repo /path checks each executable AST against
the supplied checkout. --native uses real production intervals for _scaled_dot
and real production functions (requires that checkout's import dependencies).
"""
from __future__ import annotations
import argparse, ast, collections, copy, hashlib, itertools, json, sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '1aee15e9df7a45f434699b7016056e6650fdae99'
ST_OK, ST_NAN, ST_PINF, ST_NINF, ST_UNDEF, ST_NE = range(6)
INT_WIDTH = {'i1':1,'i8':8,'i16':16,'i32':32,'i64':64}
FLOAT_ELEMS = {'f16','bf16','f32','f64'}
MAYBE = 2
_TRIGGER_PATH = None
class ProgramAbort(Exception): pass
@dataclass
class PtrType:
    pointee: str
@dataclass
class TV:
    kind: str
    elem: object
    lo: object
    hi: object = None
    base: object = None
    st: object = None
    cond: object = None
    reasons: frozenset = frozenset()
    d: object = None
    def __post_init__(self):
        self.lo=np.asarray(self.lo)
        if self.hi is not None:self.hi=np.asarray(self.hi)
        if self.st is None:self.st=np.zeros(self.lo.shape,dtype=np.int8)
        if self.cond is None:self.cond=np.zeros(self.lo.shape,dtype=bool)
    @property
    def shape(self):return self.lo.shape

def _merge_status(*vals):
    st=np.zeros(vals[0].shape,dtype=np.int8)
    for v in vals:
        bad=np.broadcast_to(v.st,st.shape)
        st=np.where(bad>=ST_UNDEF,np.maximum(st,bad),st)
    return st

def _merge_cond(*vals):
    out=np.zeros(vals[0].shape,dtype=bool)
    for v in vals:out=out|np.broadcast_to(v.cond,out.shape)
    return out

def _merge_reasons(*vals):
    out=frozenset()
    for v in vals:out=out|v.reasons
    return out

def _ftv(elem,lo,hi,st,cond,reasons):
    return TV('f',elem,np.asarray(lo,np.float64),np.asarray(hi,np.float64),None,
              np.asarray(st,np.int8),np.asarray(cond,bool),reasons)

class ExactFixtureArithmetic:
    """Exact for the integer/dyadic fixtures here; NOT the repository's intervals.

    Also records the interval actually handed to the dot primitive, so loss of
    the high endpoint can be established without assuming a production bound.
    """
    seen=[]
    @staticmethod
    def idot(al,ah,bl,bh):
        ExactFixtureArithmetic.seen.append((al.copy(),ah.copy(),bl.copy(),bh.copy()))
        assert np.array_equal(al,ah) and np.array_equal(bl,bh), 'fixture expects point primitive arguments'
        ans=np.empty((al.shape[0],bl.shape[1]),float)
        for i in range(al.shape[0]):
            for j in range(bl.shape[1]):
                q=sum((Fraction.from_float(float(al[i,k]))*Fraction.from_float(float(bl[k,j]))
                       for k in range(al.shape[1])), Fraction(0))
                f=float(q);assert Fraction.from_float(f)==q
                ans[i,j]=f
        return ans,ans.copy()
    @staticmethod
    def iadd(al,ah,bl,bh):
        return np.asarray(al)+np.asarray(bl),np.asarray(ah)+np.asarray(bh)

iv=ExactFixtureArithmetic


def normalize(node):
    node=copy.deepcopy(node)
    # Qualifying class vs extracted top-level function is immaterial. Docstrings
    # and source locations are excluded; all executable AST nodes are compared.
    if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
        node.body=node.body[1:]
    return ast.dump(node,include_attributes=False)


def load_functions(repo=None,native=False):
    global TV,PtrType,_ftv
    src=(ROOT/'repro/production_excerpts.py').read_text()
    tree=ast.parse(src)
    names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    verified={}
    if repo:
        path=Path(repo)/'src/kernel_analyzer/reference_eval/ttir_eval.py'
        actual=ast.parse(path.read_text())
        funcs={n.name:n for n in ast.walk(actual) if isinstance(n,ast.FunctionDef) and n.name in names}
        for n in tree.body:
            if isinstance(n,ast.FunctionDef):
                verified[n.name]=n.name in funcs and normalize(n)==normalize(funcs[n.name])
        if not all(verified.values()):raise AssertionError(f'Executable AST differs: {verified}')
    env=dict(globals())
    exec(compile('from __future__ import annotations\n'+src,'production_excerpts.py','exec'),env)
    if native:
        if not repo:raise ValueError('--native requires --repo')
        sys.path.insert(0,str(Path(repo)/'src'))
        from kernel_analyzer.reference_eval import ttir_eval as native_mod
        from kernel_analyzer.reference_eval.ttir_parser import PtrType as NativePtr
        TV=native_mod.TV;PtrType=NativePtr;_ftv=native_mod._ftv
        for name in names:
            env[name]=getattr(native_mod,name,None) or getattr(native_mod.KernelReferenceEvaluator,name)
    return env, verified


def int_value(v,bits=64):
    signed=((int(v)+(1<<(bits-1)))%(1<<bits))-(1<<(bits-1))
    return TV('i',f'i{bits}',np.array([signed],np.int64))

def float_value(lo,hi=None,elem='bf16',conditional=False):
    lo=np.asarray(lo,float);hi=lo.copy() if hi is None else np.asarray(hi,float)
    return TV('f',elem,lo,hi,None,np.zeros(lo.shape,np.int8),np.full(lo.shape,conditional,bool))

def cmp_probe(env):
    results=[]
    for bits,a,b,pred in [(64,2**63,0,'ugt'),(64,2**64-1,1,'ult'),(64,2**63-1,0,'ugt'),(32,2**31,0,'ugt')]:
        op=NS(attrs={'predicate':pred})
        out=env['_cmpi'](op,[int_value(a,bits),int_value(b,bits)])
        expected=int(a>b) if pred=='ugt' else int(a<b)
        results.append(dict(bits=bits,a=str(a),b=str(b),predicate=pred,expected=expected,
                            observed=int(out.lo[0]),status=int(out.st[0]),matches=int(out.lo[0])==expected))
    return results


def cas_probe(env):
    rows=[]
    for m,c,v in [(2**53,2**53+1,7),(2**53+1,0,7),(15,16,7),(15,15,7)]:
        buf=NS(ident=1,kind='i',elem='i64',lo=np.array([m],np.int64),hi=None,
               st=np.array([ST_OK],np.int8),writer=np.array([-1],np.int64),written=np.array([False]))
        ctx=NS(_addresses=lambda *a:(buf,np.array([0]),np.array([True])),
               _readers={},_cas_serial=False,_rules=collections.Counter(),_race_programs=set(),
               _check_read_then_write=lambda *a:None,_track_read=lambda *a:None,
               _mark_plain=lambda *a:None,_wepoch={},_stored=set())
        ptr=TV('p',PtrType('i64'),np.array([0],np.int64),base=np.array([1],np.int64))
        out=env['_op_atomic_cas'](ctx,NS(node_id='cas_probe',attrs={}),[ptr,int_value(c),int_value(v)],{},NS(pid_index=0,rel_epoch=0))
        expected_new=v if m==c else m
        rows.append(dict(initial=str(m),compare=str(c),replacement=v,
                         expected_new=str(expected_new),actual_new=str(int(buf.lo[0])),
                         expected_return=str(m),actual_return=str(int(out.lo[0])),
                         result_status=int(out.st[0]),memory_status=int(buf.st[0]),
                         matches=(int(buf.lo[0])==expected_new and int(out.lo[0])==m)))
    return rows


def scaled_probe(env,native):
    # a admits either 1 or 2; each row contracts 32 elements with ones.
    a=float_value(np.ones((1,32)),np.full((1,32),2.0))
    b=float_value(np.ones((32,1)))
    c=float_value([[0.]],elem='f32')
    s=int_value(127,8);s.lo=np.array([[127]],np.int64);s.st=np.zeros((1,1),np.int8);s.cond=np.zeros((1,1),bool)
    op=NS(node_id='scaled_dot_probe');ctx=NS(_rules=collections.Counter())
    out=env['_scaled_dot'](ctx,op,a,s,b,s,c,'bf16','bf16')
    wide=dict(target_set_lower=32.,target_set_upper=64.,observed_lower=float(out.lo[0,0]),
              observed_upper=float(out.hi[0,0]),status=int(out.st[0,0]),
              covers_target_upper=bool(out.lo[0,0]<=64<=out.hi[0,0]),
              backend='production' if native else 'independent exact fixture backend; not production intervals')
    if not native:
        seen=ExactFixtureArithmetic.seen[-1]
        wide['high_endpoint_received_by_dot']=float(seen[1][0,0]);wide['true_input_high']=2.
    s.cond[:]=True
    a_point=float_value(np.ones((1,32)))
    out2=env['_scaled_dot'](ctx,op,a_point,s,b,None,c,'bf16','bf16')
    return {'interval_propagation':wide,
            'scale_dependency':dict(scale_conditional=True,output_conditional=bool(out2.cond[0,0]),status=int(out2.st[0,0]))}


class MiniBuffer:
    def __init__(self,x,written=True):
        self.lo=np.array([x],np.int64);self.hi=None;self.st=np.array([ST_OK],np.int8)
        self.kind='i';self.written=np.array([written]);self.cond=np.array([False])
    def copy(self):return copy.deepcopy(self)


def order_probe(env):
    def exact(order):
        x=0
        for i in order:x=2*x if i==1 else x+1
        return x
    all_results=[{'order':list(o),'value':exact(o)} for o in itertools.permutations(range(3))]
    ctx=NS(_rules=collections.Counter())
    mem={1:MiniBuffer(exact([0,1,2]))};snapshot={1:MiniBuffer(0,False)}
    def run(programs,grid,bindings,memory,order):
        memory[1]=MiniBuffer(exact(order));return {},collections.Counter()
    ctx._run_programs=run
    _,reasons=env['_cas_reverse_order'](ctx,[(0,),(1,),(2,)],(3,1,1),{},mem,snapshot,{},collections.Counter())
    return dict(all_serializations=all_results,all_results=sorted(set(x['value'] for x in all_results)),
                checked_orders=[[0,1,2],[2,1,0]],observed_result=int(mem[1].lo[0]),
                observed_status=int(mem[1].st[0]),observed_conditional=bool(mem[1].cond[0]),
                reasons=dict(reasons),scope='production agreement function with exact serial execution fixture; not GPU lock kernel')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo');ap.add_argument('--native',action='store_true')
    args=ap.parse_args();env,verified=load_functions(args.repo,args.native)
    out={'commit':COMMIT,'execution_scope':'CPU isolated functions; no GPU, no whole interpreter/run',
         'native_production_import':args.native,'source_ast_comparison':verified or 'not run: remote file not mounted',
         'numpy_version':np.__version__,
         'unsigned_compare':cmp_probe(env),'integer_cas':cas_probe(env),
         'scaled_dot':scaled_probe(env,args.native),'two_order_assumption':order_probe(env)}
    p=ROOT/'results'/('cpu_probes_native.json' if args.native else 'cpu_probes_excerpts.json')
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))
    # Exit status certifies only that the registered audit counterexamples
    # reproduced, not that the tool is correct.
    assert [r['matches'] for r in out['unsigned_compare']]==[False,False,True,True]
    assert [r['matches'] for r in out['integer_cas']]==[False,False,True,True]
    assert not out['scaled_dot']['interval_propagation']['covers_target_upper']
    assert not out['scaled_dot']['scale_dependency']['output_conditional']
    assert out['two_order_assumption']['all_results']==[2,3,4]
    assert out['two_order_assumption']['observed_result']==3
    assert out['two_order_assumption']['observed_status']==0
    print('Audit counterexamples reproduced; no production correctness certification.')
if __name__=='__main__':main()
