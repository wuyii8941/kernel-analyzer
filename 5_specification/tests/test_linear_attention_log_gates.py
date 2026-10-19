import sys,unittest,math
from pathlib import Path
from fractions import Fraction as Q
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import linear_attention as L,d2_common as U
from kernel_spec.num import Invalid,Unsupported

def plain(result):return {k:U.plain(v) for k,v in result.items()}

def log2_bounds():
    # log(2)=2*atanh(1/3). Bound the positive tail by a geometric series.
    n=200;u=Q(1,3)
    lower=2*sum((u**(2*j+1)/Q(2*j+1) for j in range(n)),Q(0))
    return lower,lower+2*u**(2*n+1)/(Q(2*n+1)*(1-u*u))

class LogGates(unittest.TestCase):
    def contains(self,value,lo,hi=None):
        a,b=value.bounds();self.assertLessEqual(a,lo);self.assertGreaterEqual(b,lo if hi is None else hi)

    def test_declared_base_changes_forward_and_gradient(self):
        args=([[1]],[[1]],[[3]],[-1]);kw=dict(profile='scalar_decay_core',S0=[[2]])
        lnlo,lnhi=log2_bounds()
        for size in (None,1,3):
            result=L.decayed(*args,gate_parameterization='log2',chunk_size=size,**kw)
            self.contains(result['out'][0][0],4);self.contains(result['state'][0][0],4)
        grad=L.decayed_backward(*args,[[1]],dST=[[2]],gate_parameterization='log2',**kw)
        self.assertEqual(set(grad),{'dq','dk','dv','dS0','dg'})
        for key,value in (('dq',4),('dk',9),('dv',3),('dS0',Q(3,2))):self.contains(grad[key][0][0],value)
        self.contains(grad['dg'][0],3*lnlo,3*lnhi)
        # Alternating factorial series bounds exp(-1), independently of mpmath.
        upper=sum((Q((-1)**n,math.factorial(n)) for n in range(201)),Q(0))
        lower=upper-Q(1,math.factorial(201))
        natural=L.decayed(*args,gate_parameterization='ln',**kw)
        self.contains(natural['out'][0][0],3+2*lower,3+2*upper)
        self.assertLess(natural['out'][0][0].bounds()[1],result['out'][0][0].bounds()[0])
        grad=L.decayed_backward(*args,[[1]],dST=[[2]],gate_parameterization='ln',**kw)
        self.contains(grad['dg'][0],6*lower,6*upper)

    def test_diagonal_chain_rule_and_varlen_state_positions(self):
        args=([[1,2],[2,1]],[[1,1],[1,2]],[[3],[4]],[[0,-1],[-2,0]])
        kw=dict(profile='diagonal_decay_core',scale=1,S0_list=[[[2],[3]],[[7],[8]],[[4],[5]]])
        cu=[0,1,1,2];direct=[[1,Q(1,2)],[Q(1,4),1]]
        reference=L.decayed_varlen(*args[:3],direct,cu,**kw)
        result=L.decayed_varlen(*args,cu,gate_parameterization='log2',**kw)
        def enclosed(a,b):
            if isinstance(a,list):
                self.assertEqual(len(a),len(b))
                for x,y in zip(a,b):enclosed(x,y)
            else:self.contains(a,b.v)
        for key in reference:enclosed(result[key],reference[key])
        backward=dict(dST_list=[[[2],[3]],[[4],[5]],[[6],[7]]])
        ref=L.decayed_backward_varlen(*args[:3],direct,[[1],[2]],cu,**kw,**backward)
        result=L.decayed_backward_varlen(*args,[[1],[2]],cu,gate_parameterization='log2',**kw,**backward)
        for key in ('dq','dk','dv','dS0_list'):enclosed(result[key],ref[key])
        lo,hi=log2_bounds()
        for t in range(2):
            for i in range(2):
                factor=ref['da'][t][i].v*direct[t][i]
                self.contains(result['dg'][t][i],min(factor*lo,factor*hi),max(factor*lo,factor*hi))
        self.assertEqual(plain(result)['dS0_list'][1],[[4],[5]])

    def test_empty_log_gates_keep_state_and_have_empty_gradient(self):
        for profile in ('scalar_decay_core','diagonal_decay_core'):
            for base in ('ln','log2'):
                kw=dict(profile=profile,gate_parameterization=base,S0_list=[[[2]],[[3]]])
                self.assertEqual(plain(L.decayed_varlen([],[],[],[],[0,0,0],**kw)),{'out':[],'states':kw['S0_list']})
                self.assertEqual(plain(L.decayed_backward_varlen([],[],[],[],[],[0,0,0],dST_list=[[[4]],[[5]]],**kw)),
                                 {'dq':[],'dk':[],'dv':[],'dg':[],'dS0_list':[[[4]],[[5]]]})

    def test_unknown_base_and_bad_gradients_never_reach_exp(self):
        with self.assertRaises(Unsupported):
            L.decayed([],[],[],[],profile='scalar_decay_core',S0=[[1]],gate_parameterization='log10')
        with patch.object(L.rl,'exp',side_effect=AssertionError('computed before validation')):
            with self.assertRaises(Invalid):
                L.decayed_backward([[1]],[[1]],[[1]],[-1],[[1,2]],profile='scalar_decay_core',gate_parameterization='ln')
            with self.assertRaises(Invalid):
                L.decayed_backward_varlen([[1]],[[1]],[[1]],[-1],[[1]],[0,0,1],
                                         profile='scalar_decay_core',gate_parameterization='log2',dST_list=[[[1]]])

if __name__=='__main__':unittest.main()
