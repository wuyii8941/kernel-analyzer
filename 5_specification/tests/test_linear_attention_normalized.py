import sys,unittest
from pathlib import Path
from fractions import Fraction as Q
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import linear_attention as L,d2_common as U
from kernel_spec.num import Invalid,Undefined

def plain(result):return {key:U.plain(value) for key,value in result.items()}

def components(args,scale,epsilon):
    q,k,v,s,z=args
    numerator=scale*(q@s+np.tril(q@k.T)@v)
    denominators=scale*np.sum(q*(z+np.cumsum(k,axis=0)),axis=1)+epsilon
    return numerator,denominators,s+k.T@v,z+np.sum(k,axis=0)

def exact_derivatives(args,scale,epsilon,go,gs,gz):
    n,d,_,_=components(args,scale,epsilon)
    result=[]
    for a,operand in enumerate(args):
        grad=np.empty(operand.shape,dtype=object)
        for i in np.ndindex(operand.shape):
            plus=list(args);minus=list(args);plus[a]=operand.copy();minus[a]=operand.copy()
            plus[a][i]+=1;minus[a][i]-=1
            pn,pd,ps,pz=components(plus,scale,epsilon);mn,md,ms,mz=components(minus,scale,epsilon)
            dn,dd,ds,dz=(pn-mn)/2,(pd-md)/2,(ps-ms)/2,(pz-mz)/2
            # Each component is affine in this one coordinate; the centered
            # differences are exact. Apply the quotient rule only at base d.
            grad[i]=sum((go[t,j]*(dn[t,j]*d[t]-n[t,j]*dd[t])/(d[t]*d[t])
                         for t,j in np.ndindex(n.shape)),Q(0))+np.sum(gs*ds)+np.sum(gz*dz)
        result.append(grad)
    return result

class NormalizedLinearAttention(unittest.TestCase):
    def test_initial_states_and_both_final_state_gradients(self):
        args=([[1]],[[1]],[[3]])
        self.assertEqual(plain(L.normalized(*args,epsilon=0,S0=[[2]],Z0=[4])),
                         {'out':[[1]],'state':[[5]],'zstate':[5]})
        grad=plain(L.normalized_backward(*args,[[1]],epsilon=0,S0=[[2]],Z0=[4],dST=[[2]],dZT=[3]))
        self.assertEqual(grad,{'dq':[[0]],'dk':[[Q(47,5)]],'dv':[[Q(11,5)]],'dS0':[[Q(11,5)]],'dZ0':[Q(14,5)]})
        self.assertEqual(plain(L.normalized(*args,epsilon=1,S0=[[2]],Z0=[4]))['out'],[[Q(5,6)]])
        with self.assertRaises(TypeError):L.normalized(*args)

    def test_complete_outputs_and_gradients_against_exact_global_formula(self):
        def values(shape,offset):
            out=np.empty(shape,dtype=object)
            for n,i in enumerate(np.ndindex(shape)):out[i]=Q((n+offset)%3+1,2)
            return out
        for T in (1,2):
            for K in (1,4):
                for V in (1,2):
                    for epsilon in (Q(0),Q(1,2)):
                        args=[values((T,K),0),values((T,K),1),values((T,V),2),values((K,V),1),values((K,),0)]
                        go,gs,gz=values((T,V),0),values((K,V),2),values((K,),1)
                        scale=Q(1,1 if K==1 else 2)
                        n,d,s,z=components(args,scale,epsilon)
                        kw=dict(epsilon=epsilon,S0=args[3].tolist(),Z0=args[4].tolist())
                        call=[x.tolist() for x in args[:3]]
                        expected={'out':(n/d[:,None]).tolist(),'state':s.tolist(),'zstate':z.tolist()}
                        for size in (None,1,3):
                            self.assertEqual(plain(L.normalized(*call,chunk_size=size,**kw)),expected)
                        actual=plain(L.normalized_backward(*call,go.tolist(),dST=gs.tolist(),dZT=gz.tolist(),**kw))
                        gradients=exact_derivatives(args,scale,epsilon,go,gs,gz)
                        self.assertEqual(set(actual),{'dq','dk','dv','dS0','dZ0'})
                        for key,target in zip(('dq','dk','dv','dS0','dZ0'),gradients):self.assertEqual(actual[key],target.tolist())

    def test_irrational_scale_encloses_the_known_scale_cancelling_answer(self):
        args=([[1,0,0]],[[1,0,0]],[[3]])
        kw=dict(epsilon=0,S0=[[2],[0],[0]],Z0=[4,0,0])
        output=L.normalized(*args,**kw)['out'][0][0]
        lo,hi=output.bounds();self.assertLessEqual(lo,1);self.assertGreaterEqual(hi,1)
        grad=L.normalized_backward(*args,[[1]],dST=[[2],[0],[0]],dZT=[3,0,0],**kw)
        for value,expected in ((grad['dq'][0][0],0),(grad['dk'][0][0],Q(47,5)),
                               (grad['dv'][0][0],Q(11,5)),(grad['dS0'][0][0],Q(11,5)),(grad['dZ0'][0],Q(14,5))):
            lo,hi=value.bounds();self.assertLessEqual(lo,expected);self.assertGreaterEqual(hi,expected)

    def test_empty_sequence_retains_both_states_and_cotangents(self):
        self.assertEqual(plain(L.normalized([],[],[],epsilon=0,S0=[[2,3]],Z0=[4])),
                         {'out':[],'state':[[2,3]],'zstate':[4]})
        self.assertEqual(plain(L.normalized_backward([],[],[],[],epsilon=0,S0=[[2,3]],Z0=[4],dST=[[5,6]],dZT=[7])),
                         {'dq':[],'dk':[],'dv':[],'dS0':[[5,6]],'dZ0':[7]})

    def test_zero_denominator_and_incomplete_state_are_not_hidden(self):
        with self.assertRaises(Undefined):L.normalized([[0]],[[0]],[[1]],epsilon=0)
        self.assertEqual(plain(L.normalized([[0]],[[0]],[[1]],epsilon=1))['out'],[[0]])
        for kw in ({'Z0':[1,2]},{'epsilon':-1}):
            with self.assertRaises(Invalid):L.normalized([[1]],[[1]],[[1]],**({'epsilon':0}|kw))
        with self.assertRaises(Invalid):L.normalized_backward([[1]],[[1]],[[1]],[[1]],epsilon=1,dZT=[1,2])

if __name__=='__main__':unittest.main()
