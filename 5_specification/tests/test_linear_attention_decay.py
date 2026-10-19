import sys,unittest
from pathlib import Path
from fractions import Fraction as Q
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import linear_attention as L,d2_common as U
from kernel_spec.num import Invalid,Unsupported

def plain(result):return {key:U.plain(value) for key,value in result.items()}

def matrix_formula(args,scalar,scale):
    """Global products of augmented affine matrices; no reference recurrence."""
    q,k,v,s,a=args;T,K=q.shape;V=v.shape[1];D=K*V
    transforms=[]
    for t in range(T):
        gate=np.repeat(a[t],K) if scalar else a[t]
        m=np.zeros((D+1,D+1),dtype=object)
        m[:D,:D]=np.kron(np.diag(gate),np.eye(V,dtype=object))
        m[:D,D]=np.outer(k[t],v[t]).reshape(D);m[D,D]=Q(1)
        transforms.append(m)
    initial=np.concatenate((s.reshape(D),[Q(1)]))
    states=[]
    for t in range(T):
        product=np.eye(D+1,dtype=object)
        for m in reversed(transforms[:t+1]):product=product@m
        states.append((product@initial)[:D].reshape(K,V))
    out=np.stack([scale*q[t]@states[t] for t in range(T)])
    return out,states[-1]

class DecayedLinearAttention(unittest.TestCase):
    def test_zero_gate_chunking_and_final_gradient(self):
        args=([[1],[1]],[[1],[2]],[[3],[4]],[0,Q(1,2)])
        kw=dict(profile='scalar_decay_core',S0=[[7]])
        for size in (None,1,2,4):
            self.assertEqual(plain(L.decayed(*args,chunk_size=size,**kw)),
                             {'out':[[3],[Q(19,2)]],'state':[[Q(19,2)]]})
        result=plain(L.decayed_backward(*args,[[0],[0]],dST=[[1]],**kw))
        self.assertEqual(result,{'dq':[[0],[0]],'dk':[[Q(3,2)],[4]],'dv':[[Q(1,2)],[2]],
                                 'dS0':[[0]],'da':[Q(7,2),3]})

    def test_exact_affine_products_and_complete_derivatives(self):
        def values(shape,offset):
            out=np.empty(shape,dtype=object)
            for n,i in enumerate(np.ndindex(shape)):out[i]=Q((n+offset)%5-2,2)
            return out
        for scalar in (True,False):
            profile='scalar_decay_core' if scalar else 'diagonal_decay_core'
            for K in (1,4):
                T,V=3,2;scale=Q(1,1 if K==1 else 2)
                args=[values((T,K),0),values((T,K),1),values((T,V),2),values((K,V),3),values((T,) if scalar else (T,K),1)]
                go,gs=values((T,V),4),values((K,V),0)
                expected,state=matrix_formula(args,scalar,scale)
                call=[x.tolist() for x in args[:3]]+[args[4].tolist()]
                kw=dict(profile=profile,S0=args[3].tolist())
                for size in (None,1,2,5):
                    self.assertEqual(plain(L.decayed(*call,chunk_size=size,**kw)),
                                     {'out':expected.tolist(),'state':state.tolist()})
                actual=plain(L.decayed_backward(*call,go.tolist(),dST=gs.tolist(),**kw))
                keys=('dq','dk','dv','dS0','da');self.assertEqual(set(actual),set(keys))
                for ai,operand in enumerate(args):
                    grad=np.empty(operand.shape,dtype=object)
                    for idx in np.ndindex(operand.shape):
                        plus=list(args);minus=list(args);plus[ai]=operand.copy();minus[ai]=operand.copy()
                        plus[ai][idx]+=1;minus[ai][idx]-=1
                        po,ps=matrix_formula(plus,scalar,scale);mo,ms=matrix_formula(minus,scalar,scale)
                        # This objective is affine in each individual input coordinate.
                        grad[idx]=(np.sum(go*(po-mo))+np.sum(gs*(ps-ms)))/2
                    self.assertEqual(actual[keys[ai]],grad.tolist())

    def test_varlen_empty_segments_and_complete_state_gradients(self):
        for profile,a in (('scalar_decay_core',[0,Q(1,2)]),('diagonal_decay_core',[[0],[Q(1,2)]])):
            args=([[1],[1]],[[1],[2]],[[3],[4]],a)
            kw=dict(profile=profile,S0_list=[[[7]],[[9]],[[10]]])
            for size in (None,1,3):
                self.assertEqual(plain(L.decayed_varlen(*args,[0,1,1,2],chunk_size=size,**kw)),
                                 {'out':[[3],[13]],'states':[[[3]],[[9]],[[13]]]})
            expected={'dq':[[0],[0]],'dk':[[3],[12]],'dv':[[1],[6]],'dS0_list':[[[0]],[[2]],[[Q(3,2)]]],
                      'da':[7,30] if profile=='scalar_decay_core' else [[7],[30]]}
            self.assertEqual(plain(L.decayed_backward_varlen(*args,[[0],[0]],[0,1,1,2],dST_list=[[[1]],[[2]],[[3]]],**kw)),expected)
            empty=dict(profile=profile,S0_list=[[[7]],[[9]]])
            self.assertEqual(plain(L.decayed_varlen([],[],[],[],[0,0,0],**empty)),{'out':[],'states':empty['S0_list']})
            self.assertEqual(plain(L.decayed_backward_varlen([],[],[],[],[],[0,0,0],dST_list=[[[2]],[[3]]],**empty)),
                             {'dq':[],'dk':[],'dv':[],'da':[],'dS0_list':[[[2]],[[3]]]})

    def test_invalid_gate_or_gradient_is_rejected_before_arithmetic(self):
        for profile,bad in (('scalar_decay_core',[[1]]),('scalar_decay_core',[True]),
                            ('diagonal_decay_core',[1]),('diagonal_decay_core',[[1,2]])):
            with self.assertRaises(Invalid):L.decayed([[1]],[[1]],[[1]],bad,profile=profile)
        with self.assertRaises(Unsupported):L.decayed([[1]],[[1]],[[1]],[1],profile='gated_delta')
        with patch.object(L,'_decayed_recurrent',side_effect=AssertionError('computed too soon')):
            with self.assertRaises(Invalid):
                L.decayed_backward_varlen([[1]],[[1]],[[1]],[1],[[1]],[0,0,1],profile='scalar_decay_core',dST_list=[[[1]]])

if __name__=='__main__':unittest.main()
