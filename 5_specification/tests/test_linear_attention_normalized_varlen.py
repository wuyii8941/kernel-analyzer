import sys,unittest
from pathlib import Path
from fractions import Fraction as Q
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import linear_attention as L,d2_common as U
from kernel_spec.num import Invalid,Undefined,Unsupported

def plain(result):return {key:U.plain(value) for key,value in result.items()}

def components(args,cu,scale,epsilon):
    q,k,v,s,z=args
    T=len(q)
    ids=np.array([n for n,(a,b) in enumerate(zip(cu,cu[1:])) for _ in range(a,b)])
    mask=(ids[:,None]==ids[None,:])&np.tri(T,dtype=bool)
    numerator=scale*(np.sum(q[:,:,None]*s[ids],axis=1)+((q@k.T)*mask)@v)
    denominator=scale*np.sum(q*(z[ids]+mask@k),axis=1)+epsilon
    states=np.stack([s[n]+k[a:b].T@v[a:b] for n,(a,b) in enumerate(zip(cu,cu[1:]))])
    zstates=np.stack([z[n]+np.sum(k[a:b],axis=0) for n,(a,b) in enumerate(zip(cu,cu[1:]))])
    return numerator,denominator,states,zstates

class NormalizedVariableLengthLinearAttention(unittest.TestCase):
    def test_manual_states_and_empty_segment(self):
        args=([[1],[1]],[[1],[1]],[[1],[2]])
        kw=dict(epsilon=0,S0_list=[[[2]],[[7]],[[10]]],Z0_list=[[4],[8],[2]])
        for size in (None,1,3):
            self.assertEqual(plain(L.normalized_varlen(*args,[0,1,1,2],chunk_size=size,**kw)),
                             {'out':[[Q(3,5)],[4]],'states':[[[3]],[[7]],[[12]]],'zstates':[[5],[8],[3]]})
        self.assertEqual(plain(L.normalized_backward_varlen(*args,[[1],[1]],[0,1,1,2],
                         dST_list=[[[2]],[[4]],[[6]]],dZT_list=[[3],[5],[7]],**kw)),
                         {'dq':[[0],[0]],'dk':[[Q(127,25)],[Q(55,3)]],
                          'dv':[[Q(11,5)],[Q(19,3)]],'dS0_list':[[[Q(11,5)]],[[4]],[[Q(19,3)]]],
                          'dZ0_list':[[Q(72,25)],[5],[Q(17,3)]]})

    def test_all_empty_and_default_states(self):
        kw=dict(epsilon=0,S0_list=[[[2,3]],[[4,5]]],Z0_list=[[6],[7]])
        self.assertEqual(plain(L.normalized_varlen([],[],[],[0,0,0],**kw)),
                         {'out':[],'states':kw['S0_list'],'zstates':kw['Z0_list']})
        self.assertEqual(plain(L.normalized_backward_varlen([],[],[],[],[0,0,0],
                         dST_list=[[[8,9]],[[10,11]]],dZT_list=[[12],[13]],**kw)),
                         {'dq':[],'dk':[],'dv':[],'dS0_list':[[[8,9]],[[10,11]]],'dZ0_list':[[12],[13]]})
        self.assertEqual(plain(L.normalized_varlen([[1]],[[1]],[[2]],[0,0,1],epsilon=0)),
                         {'out':[[2]],'states':[[[0]],[[2]]],'zstates':[[0],[1]]})

    def test_all_metadata_validated_before_any_segment_arithmetic(self):
        args=([[0],[1]],[[0],[1]],[[1],[1]])
        for kw in ({'Z0_list':[[1]]},{'Z0_list':[[1],[2,3]]},{'S0_list':[[[1]]]},
                   {'epsilon':-1},{'chunk_size':True}):
            with patch.object(L,'_normalized_trace',side_effect=AssertionError('computed too soon')):
                with self.assertRaises(Invalid):L.normalized_varlen(*args,[0,1,2],**({'epsilon':0}|kw))
        for kw in ({'dZT_list':[[1]]},{'dZT_list':[[1],[2,3]]},{'dST_list':[[[1]]]}):
            with patch.object(L,'_normalized_trace',side_effect=AssertionError('computed too soon')):
                with self.assertRaises(Invalid):L.normalized_backward_varlen(*args,[[1],[1]],[0,1,2],epsilon=0,**kw)
        with self.assertRaises(Undefined):L.normalized_varlen(*args,[0,1,2],epsilon=0)
        with self.assertRaises(Undefined):L.normalized_backward_varlen(*args,[[1],[1]],[0,1,2],epsilon=0)
        with self.assertRaises(Unsupported):L.normalized_varlen(*args,[0,2],epsilon=1,profile='unnormalized_core')

    def test_independent_exact_segment_mask_and_all_five_gradients(self):
        def values(shape,offset):
            out=np.empty(shape,dtype=object)
            for n,i in enumerate(np.ndindex(shape)):out[i]=Q((n+offset)%3+1,2)
            return out
        cu=[0,0,2,2,3,5,5];T,N=5,6
        for K in (1,4):
            for epsilon in (Q(0),Q(1,2)):
                V=2;scale=Q(1,1 if K==1 else 2)
                args=[values((T,K),0),values((T,K),1),values((T,V),2),values((N,K,V),1),values((N,K),0)]
                go,gs,gz=values((T,V),0),values((N,K,V),2),values((N,K),1)
                n,d,s,z=components(args,cu,scale,epsilon)
                call=[a.tolist() for a in args[:3]]
                kw=dict(epsilon=epsilon,S0_list=args[3].tolist(),Z0_list=args[4].tolist())
                for size in (None,1,3):
                    self.assertEqual(plain(L.normalized_varlen(*call,cu,chunk_size=size,**kw)),
                                     {'out':(n/d[:,None]).tolist(),'states':s.tolist(),'zstates':z.tolist()})
                actual=plain(L.normalized_backward_varlen(*call,go.tolist(),cu,dST_list=gs.tolist(),dZT_list=gz.tolist(),**kw))
                keys=('dq','dk','dv','dS0_list','dZ0_list')
                self.assertEqual(set(actual),set(keys))
                for a,operand in enumerate(args):
                    grad=np.empty(operand.shape,dtype=object)
                    for i in np.ndindex(operand.shape):
                        plus=list(args);minus=list(args);plus[a]=operand.copy();minus[a]=operand.copy()
                        plus[a][i]+=1;minus[a][i]-=1
                        pn,pd,ps,pz=components(plus,cu,scale,epsilon)
                        mn,md,ms,mz=components(minus,cu,scale,epsilon)
                        # Each component is affine in a single coordinate; these
                        # are exact component derivatives, not approximate VJPs.
                        dn,dd,ds,dz=(pn-mn)/2,(pd-md)/2,(ps-ms)/2,(pz-mz)/2
                        grad[i]=np.sum(go*(dn*d[:,None]-n*dd[:,None])/(d[:,None]**2))+np.sum(gs*ds)+np.sum(gz*dz)
                    self.assertEqual(actual[keys[a]],grad.tolist())

if __name__=='__main__':unittest.main()
