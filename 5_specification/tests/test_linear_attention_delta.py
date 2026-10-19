import sys,unittest
from pathlib import Path
from fractions import Fraction as Q
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import linear_attention as L,d2_common as U
from kernel_spec.num import Invalid,Unsupported

def plain(result):return {k:U.plain(v) for k,v in result.items()}

def matrix_formula(args,profile,scale):
    q,k,v,s,beta,a=args;T,K=q.shape;V=v.shape[1];D=K*V
    matrices=[]
    for t in range(T):
        rows=np.ones(K,dtype=object) if profile=='delta_core' else np.repeat(a[t],K) if profile=='gated_delta_core' else a[t]
        transition=(np.eye(K,dtype=object)-beta[t]*np.outer(k[t],k[t]))@np.diag(rows)
        m=np.zeros((D+1,D+1),dtype=object)
        m[:D,:D]=np.kron(transition,np.eye(V,dtype=object))
        m[:D,D]=(beta[t]*np.outer(k[t],v[t])).reshape(D);m[D,D]=Q(1)
        matrices.append(m)
    initial=np.concatenate((s.reshape(D),[Q(1)]));states=[]
    for t in range(T):
        product=np.eye(D+1,dtype=object)
        for m in reversed(matrices[:t+1]):product=product@m
        states.append((product@initial)[:D].reshape(K,V))
    return np.stack([scale*q[t]@states[t] for t in range(T)]),states[-1]

class DeltaCores(unittest.TestCase):
    def test_kda_decay_precedes_delta_correction(self):
        result=plain(L.delta([[1,0]],[[1,1]],[[0]],[1],profile='kda_core',
                             a=[[2,3]],S0=[[1],[2]],scale=1))
        self.assertEqual(result,{'out':[[-6]],'state':[[-6],[-2]]})

    def test_independent_affine_products_and_all_input_gradients(self):
        def data(shape,offset):
            array=np.empty(shape,dtype=object)
            for n,i in enumerate(np.ndindex(shape)):array[i]=Q((n+offset)%5-2,2)
            return array
        for profile in ('delta_core','gated_delta_core','kda_core'):
            for K in (1,4):
                T,V=3,2;scale=Q(1,1 if K==1 else 2)
                args=[data((T,K),0),data((T,K),1),data((T,V),2),data((K,V),3),data((T,),4),
                      data((T,K) if profile=='kda_core' else (T,),1)]
                go,gs=data((T,V),4),data((K,V),0)
                expected,state=matrix_formula(args,profile,scale)
                call=[x.tolist() for x in args[:3]]+[args[4].tolist()]
                kw=dict(profile=profile,S0=args[3].tolist())
                if profile!='delta_core':kw['a']=args[5].tolist()
                self.assertEqual(plain(L.delta(*call,**kw)),{'out':expected.tolist(),'state':state.tolist()})
                actual=plain(L.delta_backward(*call,go.tolist(),dST=gs.tolist(),**kw))
                keys=('dq','dk','dv','dS0','dbeta')+(('da',) if profile!='delta_core' else ())
                self.assertEqual(set(actual),set(keys))
                for ai,key in enumerate(keys):
                    operand=args[ai];grad=np.empty(operand.shape,dtype=object)
                    for i in np.ndindex(operand.shape):
                        plus=list(args);minus=list(args);plus[ai]=operand.copy();minus[ai]=operand.copy()
                        plus[ai][i]+=1;minus[ai][i]-=1
                        po,ps=matrix_formula(plus,profile,scale);mo,ms=matrix_formula(minus,profile,scale)
                        # At most quadratic in any single input coordinate (k
                        # occurs twice at its time step); centered difference is exact.
                        grad[i]=(np.sum(go*(po-mo))+np.sum(gs*(ps-ms)))/2
                    self.assertEqual(actual[key],grad.tolist())

    def test_varlen_state_and_final_gradient_with_empty_segment(self):
        for profile,a in (('gated_delta_core',[0,Q(1,2)]),('kda_core',[[0],[Q(1,2)]])):
            args=([[1],[1]],[[1],[1]],[[3],[4]],[Q(1,2),Q(1,2)])
            kw=dict(profile=profile,a=a,S0_list=[[[7]],[[9]],[[10]]]);cu=[0,1,1,2]
            self.assertEqual(plain(L.delta_varlen(*args,cu,**kw)),
                             {'out':[[Q(3,2)],[Q(9,2)]],'states':[[[Q(3,2)]],[[9]],[[Q(9,2)]]]})
            result=plain(L.delta_backward_varlen(*args,[[0],[0]],cu,dST_list=[[[1]],[[2]],[[3]]],**kw))
            da=[Q(7,2),15] if profile=='gated_delta_core' else [[Q(7,2)],[15]]
            self.assertEqual(result,{'dq':[[0],[0]],'dk':[[Q(3,2)],[-9]],'dv':[[Q(1,2)],[Q(3,2)]],
                                     'dbeta':[3,-3],'da':da,'dS0_list':[[[0]],[[2]],[[Q(3,4)]]]})

    def test_empty_sequences_and_log_gate_chain_rule(self):
        for profile in ('delta_core','gated_delta_core','kda_core'):
            kw=dict(profile=profile,S0_list=[[[2]],[[3]]])
            if profile!='delta_core':kw.update(a=[],gate_parameterization='ln')
            self.assertEqual(plain(L.delta_varlen([],[],[],[],[0,0,0],**kw)),{'out':[],'states':kw['S0_list']})
            expected={'dq':[],'dk':[],'dv':[],'dbeta':[],'dS0_list':[[[4]],[[5]]]}
            if profile!='delta_core':expected['dg']=[]
            self.assertEqual(plain(L.delta_backward_varlen([],[],[],[],[],[0,0,0],dST_list=[[[4]],[[5]]],**kw)),expected)
        result=plain(L.delta_backward([[1]],[[1]],[[3]],[Q(1,2)],[[1]],profile='gated_delta_core',
                                     a=[0],gate_parameterization='ln',S0=[[2]],dST=[[2]]))
        self.assertEqual(result,{'dq':[[Q(5,2)]],'dk':[[-Q(3,2)]],'dv':[[Q(3,2)]],
                                 'dbeta':[3],'dg':[3],'dS0':[[Q(3,2)]]})

    def test_invalid_parameters_and_cotangents_rejected_before_calculation(self):
        for beta in ([],[1,2],[True],[[1]]):
            with self.assertRaises(Invalid):L.delta([[1]],[[1]],[[1]],beta,profile='delta_core')
        with self.assertRaises(Unsupported):L.delta([[1]],[[1]],[[1]],[1],profile='delta_core',a=[1])
        with self.assertRaises(Invalid):L.delta([[1]],[[1]],[[1]],[1],profile='gated_delta_core')
        with patch.object(L.rl,'exp',side_effect=AssertionError('computed too soon')):
            with self.assertRaises(Invalid):
                L.delta_backward_varlen([[1]],[[1]],[[1]],[1],[[1]],[0,0,1],profile='gated_delta_core',
                                        a=[0],gate_parameterization='ln',dST_list=[[[1]]])

if __name__=='__main__':unittest.main()
