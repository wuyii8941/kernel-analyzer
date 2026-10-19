import sys,unittest
from fractions import Fraction as Q
from pathlib import Path
import copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import linear_attention as L,d2_common as U
from kernel_spec.num import Invalid,Unsupported

def plain(result):
    return {key:U.plain(value) for key,value in result.items()}


class LinearAttentionCore(unittest.TestCase):
    def test_original_state_counterexample_and_complete_rectangular_output(self):
        for fn,args in ((L.recurrent,()),(L.chunked,(2,))):
            state=[[2,4]];before=copy.deepcopy(state)
            result=plain(fn([[1]],[[1]],[[3,5]],*args,S0=state))
            self.assertEqual(result,{'out':[[5,9]],'state':[[5,9]]})
            self.assertEqual(state,before)

    def test_default_scale_is_shared_by_forward_chunked_and_backward(self):
        for K in (3,4):
            q=k=[[1]+[0]*(K-1)];v=[[1]]
            a=L.recurrent(q,k,v)['out'][0][0]
            b=L.chunked(q,k,v,1)['out'][0][0]
            gradients=L.backward(q,k,v,[[1]])
            for value in (a,b,gradients['dq'][0][0],gradients['dk'][0][0],gradients['dv'][0][0],gradients['dS0'][0][0]):
                lo,hi=value.bounds()
                self.assertGreater(lo,0)
                self.assertLessEqual(lo*lo,Q(1,K));self.assertGreaterEqual(hi*hi,Q(1,K))
                if K==4:self.assertEqual(lo,hi);self.assertEqual(lo,Q(1,2))

    def test_all_outputs_and_gradients_against_independent_quadratic_form(self):
        import torch
        for T in (1,3):
            for K in (1,4):
                for V in (1,3):
                    for scale in (None,Q(3,2)):
                        def data(shape,offset):
                            import math
                            return ((torch.arange(math.prod(shape),dtype=torch.float64)+offset)%5-2).reshape(shape).requires_grad_()
                        q,k,v,s0=data((T,K),0),data((T,K),1),data((T,V),2),data((K,V),3)
                        actual_scale=K**-0.5 if scale is None else float(scale)
                        output=actual_scale*(q@s0+(q@k.T).tril()@v)
                        state=s0+k.T@v
                        gO=torch.ones_like(output)*3;gST=torch.ones_like(state)*2
                        ((output*gO).sum()+(state*gST).sum()).backward()
                        arrays=[x.detach().tolist() for x in (q,k,v)]
                        kw={'S0':s0.detach().tolist(),'scale':scale}
                        for fn,args in ((L.recurrent,()),*( (L.chunked,(size,)) for size in (1,2,7))):
                            result=plain(fn(*arrays,*args,**kw))
                            self.assertEqual(result['out'],output.detach().tolist())
                            self.assertEqual(result['state'],state.detach().tolist())
                        gradients=plain(L.backward(*arrays,gO.tolist(),dST=gST.tolist(),**kw))
                        self.assertEqual(set(gradients),{'dq','dk','dv','dS0'})
                        for name,x in zip(('dq','dk','dv','dS0'),(q,k,v,s0)):
                            self.assertEqual(gradients[name],x.grad.tolist(),(T,K,V,scale,name))

    def test_final_state_cotangent_is_not_dropped(self):
        result=plain(L.backward([[5]],[[2]],[[3]],[[0]],S0=[[7]],dST=[[4]]))
        self.assertEqual(result,{'dq':[[0]],'dk':[[12]],'dv':[[8]],'dS0':[[4]]})

    def test_empty_sequence_preserves_state_and_its_gradient(self):
        for fn,args in ((L.recurrent,()),(L.chunked,(2,))):
            self.assertEqual(plain(fn([],[],[],*args,S0=[[2,3]])),{'out':[],'state':[[2,3]]})
        self.assertEqual(plain(L.backward([],[],[],[],S0=[[2,3]],dST=[[4,5]])),
                         {'dq':[],'dk':[],'dv':[],'dS0':[[4,5]]})

    def test_invalid_structure_and_unimplemented_profiles_are_rejected(self):
        for fn,args in ((L.recurrent,()),(L.chunked,(2,))):
            for q,k,v,s0 in [([[1]],[[1],[2]],[[3]],None),([[1]],[[1]],[[3]],[[2,4]]),
                              ([],[],[],None),([[1,2]],[[1]],[[3]],None)]:
                with self.assertRaises(Invalid):fn(q,k,v,*args,S0=s0)
            with self.assertRaises(Unsupported):fn([[1]],[[1]],[[3]],*args,profile='normalized')
        for size in (0,-1,1.5,True):
            with self.assertRaises(Invalid):L.chunked([[1]],[[1]],[[3]],size)
        for gradients in ({'dO':[[1,2]]},{'dO':[[1]],'dST':[[1,2]]}):
            with self.assertRaises(Invalid):L.backward([[1]],[[1]],[[3]],**gradients)

if __name__=='__main__':unittest.main()
