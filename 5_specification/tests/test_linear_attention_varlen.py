import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import linear_attention as L,d2_common as U
from kernel_spec.num import Invalid

def plain(result):return {key:U.plain(value) for key,value in result.items()}

class VariableLengthLinearAttention(unittest.TestCase):
    def test_each_sequence_owns_its_initial_and_final_state(self):
        q=k=[[1],[1],[1]];v=[[1],[2],[3]];cu=[0,1,1,3];initial=[[[2]],[[7]],[[10]]]
        for size in (None,1,2,8):
            self.assertEqual(plain(L.varlen(q,k,v,cu,S0_list=initial,chunk_size=size)),
                             {'out':[[3],[12],[15]],'states':[[[3]],[[7]],[[15]]]})
        result=plain(L.backward_varlen(q,k,v,[[1],[1],[1]],cu,S0_list=initial,dST_list=[[[2]],[[4]],[[6]]]))
        self.assertEqual(result,{'dq':[[3],[12],[15]],'dk':[[3],[16],[21]],'dv':[[3],[8],[7]],
                                 'dS0_list':[[[3]],[[4]],[[8]]]})

    def test_all_empty_segments_preserve_every_state_position(self):
        initial=[[[2,3]],[[4,5]]];gradients=[[[6,7]],[[8,9]]]
        for size in (None,2):
            self.assertEqual(plain(L.varlen([],[],[],[0,0,0],S0_list=initial,chunk_size=size)),
                             {'out':[],'states':initial})
        self.assertEqual(plain(L.backward_varlen([],[],[],[],[0,0,0],S0_list=initial,dST_list=gradients)),
                         {'dq':[],'dk':[],'dv':[],'dS0_list':gradients})
        self.assertEqual(plain(L.varlen([[1]],[[1]],[[2]],[0,0,1])),
                         {'out':[[2]],'states':[[[0]],[[2]]]})

    def test_boundaries_and_state_counts_are_validated_before_computation(self):
        for cu in ([1,2],[0,1],[0,3],[0,2,1,2],[0,1.5,2],[0,True,2],[0],[]):
            with self.subTest(cu=cu),self.assertRaises(Invalid):
                L.varlen([[1],[1]],[[1],[1]],[[1],[1]],cu)
        for states in ([[[1]]],[[[1]],[[2]],[[3]]],[[[1]],[[2,3]]]):
            with self.assertRaises(Invalid):L.varlen([[1],[1]],[[1],[1]],[[1],[1]],[0,1,2],S0_list=states)
        with self.assertRaises(Invalid):
            L.backward_varlen([[1]],[[1]],[[1]],[[1]],[0,0,1],dST_list=[[[1]]])
        with self.assertRaises(Invalid):L.varlen([],[],[],[0,0])

    def test_independent_segment_mask_and_autograd(self):
        import torch
        T,K,V=5,4,2;cu=[0,2,2,3,5];N=len(cu)-1
        def data(shape,offset):
            import math
            return ((torch.arange(math.prod(shape),dtype=torch.float64)+offset)%5-2).reshape(shape).requires_grad_()
        q,k,v,s0=data((T,K),0),data((T,K),1),data((T,V),2),data((N,K,V),3)
        ids=torch.tensor([n for n,(a,b) in enumerate(zip(cu,cu[1:])) for _ in range(a,b)])
        mask=(ids[:,None]==ids[None,:])&torch.ones(T,T,dtype=torch.bool).tril()
        out=(torch.einsum('tk,tkv->tv',q,s0[ids])+(q@k.T).masked_fill(~mask,0)@v)/2
        states=torch.stack([s0[n]+k[a:b].T@v[a:b] for n,(a,b) in enumerate(zip(cu,cu[1:]))])
        go=torch.ones_like(out)*3;gs=torch.ones_like(states)*2
        ((out*go).sum()+(states*gs).sum()).backward()
        args=[x.detach().tolist() for x in (q,k,v)];initial=s0.detach().tolist()
        for size in (None,1,3):
            actual=plain(L.varlen(*args,cu,S0_list=initial,chunk_size=size))
            self.assertEqual(actual['out'],out.detach().tolist());self.assertEqual(actual['states'],states.detach().tolist())
        gradients=plain(L.backward_varlen(*args,go.tolist(),cu,S0_list=initial,dST_list=gs.tolist()))
        for key,x in zip(('dq','dk','dv','dS0_list'),(q,k,v,s0)):
            self.assertEqual(gradients[key],x.grad.tolist())

if __name__=='__main__':unittest.main()
