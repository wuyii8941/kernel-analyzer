import pytest
import torch
from kernel_analyzer.router_selection import selected_router_topk,forward_sha256


class Router(torch.nn.Module):
    top_k=2
    def forward(self,x):
        values,indices=x.topk(self.top_k,dim=1)
        return torch.softmax(values,dim=1),indices


def test_scoped_topk_replacement_and_gradient():
    router=Router();records=[]
    x=torch.tensor([[1.,3.,2.]],requires_grad=True)
    with selected_router_topk(router,expected_source_sha256=forward_sha256(router),replace=True,records=records):
        gates,indices=router(x)
        (gates*torch.tensor([[2.,1.]])).sum().backward()
    assert 'forward' not in router.__dict__
    assert indices.tolist()==[[1,2]]
    assert x.grad[0,0]==0 and x.grad[0,1]!=0
    assert len(records)==1 and records[0]['replacement_enabled']


def test_source_mismatch_and_restore_on_contract_failure():
    router=Router()
    with pytest.raises(ValueError,match='source'):
        with selected_router_topk(router,expected_source_sha256='wrong',replace=True,records=[]):pass
    with pytest.raises(ValueError,match='contract'):
        with selected_router_topk(router,expected_source_sha256=forward_sha256(router),replace=False,records=[]):
            router(torch.ones(2,2,2))
    assert 'forward' not in router.__dict__
