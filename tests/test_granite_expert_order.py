import torch
import pytest
implementation=pytest.importorskip('transformers.models.granitemoe.modeling_granitemoe')
if not hasattr(implementation,'GraniteMoeExperts'):
    pytest.skip('This adapter targets the frozen Transformers 5.15.1 expert implementation',allow_module_level=True)
from transformers.models.granitemoe.configuration_granitemoe import GraniteMoeConfig
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeExperts
from scripts.run_granite_expert_order_confirmation import reversed_forward


def test_reverse_expert_order_preserves_double_precision_function_and_gradient():
    torch.manual_seed(71)
    config=GraniteMoeConfig(hidden_size=8,intermediate_size=4,num_local_experts=3,num_experts_per_tok=2)
    module=GraniteMoeExperts(config).double()
    for p in module.parameters():torch.nn.init.normal_(p,std=.1)
    x=torch.randn(5,8,dtype=torch.float64,requires_grad=True)
    indices=torch.tensor([[0,1],[1,2],[2,0],[0,2],[1,0]])
    weights=torch.full((5,2),.5,dtype=torch.float64)
    reverse,_=reversed_forward()
    a=module(x,indices,weights);ga=torch.autograd.grad(a.square().sum(),x)[0]
    b=reverse(module,x,indices,weights);gb=torch.autograd.grad(b.square().sum(),x)[0]
    torch.testing.assert_close(a,b,rtol=1e-13,atol=1e-15)
    torch.testing.assert_close(ga,gb,rtol=1e-13,atol=1e-15)
