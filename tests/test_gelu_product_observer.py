import pytest
import torch
from kernel_analyzer.gelu_product_observer import validate_pointers, observer_class


def test_output_cannot_alias_input_even_when_shapes_match():
    x=torch.ones(6,dtype=torch.bfloat16)
    pointers=dict(in_ptr0=x,in_ptr1=torch.ones(24,dtype=torch.bfloat16),
                  in_ptr2=x.clone(),out_ptr0=torch.empty_like(x))
    contract=dict(elements=6,width=2,stride=8,offset=5)
    values=validate_pointers(pointers,contract)
    assert len(values)==3
    pointers['out_ptr0']=x
    with pytest.raises(ValueError,match='shared allocation'):
        validate_pointers(pointers,contract)
    pointers['out_ptr0']=torch.empty(6)
    with pytest.raises(ValueError,match='output storage'):
        validate_pointers(pointers,contract)


def test_wrapper_executes_and_restores_after_invalid_count(tmp_path):
    import ast
    from types import SimpleNamespace
    from kernel_analyzer.gelu_product_source import expected_body,check_source
    signature=dict(in_ptr0='*bf16',in_ptr1='*bf16',in_ptr2='*bf16',out_ptr0='*bf16',xnumel='i32')
    body='\n'.join('    '+ast.unparse(n) for n in expected_body(6,2,8,5))
    definition=(f'@heuristic(triton_meta={{"signature": {signature!r}}})\n'
                'def gelu(in_ptr0,in_ptr1,in_ptr2,out_ptr0,xnumel,XBLOCK):\n'+body)
    source=f'gelu = compiler.triton("gelu", {definition!r})'
    path=tmp_path/'generated.py';path.write_text(source)
    contract=check_source(source,'gelu',elements=6,width=2,stride=8,offset=5)
    calls=[]
    def original(*args,**kwargs):calls.append(True)
    kernel=SimpleNamespace(run=original)
    module=SimpleNamespace(__file__=str(path),gelu=kernel)
    class Base:
        def __init__(self,modules):self.modules=modules
        def __enter__(self):return self
        def __exit__(self,*args):return False
    cls=observer_class(Base,{'gelu':contract},lambda k:list(signature.items()))
    tensors=[torch.ones(n,dtype=torch.bfloat16) for n in (6,24,6,6)]
    with cls(modules=[module]):
        kernel.run(*tensors,6)
        with pytest.raises(ValueError,match='element count'):
            kernel.run(*tensors,7)
    assert calls==[True]
    assert kernel.run is original
