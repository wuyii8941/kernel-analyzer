"""Replace only Top-k inside one explicitly selected eager router forward."""
from contextlib import contextmanager
import hashlib
import inspect
import types
import torch
from torch.overrides import TorchFunctionMode
from kernel_analyzer.selection_reference import stable_selection, validate_selection


class SelectionMode(TorchFunctionMode):
    def __init__(self, k, replace, records, captured=None):
        self.k, self.replace, self.records = k, replace, records
        self.captured = captured

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func not in (torch.topk, torch.Tensor.topk):
            return func(*args, **kwargs)
        scores = args[0] if args else kwargs['input']
        k = args[1] if len(args)>1 else kwargs['k']
        dim = args[2] if len(args)>2 else kwargs.get('dim',-1)
        largest = args[3] if len(args)>3 else kwargs.get('largest',True)
        sorted_output = args[4] if len(args)>4 else kwargs.get('sorted',True)
        if (k!=self.k or scores.ndim!=2 or dim not in (1,-1)
                or not largest or not sorted_output or kwargs.get('out') is not None):
            raise ValueError('Router Top-k contract differs')
        actual = func(*args, **kwargs)
        diagnostic = validate_selection(scores, actual.values, actual.indices,k,dim)
        values, indices, _ = stable_selection(scores,k,dim)
        if self.captured is not None:
            chosen_values,chosen_indices=(values,indices) if self.replace else actual
            self.captured.append(dict(scores=scores.detach().cpu().clone(),
                values=chosen_values.detach().cpu().clone(),indices=chosen_indices.detach().cpu().clone()))
        self.records.append(dict(**diagnostic,
            input_shape=list(scores.shape), input_dtype=str(scores.dtype),
            changed_index_coordinates=int((indices!=actual.indices).sum()),
            changed_value_coordinates=int((values!=actual.values).sum()),
            replacement_enabled=self.replace))
        return type(actual)((values,indices)) if self.replace else actual


def forward_sha256(router):
    return hashlib.sha256(inspect.getsource(type(router).forward).encode()).hexdigest()


@contextmanager
def selected_router_topk(router, *, expected_source_sha256, replace, records, captured=None):
    if forward_sha256(router)!=expected_source_sha256:
        raise ValueError('Router forward source changed')
    if 'forward' in router.__dict__:
        raise ValueError('Router already has an instance forward override')
    original=router.forward
    def forward(self,*args,**kwargs):
        start=len(records)
        with SelectionMode(self.top_k,replace,records,captured):
            result=original(*args,**kwargs)
        if len(records)-start!=1:
            raise ValueError('Expected exactly one router Top-k call')
        return result
    router.forward=types.MethodType(forward,router)
    try:
        yield router
    finally:
        del router.forward
