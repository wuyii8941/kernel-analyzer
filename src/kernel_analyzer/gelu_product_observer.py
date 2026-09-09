"""Pre-call validation for the shared observer's GELU product capture."""
from pathlib import Path
from kernel_analyzer.gelu_product_source import check_source
from kernel_analyzer.gelu_product_reference import decode_inputs


def validate_pointers(pointers, contract):
    import torch
    if set(pointers)!={'in_ptr0','in_ptr1','in_ptr2','out_ptr0'}:
        raise ValueError('Unexpected GELU pointer signature')
    output=pointers['out_ptr0']
    if (not isinstance(output,torch.Tensor) or output.dtype!=torch.bfloat16
            or output.numel()!=contract['elements'] or not output.is_contiguous()):
        raise ValueError('GELU output storage differs')
    inputs={k:v for k,v in pointers.items() if k!='out_ptr0'}
    decoded=decode_inputs(inputs,**{k:contract[k] for k in ('elements','width','stride','offset')})
    if any(v.device!=output.device for v in inputs.values()):
        raise ValueError('GELU output device differs')
    if any(v.untyped_storage().data_ptr()==output.untyped_storage().data_ptr() for v in inputs.values()):
        raise ValueError('GELU input/output shared allocation is unsupported')
    return decoded


def observer_class(base, contracts, runtime_signature):
    class CheckedObserver(base):
        def __init__(self, **kwargs):
            if kwargs.get('allow_missing_symbols'):
                raise ValueError('GELU source substitution is not allowed')
            modules=list(kwargs['modules']);kwargs['modules']=modules
            self.gelu_restores=[]
            for symbol,contract in contracts.items():
                found=[m for m in modules if hasattr(m,symbol)]
                if len(found)!=1: raise ValueError('Unique executed GELU module required')
                actual=check_source(Path(found[0].__file__).read_text(),symbol,
                    **{k:contract[k] for k in ('elements','width','stride','offset')})
                if actual['function_ast_sha256']!=contract['function_ast_sha256']:
                    raise ValueError('Executed GELU definition differs')
            super().__init__(**kwargs)

        def __enter__(self):
            started=False
            try:
                for module in self.modules:
                    for symbol,contract in contracts.items():
                        kernel=getattr(module,symbol,None)
                        if kernel is None: continue
                        signature=runtime_signature(kernel)
                        names=[str(n) for n,t in signature if str(t).startswith('*')]
                        if names!=['in_ptr0','in_ptr1','in_ptr2','out_ptr0']:
                            raise ValueError('Runtime GELU pointer ordering differs')
                        original=kernel.run
                        def checked(*args,_original=original,_contract=contract,**kwargs):
                            if len(args)<5 or args[4]!=_contract['elements']:
                                raise ValueError('Runtime GELU element count differs')
                            validate_pointers(dict(zip(['in_ptr0','in_ptr1','in_ptr2','out_ptr0'],args[:4])),_contract)
                            return _original(*args,**kwargs)
                        self.gelu_restores.append((kernel,original))
                        kernel.run=checked
                started=True
                return super().__enter__()
            except BaseException:
                import sys
                try:
                    if started: super().__exit__(*sys.exc_info())
                finally: self._restore()
                raise

        def _restore(self):
            for kernel,original in reversed(self.gelu_restores):kernel.run=original
            self.gelu_restores.clear()

        def __exit__(self,*args):
            try:return super().__exit__(*args)
            finally:self._restore()
    return CheckedObserver
