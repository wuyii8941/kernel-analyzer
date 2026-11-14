"""Load unchanged Trident source with legacy API names, not changed arithmetic."""
from contextlib import contextmanager
import importlib


def legacy_autotune(original,*args,**kwargs):
    if len(args)<=2:return original(*args,**kwargs)
    if len(args)>6:raise TypeError('Unexpected legacy autotune signature')
    names=['configs','key','prune_configs_by','reset_to_zero','warmup','rep']
    options=dict(zip(names,args))
    if set(options)&set(kwargs):raise TypeError('Repeated legacy autotune argument')
    return original(**options,**kwargs)


def normalize_legacy_dtype_annotations(fn,dtype_type):
    fn.__annotations__={name:str(value)if isinstance(value,dtype_type)else value
                        for name,value in fn.__annotations__.items()}
    return fn


@contextmanager
def original_trident():
    import triton
    import triton.language as tl
    try:
        from triton.language.extra.cuda import libdevice
    except ImportError:
        # Triton2.1 exposes the same legacy functions directly in tl.math.
        libdevice=tl.math
    jit=triton.jit
    if triton.__version__.startswith('2.1.'):
        def legacy_jit(fn=None,**kwargs):
            def decorate(function):
                return jit(normalize_legacy_dtype_annotations(function,tl.dtype),**kwargs)
            return decorate(fn)if fn is not None else decorate
        triton.jit=legacy_jit
    autotune=triton.autotune;old_fast=getattr(tl.math,'fast_expf',None)
    old_pow=getattr(tl.math,'pow',None)
    old_max=getattr(tl.math,'max',None)
    old_tanh=getattr(tl.math,'tanh',None)
    triton.autotune=lambda *args,**kwargs:legacy_autotune(autotune,*args,**kwargs)
    if old_fast is None:tl.math.fast_expf=libdevice.fast_expf
    if old_tanh is None:tl.math.tanh=libdevice.tanh
    # Triton2.1 math.pow and current CUDA libdevice.pow map to the same
    # __nv_powif/__nv_powi/__nv_powf/__nv_pow symbols and signatures.
    if old_pow is None:tl.math.pow=libdevice.pow
    if old_max is None:
        # Legacy math.max used these CUDA libdevice symbols. Preserve them,
        # including their NaN behavior, rather than substitute tl.maximum.
        from triton.language import core
        import inspect
        signatures={(core.dtype("fp32"),core.dtype("fp32")):("__nv_fmaxf",core.dtype("fp32")),
                    (core.dtype("fp64"),core.dtype("fp64")):("__nv_fmax",core.dtype("fp64"))}
        if "_semantic" in inspect.signature(core.extern_elementwise).parameters:
            @core.extern
            def legacy_max(a,b,_semantic=None):
                return core.extern_elementwise("","",[a,b],signatures,is_pure=True,_semantic=_semantic)
        else:
            @core.extern
            def legacy_max(a,b,_builder=None):
                return core.extern_elementwise("","",[a,b],signatures,is_pure=True,_builder=_builder)
        tl.math.max=legacy_max
    try:yield importlib.import_module('trident')
    finally:
        triton.autotune=autotune
        triton.jit=jit
        if old_fast is None:delattr(tl.math,'fast_expf')
        if old_pow is None:delattr(tl.math,'pow')
        if old_max is None:delattr(tl.math,'max')
        if old_tanh is None:delattr(tl.math,'tanh')

