"""Mathematical reference for the observed multi-output decayed recurrence.

Training replay integration remains pending. Inputs are in recurrence order,
which can differ from chronological order in a backward kernel.
"""


def evaluate(initial, log_rate, time_inputs, bias, injections):
    """Return every h[t+1] = exp(-exp(log_rate)*softplus(dt[t]+bias))*h[t]+v[t].

    h/log_rate: [channels, state_width], dt: [steps, channels], bias:
    [channels], v: [steps, channels, state_width]. All arithmetic uses the
    supplied common FP32/FP64 dtype. Softplus uses the declared threshold 20.
    No claim is made that this reference is unbiased or exact real arithmetic.
    """
    import torch
    if (initial.ndim != 2 or log_rate.shape != initial.shape
            or time_inputs.ndim != 2 or time_inputs.shape[1] != initial.shape[0]
            or time_inputs.shape[0] == 0 or bias.shape != (initial.shape[0],)
            or injections.shape != (*time_inputs.shape, initial.shape[1])
            or initial.dtype not in (torch.float32, torch.float64)
            or any(x.dtype != initial.dtype or x.device != initial.device
                   for x in (log_rate, time_inputs, bias, injections))):
        raise ValueError('Invalid recurrence shapes, dtype or device')
    if any(not torch.isfinite(x).all() for x in (initial, log_rate, time_inputs, bias, injections)):
        raise ValueError('Finite recurrence inputs required')
    dt = torch.nn.functional.softplus(time_inputs + bias, beta=1, threshold=20)
    decay = torch.exp(-torch.exp(log_rate)[None] * dt[..., None])
    state = initial
    outputs = []
    for factor, addition in zip(decay, injections):
        state = factor * state + addition
        outputs.append(state)
    result = torch.stack(outputs)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite recurrence result')
    return result


def from_runtime_pointers(pointers, *, channels, width, outputs, output_index):
    """Decode the source-checked pointer layout; not proof of runtime identity.

    Flat storage is required explicitly. A caller must establish executed source
    identity, output layout and non-aliasing before treating this as a replay.
    """
    import torch
    if min(channels,width,outputs)<1 or not 0<=output_index<outputs:
        raise ValueError('Invalid recurrence dimensions or output selection')
    sizes={0:channels,1:width,2:channels*width,3:(outputs+1)*channels,4:channels}
    for j in range(outputs):
        sizes[5+2*j]=channels
        sizes[6+2*j]=width
    values={}
    device=None
    for i,size in sizes.items():
        tensor=pointers.get(f'in_ptr{i}')
        dtype=torch.float32 if i==2 else torch.bfloat16
        if (not isinstance(tensor,torch.Tensor) or tensor.dtype!=dtype
                or tensor.numel()!=size or not tensor.is_contiguous()
                or not torch.isfinite(tensor).all()):
            raise ValueError('Invalid recurrence pointer: in_ptr'+str(i))
        if device is None: device=tensor.device
        if tensor.device!=device: raise ValueError('Recurrence device mismatch')
        values[i]=tensor.reshape(-1).float()
    initial=values[0][:,None]*values[1][None,:]
    times=values[3].reshape(outputs+1,channels)[1:].flip(0)
    injections=torch.stack([values[5+2*j][:,None]*values[6+2*j][None,:] for j in range(outputs)])
    result=evaluate(initial,values[2].reshape(channels,width),times,values[4],injections)
    return result[output_index]


def check_source(source, symbol):
    from pathlib import Path
    import hashlib
    from kernel_analyzer import decayed_recurrence_source
    contract=decayed_recurrence_source.check_source(source,symbol)
    dependency=Path(decayed_recurrence_source.__file__)
    # decayed_recurrence_source already validates the complete generated body.
    # Its full digest remains provenance; this stable digest excludes only
    # generated decorator metadata such as the logical CUDA device index.
    import ast
    tree=ast.parse(source)
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id==symbol for t in n.targets))
    fn=next(n for n in ast.parse(assignment.value.args[1].value).body
            if isinstance(n,ast.FunctionDef) and n.name==symbol)
    semantic=ast.Module(body=[ast.FunctionDef(name=fn.name,args=fn.args,body=fn.body,
        decorator_list=[],returns=fn.returns,type_comment=fn.type_comment)],type_ignores=[])
    return dict(contract, family='DECAYED_RECURRENCE', output_pointer='out_ptr0',
                function_semantic_ast_sha256=hashlib.sha256(ast.dump(semantic).encode()).hexdigest(),
                checker_dependency_sha256=hashlib.sha256(dependency.read_bytes()).hexdigest())


def reference(metadata, candidate, contract):
    """Evaluate exactly the declared output, with fail-closed pointer checks."""
    import hashlib
    import torch
    from pathlib import Path
    from kernel_analyzer import decayed_recurrence_source
    digest=hashlib.sha256(Path(decayed_recurrence_source.__file__).read_bytes()).hexdigest()
    if contract.get('checker_dependency_sha256')!=digest:
        raise ValueError('Recurrence source checker changed')
    pointer=contract.get('output_pointer')
    if (pointer not in contract['output_pointers'] or metadata.get('formal_pointer')!=pointer
            or metadata.get('symbol')!=contract['symbol']
            or metadata.get('input_output_storage_aliases')!=[]):
        raise ValueError('Recurrence output identity or alias mismatch')
    channels,width=contract['channels'],contract['state_width']
    if (candidate.dtype!=torch.float32 or candidate.numel()!=channels*width
            or not candidate.is_contiguous()):
        raise ValueError('Recurrence output layout differs')
    result=from_runtime_pointers(metadata.get('runtime_pointers',{}),channels=channels,
        width=width,outputs=contract['outputs'],output_index=contract['output_pointers'].index(pointer))
    if result.device!=candidate.device: raise ValueError('Recurrence output device mismatch')
    return result.reshape(candidate.shape)
