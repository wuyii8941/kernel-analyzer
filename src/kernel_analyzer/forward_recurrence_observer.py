"""Same-call forward recurrence capture; no replacement or tuning changes.

Loaded source is checked. Binary identity and resolved launch geometry require
additional evidence before a training-level conclusion is permitted.
"""
from pathlib import Path
from kernel_analyzer.forward_state_recurrence_source import check_source
from kernel_analyzer.forward_state_recurrence_reference import decode_inputs
from kernel_analyzer.forward_recurrence_diagnostic import analyze


class Observer:
    def __init__(self, modules, contracts, sink):
        self.modules, self.contracts, self.sink = list(modules), contracts, sink
        self.restores, self.counts = [], {}

    def __enter__(self):
        if self.restores:
            raise ValueError('Observer already installed')
        try:
            for symbol, contract in self.contracts.items():
                modules = [m for m in self.modules if hasattr(m, symbol)]
                if len(modules) != 1:
                    raise ValueError('Missing or ambiguous loaded recurrence')
                module = modules[0]
                if check_source(Path(module.__file__).read_text(), symbol) != contract:
                    raise ValueError('Loaded recurrence source changed')
                kernel = getattr(module, symbol)
                signature = {f'in_ptr{i}': '*fp32' if i == 0 else '*bf16' for i in range(5)}
                signature.update({f'out_ptr{i}': '*bf16' if i == contract['steps']-1 else '*fp32'
                                  for i in range(contract['steps'])})
                signature['xnumel'] = 'i32'
                if list(kernel.triton_meta['signature'].items()) != list(signature.items()):
                    raise ValueError('Loaded ordered storage signature differs')
                original = kernel.run
                def wrapped(*args, _run=original, _c=contract, _symbol=symbol, **kwargs):
                    import torch
                    steps = _c['steps']
                    if (len(args) != steps+6 or set(kwargs) != {'stream'}
                            or type(args[-1]) is not int
                            or args[-1] != _c['channels']*_c['state_width']):
                        raise ValueError('Unexpected recurrence invocation')
                    dimensions = {k: _c[k] for k in ('steps', 'channels', 'state_width',
                                                     'packed_width', 'state_offset')}
                    inputs = dict(zip((f'in_ptr{i}' for i in range(5)), args[:5]))
                    decode_inputs(inputs, **dimensions)  # Validate before execution.
                    before = {k: v.detach().clone() for k, v in inputs.items()}
                    buffers = args[5:-1]
                    device = args[0].device
                    for i, v in enumerate(buffers):
                        if (not isinstance(v, torch.Tensor) or not v.is_contiguous()
                                or v.numel() != args[-1] or v.device != device
                                or v.dtype != (torch.bfloat16 if i == steps-1 else torch.float32)):
                            raise ValueError('Invalid output storage')
                    # Reject shared allocations conservatively; no reading of
                    # uninitialized outputs is used for this check.
                    addresses = [v.untyped_storage().data_ptr() for v in args[:-1]]
                    if len(set(addresses)) != len(addresses):
                        raise ValueError('Aliased recurrence storage requires separate review')
                    result = _run(*args, **kwargs)
                    after = {f'out_ptr{i}': v.detach().clone() for i, v in enumerate(buffers)}
                    diagnostic = analyze(before, after, **dimensions)
                    index = self.counts.get(_symbol, 0)
                    self.sink(dict(pre_call_inputs=before, post_call_outputs=after,
                                   diagnostic=diagnostic, dimensions=dimensions,
                                   symbol=_symbol, invocation_index=index,
                                   function_ast_sha256=_c['function_ast_sha256'],
                                   binary_identity_verified=False, resolved_launch_verified=False))
                    self.counts[_symbol] = index+1
                    return result
                self.restores.append((kernel, 'run' in vars(kernel), vars(kernel).get('run')))
                kernel.run = wrapped
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *unused):
        for kernel, had_run, previous in reversed(self.restores):
            if had_run:
                kernel.run = previous
            else:
                del kernel.run
        self.restores.clear()
