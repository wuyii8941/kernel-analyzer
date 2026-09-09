"""Observe checked kernel calls without replacing outputs or tuning configs.

Loaded Python source matching is recorded; compiled-binary identity and launch
geometry remain separate requirements. This does not sign an equivalence claim.
"""
from pathlib import Path
from kernel_analyzer.grouped_causal_softmax_source import check_source
from kernel_analyzer.softmax_same_call_capture import call_and_capture


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
                    raise ValueError('Missing or ambiguous loaded symbol')
                module = modules[0]
                checked = check_source(Path(module.__file__).read_text(), symbol)
                if checked != contract:
                    raise ValueError('Loaded source differs from declared contract')
                kernel = getattr(module, symbol)
                expected = dict(in_out_ptr0='*bf16', in_ptr0='*i64', out_ptr0='*fp32',
                                out_ptr1='*fp32', out_ptr2='*bf16', xnumel='i32', r0_numel='i32')
                if list(kernel.triton_meta['signature'].items()) != list(expected.items()):
                    raise ValueError('Actual ordered argument signature differs')
                original = kernel.run
                had_run = 'run' in vars(kernel)
                previous = vars(kernel).get('run')
                def wrapped(*args, _original=original, _contract=contract, _symbol=symbol, **kwargs):
                    if len(args) != 7 or set(kwargs) != {'stream'}:
                        raise ValueError('Unexpected invocation arguments')
                    if (type(args[5]) is not int or type(args[6]) is not int
                            or args[5] != _contract['rows'] or args[6] != _contract['width']):
                        raise ValueError('Invocation dimensions differ')
                    pointers = dict(zip(('in_out_ptr0', 'in_ptr0', 'out_ptr0', 'out_ptr1', 'out_ptr2'), args[:5]))
                    index = self.counts.get(_symbol, 0)
                    def emit(record):
                        self.sink(dict(record, symbol=_symbol, invocation_index=index,
                            function_ast_sha256=_contract['function_ast_sha256'],
                            binary_identity_verified=False, resolved_launch_verified=False))
                    result = call_and_capture(_original, pointers,
                        rows=_contract['rows'], width=_contract['width'], scale=_contract['scale'],
                        invoke=lambda run: run(*args, **kwargs), sink=emit)
                    self.counts[_symbol] = index + 1
                    return result
                self.restores.append((kernel, had_run, previous))
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
