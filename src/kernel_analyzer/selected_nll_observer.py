"""Validate loaded NLL code and ABI around the existing semantic observer."""
from pathlib import Path
from kernel_analyzer.selected_nll_source import check_source, SIGNATURE
from kernel_analyzer.selected_nll_runtime import snapshot_inputs


def observer_class(base, contracts, runtime_signature):
    class CheckedObserver(base):
        def __init__(self, **kwargs):
            if kwargs.get('allow_missing_symbols'):
                raise ValueError('NLL source substitution is not allowed')
            modules = list(kwargs['modules'])
            kwargs['modules'] = modules
            self.nll_restores = []
            for symbol, contract in contracts.items():
                matches = [m for m in modules if hasattr(m, symbol)]
                if len(matches) != 1:
                    raise ValueError('Unique executed NLL module required')
                actual = check_source(Path(matches[0].__file__).read_text(), symbol)
                if actual['body_sha256'] != contract['body_sha256']:
                    raise ValueError('Executed NLL definition differs')
            super().__init__(**kwargs)

        def __enter__(self):
            started = False
            try:
                for module in self.modules:
                    for symbol, contract in contracts.items():
                        kernel = getattr(module, symbol, None)
                        if kernel is None:
                            continue
                        signature = [(str(n), str(t)) for n, t in runtime_signature(kernel)]
                        pointers = [(n, t) for n, t in signature if t.startswith('*')]
                        expected = [(n, t) for n, t in SIGNATURE.items() if t.startswith('*')]
                        if pointers != expected:
                            raise ValueError('Runtime NLL pointer types or ordering differ')
                        original = kernel.run
                        def checked(*args, _original=original, _contract=contract, **kwargs):
                            if len(args) < 8 or tuple(args[6:8]) != (_contract['tokens'], _contract['vocabulary']):
                                raise ValueError('Runtime NLL dimensions differ')
                            # The base observer independently clones pre-call operands
                            # for its reference callback. Check aliasing on live inputs.
                            snapshot_inputs(dict(zip(list(SIGNATURE)[:6], args[:6])), _contract)
                            return _original(*args, **kwargs)
                        self.nll_restores.append((kernel, original))
                        kernel.run = checked
                started = True
                return super().__enter__()
            except BaseException:
                import sys
                try:
                    if started:
                        super().__exit__(*sys.exc_info())
                finally:
                    self._restore_nll()
                raise

        def _restore_nll(self):
            for kernel, original in reversed(self.nll_restores):
                kernel.run = original
            self.nll_restores.clear()

        def __exit__(self, *args):
            try:
                return super().__exit__(*args)
            finally:
                self._restore_nll()
    return CheckedObserver
