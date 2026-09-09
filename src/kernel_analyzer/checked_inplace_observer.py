"""Reusable source/launch validation around the existing semantic observer.

Family adapters supply reviewed source and operand checks; statistics and
candidate/reference execution remain the responsibility of the base observer.
"""
from pathlib import Path


def observer_class(base, contracts, runtime_signature, *, check_source,
                   signature, snapshot_inputs):
    class CheckedObserver(base):
        def __init__(self, **kwargs):
            if kwargs.get('allow_missing_symbols'):
                raise ValueError('Source substitution is not allowed')
            modules = list(kwargs['modules'])
            kwargs['modules'] = modules
            self.checked_restores = []
            for symbol, contract in contracts.items():
                matches = [m for m in modules if hasattr(m, symbol)]
                if len(matches) != 1:
                    raise ValueError('Unique executed module required')
                actual = check_source(Path(matches[0].__file__).read_text(), symbol)
                # Validate the entire contract, not only the source digest: a
                # stale shape or formula constant must not survive rebinding.
                compared = lambda value: {k: v for k, v in value.items() if k != 'source_sha256'}
                if compared(actual) != compared(contract):
                    raise ValueError('Executed source contract differs')
            super().__init__(**kwargs)

        def __enter__(self):
            started = False
            try:
                for module in self.modules:
                    for symbol, contract in contracts.items():
                        kernel = getattr(module, symbol, None)
                        if kernel is None:
                            continue
                        actual = [(str(n), str(t)) for n, t in runtime_signature(kernel)]
                        pointers = [(n, t) for n, t in actual if t.startswith('*')]
                        expected = [(n, t) for n, t in signature.items() if t.startswith('*')]
                        if pointers != expected:
                            raise ValueError('Runtime pointer types or ordering differ')
                        original = kernel.run
                        def checked(*args, _original=original, _contract=contract, **kwargs):
                            count = len(expected)
                            if len(args) < count+2 or tuple(args[count:count+2]) != (
                                    _contract['tokens'], _contract['vocabulary']):
                                raise ValueError('Runtime dimensions differ')
                            snapshot_inputs(dict(zip([n for n, _ in expected], args[:count])), _contract)
                            return _original(*args, **kwargs)
                        self.checked_restores.append((kernel, original))
                        kernel.run = checked
                started = True
                return super().__enter__()
            except BaseException:
                import sys
                try:
                    if started:
                        super().__exit__(*sys.exc_info())
                finally:
                    self._restore_checked()
                raise

        def _restore_checked(self):
            for kernel, original in reversed(self.checked_restores):
                kernel.run = original
            self.checked_restores.clear()

        def __exit__(self, *args):
            try:
                return super().__exit__(*args)
            finally:
                self._restore_checked()
    return CheckedObserver
