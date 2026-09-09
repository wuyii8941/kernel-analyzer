"""Checked pre-call adapter for the existing semantic endpoint observer."""
from pathlib import Path
from kernel_analyzer.residual_rms_forward_source import check_source
from kernel_analyzer.residual_rms_forward_reference import snapshot_pre_call


def observer_class(base, contracts, runtime_signature):
    class CheckedObserver(base):
        def __init__(self, **kwargs):
            if kwargs.get('allow_missing_symbols'):
                raise ValueError('Source substitution is not allowed')
            modules = list(kwargs['modules'])
            kwargs['modules'] = modules
            self.precheck_restores = []
            for symbol, contract in contracts.items():
                found = [check_source(Path(module.__file__).read_text(), symbol)
                         for module in modules if hasattr(module, symbol)]
                if len(found) != 1 or found[0]['function_ast_sha256'] != contract['function_ast_sha256']:
                    raise ValueError('Actual RMS function differs from declared source')
            super().__init__(**kwargs)

        def __enter__(self):
            import torch
            base_started = False
            try:
                for module in self.modules:
                    for symbol, contract in contracts.items():
                        kernel = getattr(module, symbol, None)
                        if kernel is None:
                            continue
                        original = kernel.run
                        names = [str(name) for name, annotation in runtime_signature(kernel)
                                 if str(annotation).startswith('*')]
                        def checked(*args, _original=original, _names=names, _contract=contract, **kwargs):
                            tensors = [value for value in args if isinstance(value, torch.Tensor)]
                            if len(tensors) != len(_names):
                                raise ValueError('RMS runtime pointer ABI differs')
                            # Validation must see original storage, not independent clones.
                            snapshot_pre_call(dict(zip(_names, tensors)),
                                rows=_contract['rows'], width=_contract['width'])
                            return _original(*args, **kwargs)
                        self.precheck_restores.append((kernel, original))
                        kernel.run = checked
                base_started = True
                return super().__enter__()
            except BaseException:
                import sys
                try:
                    if base_started:
                        super().__exit__(*sys.exc_info())
                finally:
                    self._restore_prechecks()
                raise

        def _restore_prechecks(self):
            for kernel, original in reversed(self.precheck_restores):
                kernel.run = original
            self.precheck_restores.clear()

        def __exit__(self, *args):
            try:
                return super().__exit__(*args)
            finally:
                self._restore_prechecks()
    return CheckedObserver
