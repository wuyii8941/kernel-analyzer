"""Runtime checks for source-audited grouped causal softmax outputs.

The shared semantic observer owns endpoint selection and replacement.  This
adapter only verifies that the live Triton function, pointer ABI, scalar
dimensions, and pre-call storage still satisfy the reviewed family contract.
"""

from pathlib import Path

from kernel_analyzer.grouped_causal_softmax_reference import snapshot_pre_call
from kernel_analyzer.grouped_causal_softmax_source import check_source


POINTER_NAMES = ("in_out_ptr0", "in_ptr0", "out_ptr0", "out_ptr1", "out_ptr2")


def observer_class(base, contracts, runtime_signature):
    class CheckedObserver(base):
        def __init__(self, **kwargs):
            if kwargs.get("allow_missing_symbols"):
                raise ValueError("Grouped softmax source substitution is not allowed")
            modules = list(kwargs["modules"])
            kwargs["modules"] = modules
            self.grouped_softmax_restores = []
            for symbol, contract in contracts.items():
                matches = [module for module in modules if hasattr(module, symbol)]
                if len(matches) != 1:
                    raise ValueError("Unique executed grouped softmax module required")
                actual = check_source(Path(matches[0].__file__).read_text(), symbol)
                compared = lambda value: {
                    key: item for key, item in value.items()
                    if key not in {"source_sha256", "runtime_binding_complete"}
                }
                if compared(actual) != compared(contract):
                    raise ValueError("Executed grouped softmax contract differs")
            super().__init__(**kwargs)

        def __enter__(self):
            base_started = False
            try:
                for module in self.modules:
                    for symbol, contract in contracts.items():
                        kernel = getattr(module, symbol, None)
                        if kernel is None:
                            continue
                        signature = [(str(name), str(kind)) for name, kind in runtime_signature(kernel)]
                        pointer_names = [name for name, kind in signature if kind.startswith("*")]
                        if pointer_names != list(POINTER_NAMES):
                            raise ValueError("Runtime grouped softmax pointer ABI differs")
                        original = kernel.run

                        def checked(*args, _original=original, _contract=contract, **kwargs):
                            if len(args) < len(POINTER_NAMES) + 2:
                                raise ValueError("Runtime grouped softmax arguments are incomplete")
                            xnumel, r0_numel = args[len(POINTER_NAMES):len(POINTER_NAMES) + 2]
                            if xnumel != _contract["rows"] or r0_numel != _contract["width"]:
                                raise ValueError("Runtime grouped softmax dimensions differ")
                            pointers = dict(zip(POINTER_NAMES, args[:len(POINTER_NAMES)]))
                            # Validate live storage before the in-place score write.
                            snapshot_pre_call(
                                pointers, rows=_contract["rows"], width=_contract["width"]
                            )
                            return _original(*args, **kwargs)

                        self.grouped_softmax_restores.append((kernel, original))
                        kernel.run = checked
                base_started = True
                return super().__enter__()
            except BaseException:
                import sys
                try:
                    if base_started:
                        super().__exit__(*sys.exc_info())
                finally:
                    self._restore_grouped_softmax()
                raise

        def _restore_grouped_softmax(self):
            for kernel, original in reversed(self.grouped_softmax_restores):
                kernel.run = original
            self.grouped_softmax_restores.clear()

        def __exit__(self, *args):
            try:
                return super().__exit__(*args)
            finally:
                self._restore_grouped_softmax()

    return CheckedObserver
