"""Runtime validation for source-audited pure Triton embedding lookup."""

from pathlib import Path

from kernel_analyzer.embedding_lookup_reference import snapshot_pre_call
from kernel_analyzer.embedding_lookup_source import check_source


POINTER_NAMES = ("in_ptr0", "in_ptr1", "out_ptr0")


def observer_class(base, contracts, runtime_signature):
    class CheckedObserver(base):
        def __init__(self, **kwargs):
            if kwargs.get("allow_missing_symbols"):
                raise ValueError("Embedding source substitution is not allowed")
            modules = list(kwargs["modules"])
            kwargs["modules"] = modules
            self.embedding_restores = []
            self.embedding_runtime_sources = {}
            for symbol, contract in contracts.items():
                matches = [module for module in modules if hasattr(module, symbol)]
                if not matches:
                    raise ValueError("Executed embedding module is absent")
                compared = lambda value: {
                    key: item for key, item in value.items()
                    if key not in {"source_sha256", "runtime_binding_complete"}
                }
                runtime_sources = []
                for module in matches:
                    path = Path(module.__file__).resolve()
                    actual = check_source(path.read_text(), symbol)
                    if compared(actual) != compared(contract):
                        raise ValueError("Executed embedding contract differs")
                    runtime_sources.append({
                        "path": str(path),
                        "source_sha256": actual["source_sha256"],
                        "function_ast_sha256": actual["function_ast_sha256"],
                    })
                # TorchInductor may expose the same compiled kernel through a
                # graph wrapper and a split cache module.  Repeated, audited
                # copies are aliases rather than a new implementation.  Every
                # copy must still satisfy the complete source contract above.
                self.embedding_runtime_sources[symbol] = runtime_sources
            super().__init__(**kwargs)

        def __enter__(self):
            base_started = False
            try:
                seen_kernels = set()
                for module in self.modules:
                    for symbol, contract in contracts.items():
                        kernel = getattr(module, symbol, None)
                        if kernel is None or id(kernel) in seen_kernels:
                            continue
                        seen_kernels.add(id(kernel))
                        signature = [(str(name), str(kind)) for name, kind in runtime_signature(kernel)]
                        pointers = [name for name, kind in signature if kind.startswith("*")]
                        if pointers != list(POINTER_NAMES):
                            raise ValueError("Runtime embedding pointer ABI differs")
                        original = kernel.run

                        def checked(*args, _original=original, _contract=contract, **kwargs):
                            if len(args) < len(POINTER_NAMES) + 1:
                                raise ValueError("Runtime embedding arguments are incomplete")
                            if args[len(POINTER_NAMES)] != _contract["elements"]:
                                raise ValueError("Runtime embedding element count differs")
                            live = dict(zip(POINTER_NAMES, args[:len(POINTER_NAMES)]))
                            snapshot_pre_call(
                                live, tokens=_contract["tokens"], width=_contract["width"],
                                vocabulary_size=_contract["vocabulary_size"],
                            )
                            return _original(*args, **kwargs)

                        self.embedding_restores.append((kernel, original))
                        kernel.run = checked
                base_started = True
                return super().__enter__()
            except BaseException:
                import sys
                try:
                    if base_started:
                        super().__exit__(*sys.exc_info())
                finally:
                    self._restore_embedding()
                raise

        def _restore_embedding(self):
            for kernel, original in reversed(self.embedding_restores):
                kernel.run = original
            self.embedding_restores.clear()

        def __exit__(self, *args):
            try:
                return super().__exit__(*args)
            finally:
                self._restore_embedding()

    return CheckedObserver
