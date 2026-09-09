"""Extend the shared observer only for source-bound external convolution."""
from pathlib import Path
from scripts.same_dtype_semantic_observer import SameDtypeSemanticCandidateObserver
from scripts.generated_nontriton_fp32_observer import _AttributeProxy
from kernel_analyzer.depthwise_conv1d_source import check_call


def observer_class(contracts):
    class ConvolutionObserver(SameDtypeSemanticCandidateObserver):
        def _install_externals(self):
            for module in self.modules:
                namespace = getattr(module, 'extern_kernels', None)
                original = getattr(namespace, 'convolution', None)
                if not callable(original): continue
                def wrapped(*args, _original=original, **kwargs):
                    filename, line, digest = self._source_identity()
                    key = ('EXTERN', digest)
                    if key not in self.nontriton_rows:
                        return _original(*args, **kwargs)
                    row = self._take_nontriton(*key)
                    contract = contracts.get(digest)
                    if contract is None:
                        raise ValueError('Selected convolution lacks frozen source binding')
                    source_line = Path(filename).read_text().splitlines()[line-1].strip()
                    # Assignment output name is irrelevant to arithmetic, but
                    # the whole call AST must match the declared call.
                    import ast
                    statement = ast.parse(source_line).body[0]
                    if not isinstance(statement, ast.Assign):
                        raise ValueError('Actual convolution call is not an assignment')
                    actual = check_call(ast.unparse(statement.value))
                    if actual != contract['convolution']:
                        raise ValueError('Actual convolution expression differs')
                    from scripts.same_dtype_semantic_observer import snapshot_external_inputs
                    before_args, before_kwargs = snapshot_external_inputs(args, kwargs)
                    output = _original(*args, **kwargs)
                    self._emit_nontriton(row, output, 'output_0', dict(
                        external_symbol='convolution', runtime_args=before_args,
                        runtime_kwargs=before_kwargs, reference_operand_capture='PRE_INVOCATION_CLONE',
                        executing_filename=filename, executing_line=line, source_line_sha256=digest,
                        convolution_contract=actual))
                    return output
                module.extern_kernels = _AttributeProxy(namespace, {'convolution': wrapped})
                self.nontriton_restores.append(lambda m=module, ns=namespace: setattr(m,'extern_kernels',ns))
    return ConvolutionObserver
