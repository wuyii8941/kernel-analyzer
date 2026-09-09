"""Pre-call operands for indexed accumulation using the existing census binding.

Separate subclass preserves currently frozen measurement protocols. A new
capture must explicitly select and record this observer; no silent fallback.
"""
import torch
from scripts.same_dtype_semantic_observer import SameDtypeSemanticCandidateObserver
from scripts.generated_nontriton_fp32_observer import _AttributeProxy


def snapshot_indexed_inputs(buffer, indices, values, accumulate):
    if accumulate is not True or not isinstance(indices,(list,tuple)) or len(indices)!=1:
        raise ValueError('Only single-index accumulate=True is declared')
    index=indices[0]
    if (not isinstance(index,torch.Tensor) or index.dtype!=torch.int64 or index.ndim<1
            or buffer.ndim!=2 or buffer.dtype!=torch.float32 or values.dtype!=torch.float32
            or values.shape!=(*index.shape,buffer.shape[1])
            or any(x.device!=buffer.device for x in (index,values))):
        raise ValueError('Indexed row accumulation contract differs')
    return dict(initial=buffer.detach().clone(),indices=index.detach().clone(),values=values.detach().clone())


class IndexedAccumulationObserver(SameDtypeSemanticCandidateObserver):
    def _install_direct_aten(self):
        for module in self.modules:
            namespace=getattr(module,'aten',None)
            original=getattr(namespace,'index_put_',None) if namespace is not None else None
            if not callable(original): continue

            def wrapped(buffer,indices,values,accumulate=False,_original=original):
                filename,line,digest=self._source_identity()
                key=('DIRECT_ATEN',digest)
                if key not in self.nontriton_rows:
                    return _original(buffer,indices,values,accumulate)
                row=self._take_nontriton(*key)
                operands=snapshot_indexed_inputs(buffer,indices,values,accumulate)
                result=_original(buffer,indices,values,accumulate)
                self._emit_nontriton(row,buffer,'mutated_output_0',dict(
                    indexed_operands=operands,accumulate=True,
                    reference_operand_capture='PRE_INVOCATION_CLONE',
                    executing_filename=filename,executing_line=line,source_line_sha256=digest))
                return result

            module.aten=_AttributeProxy(namespace,{'index_put_':wrapped})
            self.nontriton_restores.append(lambda target=module,value=namespace:setattr(target,'aten',value))
