"""Explicit time-window decoding for a first recurrence segment.

Reuses the existing mathematical recurrence. This helper alone does not prove
source identity or provide training integration.
"""
from kernel_analyzer.decayed_recurrence_reference import from_runtime_pointers


def reference(metadata,candidate,contract):
    """Runtime boundary checks for an explicitly declared first segment."""
    import hashlib
    from pathlib import Path
    import torch
    from kernel_analyzer import segmented_recurrence_source,decayed_recurrence_source
    expected={str(Path(m.__file__).resolve()):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
              for m in (segmented_recurrence_source,decayed_recurrence_source)}
    if contract.get('checker_dependencies_sha256')!=expected:
        raise ValueError('Segmented recurrence checker dependencies changed')
    pointer=contract.get('output_pointer')
    if (contract.get('segment_input_kind')!='BF16_OUTER_PRODUCT'
            or pointer not in contract['output_pointers']
            or metadata.get('formal_pointer')!=pointer
            or metadata.get('symbol')!=contract['symbol']
            or metadata.get('input_output_storage_aliases')!=[]):
        raise ValueError('Segmented recurrence boundary differs')
    if (candidate.dtype!=torch.float32 or not candidate.is_contiguous()
            or candidate.numel()!=contract['channels']*contract['state_width']):
        raise ValueError('Segmented recurrence output layout differs')
    result=first_segment(metadata.get('runtime_pointers',{}),channels=contract['channels'],
        width=contract['state_width'],outputs=contract['outputs'],
        output_index=contract['output_pointers'].index(pointer),time_start=contract['time_start'],
        sequence_length=contract['sequence_length'])
    if result.device!=candidate.device: raise ValueError('Segmented recurrence device mismatch')
    return result.reshape(candidate.shape)


def first_segment(pointers,*,channels,width,outputs,output_index,time_start,sequence_length):
    import torch
    if (not isinstance(time_start,int) or isinstance(time_start,bool)
            or not isinstance(sequence_length,int) or isinstance(sequence_length,bool)
            or min(channels,width,outputs)<1 or not outputs<=time_start<sequence_length):
        raise ValueError('Invalid first-segment time window')
    times=pointers.get('in_ptr3')
    if (not isinstance(times,torch.Tensor) or times.dtype!=torch.bfloat16
            or times.numel()!=sequence_length*channels or not times.is_contiguous()
            or not torch.isfinite(times).all()):
        raise ValueError('Full time-input layout differs')
    selected=dict(pointers)
    # The shared decoder excludes row zero, then reverses the remaining rows.
    # Retain the preceding row so its semantics remain exactly unchanged.
    selected['in_ptr3']=times.reshape(sequence_length,channels)[time_start-outputs:time_start+1]
    return from_runtime_pointers(selected,channels=channels,width=width,outputs=outputs,
                                 output_index=output_index)
