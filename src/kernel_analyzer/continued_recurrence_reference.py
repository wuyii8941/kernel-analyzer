"""Common-input reference for a segment starting from a stored FP32 state."""
from kernel_analyzer.decayed_recurrence_reference import evaluate


def dependency_hashes():
    import hashlib
    from pathlib import Path
    from kernel_analyzer import continued_recurrence_source, decayed_recurrence_reference
    return {str(Path(m.__file__).resolve()): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (continued_recurrence_source, decayed_recurrence_reference)}


def reference(metadata, candidate, contract):
    """Apply only to the declared output of a separately source-bound execution."""
    import torch
    if contract.get('reference_dependencies_sha256') != dependency_hashes():
        raise ValueError('Continued recurrence dependencies changed')
    pointer = contract.get('output_pointer')
    expected_pointers = [f'out_ptr{i}' for i in range(contract['outputs'])]
    if (contract.get('segment_input_kind') != 'FP32_PREVIOUS_STATE'
            or contract.get('output_pointers') != expected_pointers
            or pointer not in expected_pointers
            or metadata.get('formal_pointer') != pointer
            or metadata.get('symbol') != contract['symbol']
            or metadata.get('input_output_storage_aliases') != []):
        raise ValueError('Continued recurrence boundary differs')
    if (not isinstance(candidate, torch.Tensor) or candidate.dtype != torch.float32
            or not candidate.is_contiguous()
            or candidate.numel() != contract['channels']*contract['state_width']):
        raise ValueError('Continued recurrence output layout differs')
    result = from_runtime_pointers(metadata.get('runtime_pointers', {}),
        channels=contract['channels'], width=contract['state_width'],
        outputs=contract['outputs'], output_index=expected_pointers.index(pointer),
        time_start=contract['time_start'], sequence_length=contract['sequence_length'])
    if result.device != candidate.device:
        raise ValueError('Continued recurrence output device mismatch')
    return result.reshape(candidate.shape)


def from_runtime_pointers(pointers, *, channels, width, outputs, output_index,
                          time_start, sequence_length):
    import torch
    dimensions = (channels, width, outputs, output_index, time_start, sequence_length)
    if (any(not isinstance(v, int) or isinstance(v, bool) for v in dimensions)
            or min(channels, width, outputs) < 1
            or not 0 <= output_index < outputs
            or not outputs <= time_start < sequence_length):
        raise ValueError('Invalid continued recurrence dimensions')
    sizes = {0: channels*width, 1: channels*width,
             2: sequence_length*channels, 3: channels}
    for j in range(outputs):
        sizes[4+2*j] = channels
        sizes[5+2*j] = width
    values = {}
    device = None
    for i, size in sizes.items():
        tensor = pointers.get(f'in_ptr{i}')
        dtype = torch.float32 if i < 2 else torch.bfloat16
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype
                or tensor.numel() != size or not tensor.is_contiguous()
                or not torch.isfinite(tensor).all()):
            raise ValueError(f'Invalid continued recurrence pointer: in_ptr{i}')
        if device is None:
            device = tensor.device
        if tensor.device != device:
            raise ValueError('Continued recurrence device mismatch')
        values[i] = tensor.reshape(-1).float()
    times = values[2].reshape(sequence_length, channels)
    times = times[time_start-outputs+1:time_start+1].flip(0)
    injections = torch.stack([values[4+2*j][:, None]*values[5+2*j][None, :]
                              for j in range(outputs)])
    return evaluate(values[0].reshape(channels, width),
                    values[1].reshape(channels, width), times,
                    values[3], injections)[output_index]
