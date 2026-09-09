"""Observed state errors relative to a declared FP64 recurrence.

The last pre-cast candidate state is not observable from these outputs. Its
actual cast error must not be inferred from rounding the reference instead.
"""
from kernel_analyzer.forward_state_recurrence_reference import decode_inputs, evaluate


def analyze(inputs, outputs, *, steps, channels, state_width, packed_width,
            state_offset):
    import torch
    decoded = decode_inputs(inputs, steps=steps, channels=channels,
                            state_width=state_width, packed_width=packed_width,
                            state_offset=state_offset, dtype=torch.float64)
    reference = evaluate(**decoded)
    if set(outputs) != {f'out_ptr{i}' for i in range(steps)}:
        raise ValueError('Complete ordered recurrence outputs required')
    observed = []
    for i in range(steps):
        tensor = outputs[f'out_ptr{i}']
        expected = torch.bfloat16 if i == steps-1 else torch.float32
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != expected
                or tensor.numel() != channels*state_width
                or not tensor.is_contiguous()
                or tensor.device != decoded['log_rate'].device
                or not torch.isfinite(tensor).all()):
            raise ValueError(f'Invalid recurrence output {i}')
        observed.append(tensor.detach().reshape(channels, state_width).double().clone())
    observed = torch.stack(observed)
    error = observed-reference['states']
    previous_error = torch.cat([torch.zeros_like(error[:1]), error[:-1]])
    # This residual includes coefficient, injection, fused arithmetic and write
    # differences. It does not identify those sources individually.
    local_remainder = error-reference['decay']*previous_error
    rebuilt, state = [], torch.zeros_like(error[0])
    for i in range(steps):
        state = reference['decay'][i]*state + local_remainder[i]
        rebuilt.append(state)
    rounded_reference = reference['states'][-1].to(torch.bfloat16).double()
    return dict(
        reference_states=reference['states'], observed_states=observed,
        state_difference=error, reference_decay=reference['decay'],
        combined_local_remainder=local_remainder,
        reconstruction_max_abs=(torch.stack(rebuilt)-error).abs().max(),
        per_step_mean=error.mean(dim=(1, 2)),
        per_step_rms=error.square().mean(dim=(1, 2)).sqrt(),
        final_reference_cast_difference=rounded_reference-reference['states'][-1],
        final_difference_from_rounded_reference=observed[-1]-rounded_reference,
        actual_final_candidate_cast_error_identified=False,
        scope='Observed outputs relative to FP64 reference; coordinate means are not state-population bias',
        new_mechanism_confirmed=False)
