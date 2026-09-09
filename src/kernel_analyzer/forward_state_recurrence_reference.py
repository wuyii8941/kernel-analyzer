"""Reference for forward recurrence with a shared positive time step.

This is a mathematical input interface, not a completed Triton source adapter.
Unlike the backward recurrence, injection depends on the same dt as decay.
"""


def decode_inputs(pointers, *, steps, channels, state_width, packed_width,
                  state_offset, dtype=None):
    """Decode the audited contiguous pointer layout, not a source match.

    Time inputs are time-major; signal storage is channel-major. Packed
    storage contains other inputs that must not enter the recurrence.
    Original inputs are copied before any reference computation.
    """
    import torch
    dimensions = (steps, channels, state_width, packed_width)
    if (any(type(x) is not int or x <= 0 for x in dimensions)
            or type(state_offset) is not int or state_offset < 0
            or state_offset + state_width > packed_width
            or set(pointers) != {f'in_ptr{i}' for i in range(5)}):
        raise ValueError('Invalid forward pointer layout')
    dtype = torch.float32 if dtype is None else dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('Reference accumulation must be FP32 or FP64')
    sizes = (channels*state_width, steps*channels, channels,
             steps*packed_width, channels*steps)
    values = [pointers[f'in_ptr{i}'] for i in range(5)]
    for i, (value, size) in enumerate(zip(values, sizes)):
        expected = torch.float32 if i == 0 else torch.bfloat16
        if (not isinstance(value, torch.Tensor) or value.numel() != size
                or value.dtype != expected or not value.is_contiguous()
                or value.device != values[0].device
                or not torch.isfinite(value).all()):
            raise ValueError(f'Invalid input pointer {i}')
    copied = [v.detach().reshape(-1).to(dtype=dtype).clone() for v in values]
    return dict(log_rate=copied[0].reshape(channels, state_width),
                time_inputs=copied[1].reshape(steps, channels),
                bias=copied[2],
                input_state=copied[3].reshape(steps, packed_width)[
                    :, state_offset:state_offset+state_width].contiguous(),
                input_signal=copied[4].reshape(channels, steps).T.contiguous())


def evaluate(log_rate, time_inputs, bias, input_state, input_signal):
    """Shapes: rate[C,N], time[T,C], bias[C], state[T,N], signal[T,C]."""
    import torch
    from kernel_analyzer.decayed_recurrence_reference import evaluate as recurrence
    if (log_rate.ndim != 2 or time_inputs.ndim != 2 or time_inputs.shape[0] <= 0
            or time_inputs.shape[1] != log_rate.shape[0]
            or bias.shape != (log_rate.shape[0],)
            or input_state.shape != (time_inputs.shape[0], log_rate.shape[1])
            or input_signal.shape != time_inputs.shape
            or log_rate.dtype not in (torch.float32, torch.float64)
            or any(t.dtype != log_rate.dtype or t.device != log_rate.device
                   for t in (time_inputs, bias, input_state, input_signal))
            or any(not torch.isfinite(t).all()
                   for t in (log_rate, time_inputs, bias, input_state, input_signal))):
        raise ValueError('Invalid forward recurrence inputs')
    dt = torch.nn.functional.softplus(time_inputs + bias, beta=1, threshold=20)
    # Preserve declared multiplication order: (dt * input_state) * input_signal.
    injection = (dt[:, :, None] * input_state[:, None, :]) * input_signal[:, :, None]
    states = recurrence(torch.zeros_like(log_rate), log_rate, time_inputs, bias, injection)
    decay = torch.exp(-torch.exp(log_rate)[None] * dt[:, :, None])
    return dict(states=states, positive_time=dt, decay=decay, injection=injection,
                runtime_binding_complete=False, population_bias_proved=False)
