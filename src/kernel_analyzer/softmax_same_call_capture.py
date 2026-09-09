"""Collect a complete softmax invocation for mechanism analysis.

The caller supplies an already source/launch-checked callable and its ordered
pointer arguments. This helper establishes pre/post timing, not kernel identity.
It is separate from the historical observer to preserve frozen running jobs.
"""
from kernel_analyzer.grouped_causal_softmax_reference import snapshot_pre_call
from kernel_analyzer.softmax_saved_state_diagnostic import evaluate


def call_and_capture(run, pointers, *, rows, width, scale, invoke, sink):
    """Invoke once; send independent snapshots to sink after that invocation.

    invoke(run) must call the supplied implementation with the declared buffers.
    Actual source identity and argument binding are the integration's duty.
    """
    before = snapshot_pre_call(pointers, rows=rows, width=width)
    calls = 0
    def once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls != 1:
            raise ValueError('A same-call record cannot combine multiple invocations')
        return run(*args, **kwargs)
    result = invoke(once)
    if calls != 1:
        raise ValueError('No implementation invocation observed')
    after = {name: pointers[name].detach().clone()
             for name in ('in_out_ptr0', 'out_ptr0', 'out_ptr1', 'out_ptr2')}
    diagnostic = evaluate(after['in_out_ptr0'].reshape(rows, width),
                          after['out_ptr0'].reshape(rows), after['out_ptr1'].reshape(rows),
                          scale=scale)
    sink(dict(pre_call_inputs=before, post_call_outputs=after, diagnostic=diagnostic,
              scope='SAME_CALL_BUFFER_SNAPSHOTS_REQUIRES_EXTERNAL_EXECUTION_BINDING'))
    return result
