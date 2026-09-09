import pytest
from kernel_analyzer.capture_cost import measured_capture


def test_metrics_do_not_replace_callback_return_or_claim_training_speed():
    metrics=[]
    value=object()
    assert measured_capture(lambda:value,device='cpu',emit=metrics.append) is value
    assert metrics[0]['execution_status']=='CALLBACK_RETURNED'
    assert metrics[0]['elapsed_seconds']>=0
    assert metrics[0]['process_cpu_seconds']>=0
    assert metrics[0]['cuda_peak_allocated_bytes'] is None
    assert not metrics[0]['training_throughput_claim']


def test_failure_retained_and_not_converted_to_success():
    metrics=[]
    def fail(): raise RuntimeError('original failure')
    with pytest.raises(RuntimeError,match='original failure'):
        measured_capture(fail,device='cpu',emit=metrics.append)
    assert metrics[0]['execution_status']=='FAILED'
