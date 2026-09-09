"""Measure analysis execution cost, not model training performance."""
import resource
import time
import torch


def measured_capture(callback, *, device, emit):
    started = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_SELF)
    cuda = torch.device(device).type == 'cuda'
    ready = False
    status = 'FAILED'
    try:
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            ready = True
        result = callback()
        if cuda:
            torch.cuda.synchronize(device)
        status = 'CALLBACK_RETURNED'
        return result
    finally:
        after = resource.getrusage(resource.RUSAGE_SELF)
        metrics = {
            'schema': 'training-analysis-capture-cost-v1', 'execution_status': status,
            'elapsed_seconds': time.perf_counter() - started,
            'process_cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
            'process_peak_rss_kib': after.ru_maxrss,
            'device': str(device), 'torch_version': torch.__version__,
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if ready else None,
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if ready else None,
            'scope': 'Capture including load, compile, replay, measurement and serialization; process RSS lifetime high-water mark; PyTorch CUDA allocator only',
            'training_throughput_claim': False,
            'numerical_verdict': 'NOT_INFERRED_FROM_EXECUTION_RETURN',
        }
        emit(metrics)
