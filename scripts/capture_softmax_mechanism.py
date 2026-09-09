"""Capture actual same-call softmax buffers; no update or population claim."""
import argparse
import os
from pathlib import Path


def main(*, family='softmax'):
    if family not in ('softmax', 'forward_recurrence'):
        raise ValueError('Unsupported mechanism capture family')
    p = argparse.ArgumentParser(description=f'Capture actual same-call {family} buffers; no update or population claim.')
    for name in ('model', 'input-bank', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--architecture', default='qwen' if family == 'softmax' else 'mamba')
    p.add_argument('--device', required=True)
    p.add_argument('--states', type=int, default=2)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('New output under /data1/tzh required')
    from scripts.launch_detached_experiment import DATA_CACHE_ENV
    os.environ.update(DATA_CACHE_ENV)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from torch._inductor.codecache import PyCodeCache
    from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime
    from scripts.run_generated_fp32_screen import load_model
    from scripts.run_numerical_coverage import read, save, sha
    if family == 'softmax':
        from scripts.scan_grouped_causal_softmax_sources import scan
        from kernel_analyzer.softmax_mechanism_observer import Observer
    else:
        from scripts.scan_recurrence_sources import scan as scan_definitions
        from kernel_analyzer.forward_state_recurrence_source import check_source
        from kernel_analyzer.forward_recurrence_observer import Observer
        def scan(source):
            return scan_definitions(source, checker=check_source)
    bank = read(a.input_bank)
    states = bank.get('states', bank.get('records', []))[:a.states]
    if a.states <= 0 or len(states) != a.states:
        raise ValueError('Insufficient declared inputs')
    tokens = [s.get('input_ids', s.get('token_ids')) for s in states]
    if any(not isinstance(t, list) or not t for t in tokens) or len({len(t) for t in tokens}) != 1:
        raise ValueError('Fixed nonempty sequence shape required')
    dependencies = [Path(__file__), a.input_bank, a.model/'config.json']
    base = Path(__file__).resolve().parents[1]
    dependencies += [base/'src/kernel_analyzer'/n for n in (
        'softmax_mechanism_observer.py', 'softmax_same_call_capture.py',
        'softmax_saved_state_diagnostic.py', 'grouped_causal_softmax_source.py',
        'grouped_causal_softmax_reference.py')]
    dependencies += [base/'scripts'/n for n in ('qwen_candidate_step.py',
        'run_generated_fp32_screen.py', 'scan_grouped_causal_softmax_sources.py',
        'scan_recurrence_sources.py')]
    if family == 'forward_recurrence':
        dependencies += [base/'src/kernel_analyzer'/n for n in (
            'forward_state_recurrence_source.py', 'forward_state_recurrence_reference.py',
            'forward_recurrence_observer.py', 'forward_recurrence_diagnostic.py',
            'decayed_recurrence_reference.py')]
        dependencies.append(base/'scripts/capture_forward_recurrence_mechanism.py')
    save(a.output/'protocol.json', dict(schema=f'{family}-mechanism-capture-v1',
        mechanism_family=family,
        architecture=a.architecture, model=str(a.model), device=a.device,
        selected_states=states, source_sha256={str(x.resolve()):sha(x) for x in dependencies},
        scope='FIXED_INPUT_SAME_CALL_DIAGNOSTIC', optimizer_update_measured=False,
        population_bias_established=False, numerical_result_selection=False))
    configure_candidate_runtime(24000)
    device = torch.device(a.device)
    model = load_model(a.architecture, a.model, device)
    start = len(PyCodeCache.modules)
    compiled = torch.compile(LossStep(model), backend='inductor', fullgraph=True, dynamic=False)
    warm = torch.tensor([tokens[0]], dtype=torch.long, device=device)
    compiled(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    contracts, sources = {}, []
    for module in modules:
        path = Path(module.__file__)
        checked = [r for r in scan(path.read_text()) if r['status'] == 'SOURCE_CHECKED']
        for row in checked:
            if row['symbol'] in contracts:
                raise ValueError('Ambiguous loaded definition')
            contracts[row['symbol']] = row['contract']
        if checked:
            sources.append(dict(path=str(path), sha256=sha(path), text=path.read_text()))
    if not contracts:
        raise ValueError('No checked forward implementation found; no fallback')
    save(a.output/'loaded_sources.json', dict(contracts=contracts, sources=sources,
                                            binary_identity_verified=False))
    for i, token_ids in enumerate(tokens):
        model.zero_grad(set_to_none=True)
        configure_candidate_runtime(24000+i)
        count = [0]
        def sink(record):
            def cpu(value):
                if isinstance(value, torch.Tensor): return value.detach().cpu()
                if isinstance(value, dict): return {k:cpu(v) for k,v in value.items()}
                return value
            path = a.output/f'state_{i:03d}_call_{count[0]:03d}.pt'
            with path.open('xb') as stream: torch.save(cpu(record), stream)
            count[0] += 1
        with Observer(modules, contracts, sink) as observation:
            loss = compiled(torch.tensor([token_ids], dtype=torch.long, device=device))
            loss.backward()
        torch.cuda.synchronize(device)
        if not count[0]: raise ValueError('Checked kernel was not invoked')
        save(a.output/f'state_{i:03d}.json', dict(calls=count[0], loss=float(loss.detach()),
             symbol_counts=observation.counts, training_effect_not_established=True))
        print(dict(state=i, captured_calls=count[0]), flush=True)
    save(a.output/'completion.json', dict(status='SAME_CALL_DIAGNOSTIC_CAPTURED',
        states=len(states), complete_training_measurement=False, new_mechanism_confirmed=False))


if __name__ == '__main__':
    main()
