"""Freeze source-checked conv/bias pairs and existing semantic task identities."""
import argparse
import ast
import hashlib
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha
from kernel_analyzer.convolution_bias_binding import associate
from kernel_analyzer.channel_bias_source import check_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--length', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('New output under /data1/tzh required')
    source = a.release/'trace/model__0_forward_segment0_executed/output_code.py'
    text = source.read_text()
    symbol = 'triton_poi_fused__unsafe_view_convolution_split_transpose_2'
    bias_contract = check_source(text, symbol, length=a.length+3)
    calls = read(a.release/'inventory.json.gz')['runtime_call_audit']['rows']
    tasks = read(a.release/'same_dtype_tasks.json.gz')['rows']
    by_task = {t['task_id']:t for t in tasks}
    if len(by_task) != len(tasks): raise ValueError('Duplicate task IDs')
    records = []
    for parent in ast.walk(ast.parse(text)):
        for _, body in ast.iter_fields(parent):
            if not isinstance(body, list): continue
            for i, n in enumerate(body):
                if (not isinstance(n, ast.Assign) or not isinstance(n.value, ast.Call)
                        or ast.unparse(n.value.func) != 'extern_kernels.convolution'): continue
                pair = associate(body, i, symbol)
                if pair['numel'] != 1536*(a.length+3): raise ValueError('Bias element count differs')
                matches = [r for r in calls if r['source_path'] == 'model__0_forward_segment0_executed/output_code.py'
                           and r['source_line'] == n.lineno and r.get('function') == 'extern_kernels.convolution']
                if len(matches) != 1: raise ValueError('Ambiguous census convolution call')
                task = by_task[matches[0]['compute_region_id']+':output_0']
                closures = [by_task[t] for t in task.get('closed_by_semantic_endpoint_tasks', [])]
                closures = [t for t in closures if t.get('symbol') == symbol
                            and t.get('formal_pointer') == 'in_out_ptr0'
                            and t.get('status') == 'EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT']
                if len(closures) != 1: raise ValueError('Missing unique recorded bias closure')
                closure = closures[0]
                bias_calls = [r for r in calls if r.get('compute_region_id') == closure['candidate_region_id']
                              and r.get('source_path') == 'model__0_forward_segment0_executed/output_code.py'
                              and r.get('source_line') == pair['bias_line']
                              and r.get('function') == symbol+'.run']
                if len(bias_calls) != 1: raise ValueError('Closure points to another bias invocation')
                for row in (matches[0], bias_calls[0]):
                    line = text.splitlines()[row['source_line']-1].strip()
                    if hashlib.sha256(line.encode()).hexdigest() != row['source_line_sha256']:
                        raise ValueError('Recorded call line digest differs')
                records.append(dict(pair, convolution_task_id=task['task_id'],
                    closed_task_id=closure['task_id'], exact_aot_endpoint_id=closure['exact_aot_endpoint_id'],
                    convolution_line_sha256=matches[0]['source_line_sha256'],
                    bias_line_sha256=bias_calls[0]['source_line_sha256'],
                    parameter_binding_status='NOT_YET_BOUND', runtime_measurement_complete=False))
    if not records: raise ValueError('No convolution pairs found')
    base = Path(__file__).resolve().parents[1]
    deps = [source, a.release/'inventory.json.gz', a.release/'same_dtype_tasks.json.gz', Path(__file__)]
    deps += [base/'src/kernel_analyzer'/name for name in (
        'convolution_bias_binding.py', 'channel_bias_source.py', 'depthwise_conv1d_source.py',
        'depthwise_conv1d_reference.py', 'convolution_bias_capture.py')]
    save(a.output, dict(schema='convolution-bias-source-pairs-v1', records=records,
         bias_contract=bias_contract, numerical_results_read=False,
         source_sha256={str(p.resolve()):sha(p) for p in deps},
         scope='Source association and recorded closure; runtime identity and parameter reach pending'))
    print(dict(source_pairs=len(records)))


if __name__ == '__main__': main()
