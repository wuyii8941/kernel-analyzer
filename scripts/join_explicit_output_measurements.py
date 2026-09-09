"""Recheck explicit-output records and attach them to the matching release only."""
import argparse
from pathlib import Path
from scripts.finalize_explicit_output_capture import finalize
from scripts.join_coverage_measurements import join, validate_retained_measurements
from scripts.run_numerical_coverage import read, save, sha

FAMILIES = {'depthwise-convolution-capture-v1': 'DEPTHWISE_CONV1D',
            'gelu-product-capture-v1': 'GELU_PRODUCT',
            'embedding-lookup-forward-capture-v1': 'EMBEDDING_LOOKUP',
            'grouped-causal-softmax-forward-capture-v1': 'GROUPED_CAUSAL_SOFTMAX',
            'selected-nll-capture-v1': 'SELECTED_NLL',
            'indexed-accumulation-capture-v1': 'INDEXED_ROW_ACCUMULATION'}


def apply(inventory, release, root):
    protocol_path = root/'raw/family_execution_protocol.json'
    protocol = read(protocol_path)
    family = FAMILIES[protocol['schema']]
    tasks = (release/'same_dtype_tasks.json.gz').resolve()
    hashes = {str(Path(k).resolve()): v for k, v in protocol['source_sha256'].items()}
    if hashes.get(str(tasks)) != sha(tasks):
        raise ValueError('Measurement release task inventory differs')
    retained = validate_retained_measurements(inventory)
    runtime = None
    if protocol['schema']=='indexed-accumulation-capture-v1':
        from scripts.finalize_indexed_accumulation import verify_runtime
        from scripts.finalize_numerical_family import finalize as finalize_indexed
        runtime = verify_runtime(root)
        report = finalize_indexed(root)
    elif protocol['schema']=='selected-nll-capture-v1':
        from scripts.finalize_selected_nll_capture import verify
        report = verify(root)
    else:
        report = finalize(root)
    records = []
    for original in report['records']:
        row = dict(original)
        if 'raw_path' in row:
            row['raw_artifact'] = row.pop('raw_path')
        records.append(row)
    result = join(inventory, release, dict(records=records, errors=[]))
    measured = {r['task_id'] for r in records if r['status'] in ('VERIFIED','RECORDED_MEASUREMENT_CHECKED')}
    for row in result['records']:
        if Path(row['release']).resolve()==release.resolve() and row['task_id'] in measured:
            refs = list(row.get('reference_candidates', []))
            if not any(r['family']==family for r in refs):
                refs.append(dict(family=family, source='EXPLICIT_OUTPUT_CAPTURE_PROTOCOL',
                                 protocol_sha256=sha(protocol_path)))
            row['reference_candidates'] = refs
    result['retained_measurement_integrity'] = dict(checked=retained)
    result['explicit_output_record_audit'] = dict(counts=report['counts'],
        protocol_sha256=sha(protocol_path),
        indexed_runtime_verification=runtime,
        scope='Saved record check; no new bias, population or independent execution claim')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inventory','release','root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('New output under /data1/tzh required')
    result=apply(read(a.inventory),a.release,a.root)
    result['input_inventory_sha256']=sha(a.inventory)
    save(a.output,result)
    print(result['runtime_counts'])


if __name__=='__main__':main()
