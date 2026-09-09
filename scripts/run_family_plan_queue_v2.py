#!/usr/bin/env python3
"""Version 2: extend the frozen queue with forward RMS capture support.

Version 1 remains unchanged for active hash-pinned jobs.
Run capacity-partitioned family plans through existing capture and verifier.

Never resumes a possibly live capture or overwrites an existing queue. A failed
job stops this queue, with its complete log retained for explicit diagnosis.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPLICIT_OUTPUT_FAMILIES = {
    'embedding-lookup-forward-capture-v1': (
        'run_embedding_lookup_capture.py', 'embedding_lookup'),
    'grouped-causal-softmax-forward-capture-v1': (
        'run_grouped_causal_softmax_capture.py', 'grouped_causal_softmax'),
}


def replace_argument(arguments, name, value):
    result = list(arguments)
    if result.count(name) != 1:
        raise ValueError('Expected exactly one argument: ' + name)
    index = result.index(name)
    if index + 1 == len(result):
        raise ValueError('Missing argument value: ' + name)
    result[index+1] = str(value)
    return result


def build_command(protocol, manifest, plan, root, spool):
    explicit = EXPLICIT_OUTPUT_FAMILIES.get(protocol.get('schema'))
    if explicit is not None:
        entrypoint, module_name = explicit
        module = __import__('scripts.' + 'run_' + module_name + '_capture', fromlist=['select'])
        module.select(read(manifest), read(plan)['cases'])
        arguments = protocol['capture_arguments']
        for name, value in (('--case-plan',plan), ('--output-dir',root/'legacy'),
                            ('--spool-dir',spool), ('--training-bias-profile-v2-output-dir',root/'raw')):
            arguments = replace_argument(arguments, name, value)
        return [sys.executable, str(ROOT/'scripts'/entrypoint),
                '--family-plan', str(manifest), *arguments]
    if protocol.get('schema') == 'residual-rms-forward-capture-v1':
        from scripts.run_residual_rms_forward_capture import select
        select(read(manifest), read(plan)['cases'])
        arguments = protocol['capture_arguments']
        for name, value in (('--case-plan',plan), ('--output-dir',root/'legacy'),
                            ('--spool-dir',spool), ('--training-bias-profile-v2-output-dir',root/'raw')):
            arguments = replace_argument(arguments, name, value)
        return [sys.executable, str(ROOT/'scripts/run_residual_rms_forward_capture.py'),
                '--family-plan', str(manifest), *arguments]
    if protocol.get('schema') == 'decayed-recurrence-capture-v1':
        from scripts.recurrence_queue_commands import build_command as recurrence_command
        return recurrence_command(protocol, manifest, plan, root, spool)
    if protocol.get('retain_small_update_vectors'):
        raise ValueError('Do not expand a selected small-vector mechanism replay into a new queue')
    arguments = protocol['capture_arguments']
    for name, value in (('--case-plan',plan), ('--output-dir',root/'legacy'),
                        ('--spool-dir',spool), ('--training-bias-profile-v2-output-dir',root/'raw')):
        arguments = replace_argument(arguments, name, value)
    extra = ['--reference-manifest',str(manifest),'--reference-variant',protocol['variant']]
    for field in ('parallel_measurement','train_only_declared_parameters'):
        if protocol.get(field): extra += ['--'+field.replace('_','-')]
    tolerance = protocol.get('local_tolerance_baseline')
    if tolerance and tolerance.get('rtol') is not None:
        extra += ['--local-rtol',str(tolerance['rtol']),'--local-atol',str(tolerance['atol'])]
    entry = [sys.executable,str(ROOT/'scripts/run_row_reduction_capture.py')]
    if manifest.is_file():
        declared_manifest = read(manifest)
        additional = declared_manifest.get('additional_reference_declaration')
        if additional is not None:
            from scripts.run_declared_reference_family import declaration, validate_capture
            expected = declaration(additional['family'],additional['module'],additional['case_suffix'])
            validate_capture(declared_manifest,expected)
            if protocol.get('reference_family') != expected['family']:
                raise ValueError('Prototype and additional reference family differ')
            entry = [sys.executable,str(ROOT/'scripts/run_declared_reference_family.py'),
                     '--mode','capture','--new-family',expected['family'],
                     '--module',expected['module'],'--case-suffix='+expected['case_suffix']]
    return [*entry,*extra,*arguments]


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False)


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verification_command(protocol, root):
    if protocol.get('schema') == 'decayed-recurrence-capture-v1':
        script = 'finalize_decayed_recurrence.py'
        extra = []
    elif protocol.get('schema') == 'residual-rms-forward-capture-v1':
        script = 'finalize_residual_rms_forward.py'
        extra = []
    elif protocol.get('schema') in EXPLICIT_OUTPUT_FAMILIES:
        script = 'finalize_explicit_output_capture.py'
        extra = []
    else:
        script = 'finalize_numerical_family.py'
        extra = ['--baselines-dir', str(root/'verified_baselines')]
    return [sys.executable, str(ROOT/'scripts'/script), '--root', str(root),
            '--output', str(root/'completion_verification.json'), *extra]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prototype-root',type=Path,required=True)
    p.add_argument('--reference-manifest',type=Path,required=True)
    p.add_argument('--plans',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--spool',type=Path,required=True)
    a=p.parse_args()
    for path in (a.output,a.spool):
        if path.exists() or not path.resolve().is_relative_to(Path('/data1/tzh')):
            p.error('Choose new output and spool directories under /data1/tzh')
    protocol_path=a.prototype_root/'raw/family_execution_protocol.json'
    prototype=read(protocol_path)
    manifest=a.reference_manifest.resolve()
    if sha(manifest) != prototype['source_sha256'].get(str(manifest)):
        raise ValueError('Reference manifest differs from prototype')
    for path,digest in prototype['source_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Prototype source changed; explicitly redeclare before queue: '+path)
    jobs=[]; seen=set()
    queue_sources=[Path(__file__)]
    if prototype.get('schema') == 'decayed-recurrence-capture-v1':
        queue_sources += [ROOT/'scripts'/name for name in
                          ('recurrence_queue_commands.py','finalize_decayed_recurrence.py')]
    if prototype.get('schema') == 'residual-rms-forward-capture-v1':
        queue_sources += [ROOT/'scripts/finalize_residual_rms_forward.py',
                          ROOT/'src/kernel_analyzer/forward_capture_audit.py',
                          ROOT/'scripts/preflight_capture_input.py']
    if prototype.get('schema') in EXPLICIT_OUTPUT_FAMILIES:
        entrypoint, _ = EXPLICIT_OUTPUT_FAMILIES[prototype['schema']]
        queue_sources += [ROOT/'scripts'/entrypoint,
                          ROOT/'scripts/finalize_explicit_output_capture.py',
                          ROOT/'src/kernel_analyzer/explicit_output_capture_audit.py']
    queue_hashes={str(path.resolve()):sha(path) for path in queue_sources}
    for i,plan in enumerate(a.plans):
        plan=plan.resolve(); cases=read(plan)['cases']
        ids={c['case_id'] for c in cases}
        if not ids or len(ids)!=len(cases) or seen & ids:
            raise ValueError('Empty or repeated cases in queue')
        seen |= ids
        root=a.output.resolve()/f'job_{i:03d}'
        jobs.append(dict(plan=str(plan),plan_sha256=sha(plan),case_ids=sorted(ids),
                         root=str(root),command=build_command(prototype,manifest,plan,root,a.spool.resolve()/f'job_{i:03d}')))
    from scripts.preflight_capture_input import check_command
    for job in jobs:
        job['input_preflight'] = check_command(job['command'])
    save(a.output/'queue_protocol.json',dict(schema='family-plan-queue-v2',jobs=jobs,
         prototype_protocol=str(protocol_path.resolve()),prototype_sha256=sha(protocol_path),
         source_sha256=prototype['source_sha256'], queue_script_sha256=sha(Path(__file__)),
         queue_source_sha256=queue_hashes,
         numerical_results_used_for_selection=False, measurement_complete=False))
    for job in jobs:
        for path,digest in queue_hashes.items():
            if sha(Path(path)) != digest: raise ValueError('Queue implementation changed: '+path)
        # Do not let mid-queue source edits silently change the frozen method.
        for path,digest in prototype['source_sha256'].items():
            if sha(Path(path)) != digest: raise ValueError('Frozen source changed: '+path)
        if sha(Path(job['plan'])) != job['plan_sha256']: raise ValueError('Plan changed')
        if check_command(job['command']) != job['input_preflight']:
            raise ValueError('Input identity changed after queue declaration')
        root=Path(job['root']); root.mkdir()
        child=None
        with (root/'execution.log').open('x') as log:
            child=subprocess.Popen(job['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            save(root/'process.json',dict(pid=child.pid,command=job['command']))
            frozen=root/'raw/family_execution_protocol.json'
            while child.poll() is None and not frozen.exists(): time.sleep(1)
            # The child saves its protocol before model loading. Wait for the
            # complete JSON, without treating a transient partial write as failure.
            while frozen.exists():
                try: current=read(frozen); break
                except json.JSONDecodeError:
                    if child.poll() is not None: raise
                    time.sleep(.1)
            if frozen.exists():
                sources={}
                for name,digest in current['source_sha256'].items():
                    path=Path(name)
                    if sha(path)!=digest: raise ValueError('Source changed during capture: '+name)
                    if path.suffix=='.py': sources[name]=path.read_text()
                snapshot_path = root/'source_snapshot.json'
                if snapshot_path.exists():
                    if read(snapshot_path) != sources:
                        raise ValueError('Child source snapshot differs from frozen sources')
                else:
                    save(snapshot_path,sources)
            code=child.wait()
        save(root/'exit.json',dict(exit_code=code))
        if code:
            raise SystemExit('Capture failed; retained log at '+str(root/'execution.log'))
        subprocess.run(verification_command(prototype,root),cwd=ROOT,check=True)
        print(json.dumps(dict(event='FAMILY_QUEUE_JOB_VERIFIED',root=str(root))),flush=True)
    save(a.output/'completion.json',dict(status='ALL_QUEUED_MEASUREMENTS_VERIFIED',
         jobs=len(jobs),case_count=len(seen),whole_research_plan_complete=False))


if __name__=='__main__':main()
