#!/usr/bin/env python3
"""Launch an explicitly supplied experiment with durable log and exit status."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Direct execution sets sys.path[0] to ``scripts/`` rather than the repository
# root, while this launcher intentionally imports sibling helpers as
# ``scripts.<module>``.  Make that execution mode match ``python -m`` and the
# detached worker it creates.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_CACHE_ENV = {
    'HF_HOME':'/data1/tzh/cache/huggingface',
    'HF_DATASETS_CACHE':'/data1/tzh/cache/huggingface/datasets',
    'XDG_CACHE_HOME':'/data1/tzh/cache/xdg',
    'TORCH_HOME':'/data1/tzh/cache/torch',
    'TORCH_EXTENSIONS_DIR':'/data1/tzh/cache/torch_extensions',
    'TORCHINDUCTOR_CACHE_DIR':'/data1/tzh/cache/torchinductor',
    'TRITON_CACHE_DIR':'/data1/tzh/cache/triton',
    'CUDA_CACHE_PATH':'/data1/tzh/cache/cuda',
    'MPLCONFIGDIR':'/data1/tzh/cache/matplotlib',
    'PYTHONDONTWRITEBYTECODE':'1',
}


def save(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record-dir',type=Path,required=True)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--after-record-dir',type=Path)
    p.add_argument('--verify-after-root',type=Path)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args(); root=a.record_dir.resolve()
    if not root.is_relative_to(Path('/data1/tzh')): p.error('Records must stay under /data1/tzh')
    os.environ.update(DATA_CACHE_ENV)
    if a.worker:
        record=json.loads((root/'command.json').read_text())
        started=time.time()
        try:
            if record.get('after_record_dir'):
                from scripts.check_detached_experiments import inspect
                dependency=Path(record['after_record_dir'])
                while not (dependency/'exit.json').exists():
                    status=inspect(dependency)['status']
                    if status not in ('EXPERIMENT_PROCESS_LIVE','WORKER_LIVE'):
                        raise RuntimeError('Dependency has no verified live process or recorded exit: '+status)
                    time.sleep(2)
                if json.loads((dependency/'exit.json').read_text())['exit_code']!=0:
                    raise RuntimeError('Dependency exited unsuccessfully; do not start follow-up')
            if record.get('verify_after_root'):
                from scripts.finalize_numerical_family import finalize
                from scripts.snapshot_frozen_sources import preserve
                capture_root=Path(record['verify_after_root'])
                snapshot=capture_root/'source_snapshot.json'
                if not snapshot.exists():
                    preserved=preserve(capture_root/'raw/family_execution_protocol.json', snapshot)
                    save(root/'dependency_snapshot_creation.json', dict(**preserved,
                         created_unix=time.time(), phase='AFTER_CAPTURE_BEFORE_VERIFICATION',
                         condition='ALL_CURRENT_SOURCE_HASHES_MATCH_FROZEN_PROTOCOL'))
                verified=finalize(capture_root)
                save(root/'dependency_verification.json',verified)
                if not verified['measurement_complete']:
                    raise RuntimeError('Dependency results did not verify')
            if record.get('input_preflight') is not None:
                from scripts.preflight_capture_input import check_command
                checked = check_command(record['command'])
                if checked != record['input_preflight']:
                    raise RuntimeError('Capture input preflight changed while awaiting execution')
                save(root/'execution_input_preflight.json', checked)
            with (root/'execution.log').open('x') as log:
                child=subprocess.Popen(record['command'],cwd=record['cwd'],stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT)
                save(root/'child.json',dict(pid=child.pid,started_unix=started,command=record['command'],
                     pid_namespace=os.readlink('/proc/self/ns/pid')))
                code=child.wait()
            save(root/'exit.json',dict(exit_code=code,started_unix=started,finished_unix=time.time()))
        except Exception as exc:
            save(root/'worker_error.json',dict(error=repr(exc),time_unix=time.time()))
            raise
        return
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if root.exists() or not command: p.error('Use new record directory and explicit command')
    from scripts.preflight_recurrence_launch import preflight
    preflight_result=preflight(command)
    from scripts.preflight_capture_input import check_command
    try:
        input_preflight = check_command(command)
    except Exception as exc:
        root.mkdir(parents=True)
        save(root/'preflight_error.json',dict(command=command,error=repr(exc),
             status='REJECTED_BEFORE_PROCESS_LAUNCH',created_unix=time.time()))
        raise
    for dependency in (a.after_record_dir,a.verify_after_root):
        if dependency and (not dependency.exists() or not dependency.resolve().is_relative_to(Path('/data1/tzh'))):
            p.error('Dependencies must exist under /data1/tzh')
    root.mkdir(parents=True)
    if preflight_result is not None:
        save(root/'launch_preflight.json',preflight_result)
    save(root/'command.json',dict(command=command,cwd=str(Path.cwd()),created_unix=time.time(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        after_record_dir=str(a.after_record_dir.resolve()) if a.after_record_dir else None,
        verify_after_root=str(a.verify_after_root.resolve()) if a.verify_after_root else None,
        cache_environment=DATA_CACHE_ENV,
        input_preflight=input_preflight,
        scope='Process launch only; not experiment completion'))
    with (root/'worker.log').open('x') as log:
        worker=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--record-dir',str(root),'--worker'],
            cwd=Path.cwd(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(root/'worker.json',dict(pid=worker.pid,command=[sys.executable,str(Path(__file__).resolve()),
         '--record-dir',str(root),'--worker'],started_unix=time.time(),
         pid_namespace=os.readlink('/proc/self/ns/pid')))
    print(json.dumps(dict(worker_pid=worker.pid,record_dir=str(root))))


if __name__=='__main__':main()
