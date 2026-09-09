#!/usr/bin/env python3
"""Check recorded Linux processes, never infer liveness from a log or lock."""
import argparse
import json
import os
from pathlib import Path
import time


def process(record):
    if record is None: return dict(status='NO_PROCESS_RECORD')
    if record.get('pid_namespace') and record['pid_namespace']!=os.readlink('/proc/self/ns/pid'):
        return dict(status='NAMESPACE_DIFFERS_CANNOT_CHECK_PID')
    pid=int(record['pid']); root=Path('/proc')/str(pid)
    try:
        command=root.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
        stat=root.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]
    except FileNotFoundError: return dict(pid=pid,status='NOT_PRESENT')
    except PermissionError: return dict(pid=pid,status='UNREADABLE_NOT_ASSUMED_STOPPED')
    if command!=record['command']: return dict(pid=pid,status='PID_IDENTITY_DIFFERS')
    return dict(pid=pid,status='EXITED_ZOMBIE' if stat=='Z' else 'LIVE_MATCHING_COMMAND')


def classify(worker,child,exit_record,worker_error=None):
    live='LIVE_MATCHING_COMMAND'
    if child['status']==live: return 'EXPERIMENT_PROCESS_LIVE'
    if worker['status']==live: return 'WORKER_LIVE'
    if any(r['status'] in ('UNREADABLE_NOT_ASSUMED_STOPPED','NAMESPACE_DIFFERS_CANNOT_CHECK_PID') for r in (worker,child)):
        return 'PROCESS_LIVENESS_UNKNOWN_REQUIRES_DIAGNOSIS'
    if exit_record is not None:
        return 'PROCESS_EXIT_ZERO_REQUIRES_RESULT_VERIFICATION' if exit_record['exit_code']==0 else 'PROCESS_EXIT_NONZERO'
    if worker_error is not None:
        return 'WORKER_STOPPED_WITH_RECORDED_ERROR'
    return 'NO_LIVE_MATCH_VERIFIED_NO_EXIT_STATUS_REQUIRES_DIAGNOSIS'


def inspect(root):
    def read(name):
        path=root/name
        return json.loads(path.read_text()) if path.exists() else None
    worker=process(read('worker.json')); child=process(read('child.json')); exited=read('exit.json')
    error=read('worker_error.json')
    return dict(record_dir=str(root.resolve()),worker=worker,child=child,exit=exited,worker_error=error,
                status=classify(worker,child,exited,error),numerical_completion='NOT_INFERRED_FROM_PROCESS_STATUS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record-dir',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.output and (a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh'))):
        p.error('Choose a new output under /data1/tzh')
    result=dict(observed_unix=time.time(),records=[inspect(r) for r in a.record_dir])
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
