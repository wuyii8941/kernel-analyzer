"""Build recurrence batch commands without changing the frozen measurement.

This is command construction only; it does not start jobs or certify results.
The existing queue's path replacement is reused, not a new statistical runner.
"""
from pathlib import Path
import sys
from scripts.run_family_plan_queue import replace_argument, read, sha, ROOT
from scripts.run_decayed_recurrence_capture import select_contracts


def build_command(protocol, full_plan, partition, root, spool):
    full_plan=Path(full_plan).resolve(); partition=Path(partition).resolve()
    if protocol.get('schema')!='decayed-recurrence-capture-v1':
        raise ValueError('Require a recurrence prototype')
    if protocol.get('retain_small_update_vectors'):
        raise ValueError('Do not expand a selected mechanism diagnostic')
    if sha(full_plan)!=protocol['source_sha256'].get(str(full_plan)):
        raise ValueError('Full recurrence plan differs from prototype')
    original=read(full_plan); part=read(partition)
    if part.get('source_plan_sha256')!=sha(full_plan):
        raise ValueError('Partition not bound to full recurrence plan')
    select_contracts(original,part['cases'])
    args=list(protocol['capture_arguments'])
    for name,value in (('--case-plan',partition),('--output-dir',Path(root)/'legacy'),
                       ('--spool-dir',spool),
                       ('--training-bias-profile-v2-output-dir',Path(root)/'raw')):
        args=replace_argument(args,name,value)
    return [sys.executable,str(ROOT/'scripts/run_decayed_recurrence_capture.py'),
            '--recurrence-plan',str(full_plan),*args]
