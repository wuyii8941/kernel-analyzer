import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import pytest

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(not ROOT.is_relative_to(Path('/data1/tzh')),reason='Launcher intentionally requires the declared data disk')


def wait_record(path):
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        if path.exists():
            try: return json.loads(path.read_text())
            except json.JSONDecodeError: pass
        time.sleep(.05)
    raise AssertionError('Detached test did not produce terminal record: '+str(path))


def launch(root,code,dependency=None):
    command=[sys.executable,str(ROOT/'scripts/launch_detached_experiment.py'),'--record-dir',str(root)]
    if dependency: command+=['--after-record-dir',str(dependency)]
    command+=['--',sys.executable,'-c',code]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=True)
    return json.loads(result.stdout)


def test_actual_exit_code_and_log_preserved():
    with tempfile.TemporaryDirectory(prefix='.detached-test-',dir=ROOT) as directory:
        root=Path(directory)/'job'
        record=launch(root,'import os; print("test-output",flush=True); print(os.environ["HF_HOME"],flush=True); raise SystemExit(7)')
        assert record['worker_pid']>0
        assert wait_record(root/'exit.json')['exit_code']==7
        assert 'test-output' in (root/'execution.log').read_text()
        assert '/data1/tzh/cache/huggingface' in (root/'execution.log').read_text()


def test_failed_dependency_never_runs_followup():
    with tempfile.TemporaryDirectory(prefix='.detached-test-',dir=ROOT) as directory:
        first=Path(directory)/'first'; second=Path(directory)/'second'
        launch(first,'raise SystemExit(7)')
        wait_record(first/'exit.json')
        launch(second,'print("must not run")',first)
        assert 'Dependency exited unsuccessfully' in wait_record(second/'worker_error.json')['error']
        assert not (second/'child.json').exists()


def test_input_preflight_rejection_is_recorded_without_child():
    with tempfile.TemporaryDirectory(prefix='.detached-test-',dir=ROOT) as directory:
        root = Path(directory) / 'job'
        result = subprocess.run([
            sys.executable, str(ROOT/'scripts/launch_detached_experiment.py'),
            '--record-dir', str(root), '--', sys.executable,
            'scripts/run_residual_rms_forward_capture.py', '--states', '32'],
            cwd=ROOT, capture_output=True, text=True)
        assert result.returncode != 0
        rejection = json.loads((root/'preflight_error.json').read_text())
        assert rejection['status'] == 'REJECTED_BEFORE_PROCESS_LAUNCH'
        assert not (root/'worker.json').exists()
        assert not (root/'child.json').exists()


@pytest.mark.parametrize('changed',[False,True])
def test_snapshot_automation_does_not_bypass_result_verification(changed):
    with tempfile.TemporaryDirectory(prefix='.detached-test-',dir=ROOT) as directory:
        root=Path(directory)
        capture=root/'capture'
        (capture/'raw').mkdir(parents=True)
        source=root/'declared.py'
        source.write_text('value = 1\n')
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        # Deliberately lacks measurement fields. Snapshot preservation must not
        # turn this incomplete capture into permission to execute a follow-up.
        (capture/'raw/family_execution_protocol.json').write_text(json.dumps(
            dict(source_sha256={str(source):digest})))
        if changed:
            source.write_text('value = 2\n')
        job=root/'job'
        subprocess.run([sys.executable,str(ROOT/'scripts/launch_detached_experiment.py'),
                        '--record-dir',str(job),'--verify-after-root',str(capture),'--',
                        sys.executable,'-c','print("must not run")'],
                       cwd=ROOT,capture_output=True,text=True,check=True)
        error=wait_record(job/'worker_error.json')
        assert not (job/'child.json').exists()
        assert (capture/'source_snapshot.json').exists() is (not changed)
        if changed:
            assert 'Source changed' in error['error']
        else:
            creation=json.loads((job/'dependency_snapshot_creation.json').read_text())
            assert creation['phase']=='AFTER_CAPTURE_BEFORE_VERIFICATION'
