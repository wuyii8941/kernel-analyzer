import json
from pathlib import Path
import pytest
from scripts.run_family_plan_queue_v2 import build_command, verification_command


def test_forward_queue_preserves_capture_settings(tmp_path, monkeypatch):
    import scripts.run_residual_rms_forward_capture as capture
    checked = []
    monkeypatch.setattr(capture, 'select', lambda manifest, cases: checked.append((manifest, cases)))
    manifest = tmp_path/'family.json'
    manifest.write_text(json.dumps({'cases': [{'case_id': 'new'}]}))
    plan = tmp_path/'batch.json'
    plan.write_text(json.dumps({'cases': [{'case_id': 'new'}]}))
    protocol = {'schema': 'residual-rms-forward-capture-v1', 'capture_arguments': [
        '--case-plan', 'old', '--output-dir', 'old', '--spool-dir', 'old',
        '--training-bias-profile-v2-output-dir', 'old', '--states', '32',
        '--device', 'cuda:0', '--input-bank', 'frozen-bank']}
    root = tmp_path/'job'
    cmd = build_command(protocol, manifest, plan, root, tmp_path/'spool')
    assert checked
    assert Path(cmd[1]).name == 'run_residual_rms_forward_capture.py'
    assert cmd[cmd.index('--case-plan')+1] == str(plan)
    assert cmd[cmd.index('--input-bank')+1] == 'frozen-bank'
    assert cmd[cmd.index('--states')+1] == '32'
    assert '--reference-variant' not in cmd
    verify = verification_command(protocol, root)
    assert Path(verify[1]).name == 'finalize_residual_rms_forward.py'
    assert '--baselines-dir' not in verify


def test_forward_queue_requires_valid_partition(tmp_path, monkeypatch):
    import scripts.run_residual_rms_forward_capture as capture
    def reject(*args):
        raise ValueError('Not in frozen family')
    monkeypatch.setattr(capture, 'select', reject)
    manifest = tmp_path/'family.json'
    manifest.write_text('{}')
    plan = tmp_path/'plan.json'
    plan.write_text('{"cases":[]}')
    with pytest.raises(ValueError, match='frozen family'):
        build_command({'schema':'residual-rms-forward-capture-v1'}, manifest, plan,
                      tmp_path/'job', tmp_path/'spool')


@pytest.mark.parametrize('schema,entrypoint,module_name', [
    ('embedding-lookup-forward-capture-v1', 'run_embedding_lookup_capture.py',
     'scripts.run_embedding_lookup_capture'),
    ('grouped-causal-softmax-forward-capture-v1',
     'run_grouped_causal_softmax_capture.py',
     'scripts.run_grouped_causal_softmax_capture'),
])
def test_explicit_output_family_uses_shared_queue(schema, entrypoint, module_name,
                                                  tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module(module_name)
    checked = []
    monkeypatch.setattr(module, 'select', lambda manifest, cases:
                        checked.append((manifest, cases)))
    manifest = tmp_path/'family.json'
    manifest.write_text(json.dumps({'cases': [{'case_id': 'new'}]}))
    plan = tmp_path/'batch.json'
    plan.write_text(json.dumps({'cases': [{'case_id': 'new'}]}))
    protocol = {'schema': schema, 'capture_arguments': [
        '--case-plan', 'old', '--output-dir', 'old', '--spool-dir', 'old',
        '--training-bias-profile-v2-output-dir', 'old', '--states', '32',
        '--device', 'cuda:0', '--input-bank', 'frozen-bank']}
    root = tmp_path/'job'
    command = build_command(protocol, manifest, plan, root, tmp_path/'spool')
    assert checked
    assert Path(command[1]).name == entrypoint
    assert command[command.index('--family-plan')+1] == str(manifest)
    assert command[command.index('--case-plan')+1] == str(plan)
    assert Path(verification_command(protocol, root)[1]).name == \
        'finalize_explicit_output_capture.py'
