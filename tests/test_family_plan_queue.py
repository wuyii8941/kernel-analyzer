from pathlib import Path
import json
import pytest
from scripts.run_family_plan_queue import build_command, replace_argument


def prototype():
    return dict(variant='FP32_NATIVE',parallel_measurement=True,
                local_tolerance_baseline=dict(rtol=1e-5,atol=1e-8),
                capture_arguments=['--case-plan','old','--output-dir','old',
                   '--spool-dir','old','--training-bias-profile-v2-output-dir','old',
                   '--states','32','--model','original-model'])


def test_queue_changes_only_execution_paths():
    p=prototype()
    command=build_command(p,Path('/manifest'),Path('/plan'),Path('/result'),Path('/spool'))
    assert command[command.index('--states')+1]=='32'
    assert command[command.index('--model')+1]=='original-model'
    assert command[command.index('--local-rtol')+1]=='1e-05'
    assert command[command.index('--training-bias-profile-v2-output-dir')+1]=='/result/raw'
    assert '--parallel-measurement' in command
    assert p==prototype()


@pytest.mark.parametrize('args',[[],['--x'],['--x','a','--x','b']])
def test_ambiguous_options_rejected(args):
    with pytest.raises(ValueError): replace_argument(args,'--x','new')


def test_do_not_expand_selected_diagnostics():
    p=prototype(); p['retain_small_update_vectors']=True
    with pytest.raises(ValueError): build_command(p,Path('/m'),Path('/p'),Path('/r'),Path('/s'))


def test_additional_family_uses_pinned_entrypoint(tmp_path):
    from scripts.run_declared_reference_family import declaration
    d=declaration('ROW_SUM','row_sum_reference','-row-sum-common-input')
    manifest=tmp_path/'reference.json'
    m=dict(reference_family='ROW_SUM',adapter_sha256=d['adapter_sha256'],additional_reference_declaration=d)
    manifest.write_text(json.dumps(m))
    p=dict(prototype(),reference_family='ROW_SUM')
    command=build_command(p,manifest,Path('/plan'),Path('/result'),Path('/spool'))
    assert command[1].endswith('/run_declared_reference_family.py')
    assert command[command.index('--new-family')+1]=='ROW_SUM'
    assert command[command.index('--states')+1]=='32'
    m['additional_reference_declaration']['entrypoint_sha256']='changed'
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='declaration'):
        build_command(p,manifest,Path('/plan'),Path('/result'),Path('/spool'))
