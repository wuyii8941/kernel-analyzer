"""Console summaries must preserve task conclusions and missing decompositions."""
import copy
import gzip
import json
from pathlib import Path

from kernel_analyzer import cli, measure


def render(report, monkeypatch, capsys):
    original = copy.deepcopy(report)
    monkeypatch.setattr(measure, 'run', lambda declaration, output: report)
    assert cli.measure_main(['--declaration', 'unused.json', '--out', 'unused-result.json']) == 0
    text = capsys.readouterr().out
    assert report == original
    return json.loads(text.splitlines()[0])


def test_real_rms_task_effect_is_visible_without_inventing_dweight_program_reference(monkeypatch, capsys):
    root = Path(__file__).resolve().parents[2]
    with gzip.open(root / '5_specification/evidence/flaggems_rms_norm/measurement_v1.json.gz', 'rt') as f:
        report = json.load(f)
    outputs = render(report, monkeypatch, capsys)['outputs']
    for name in ('y', 'dx', 'dweight'):
        total = outputs[name]['comparisons']['total']
        assert total['status'] == 'evaluated'
        assert total['covers_complete_public_output'] is True
        assert 'nonzero' in total['aligned']
        assert 'not confirmed' in total['fixed_mean']
        assert total['verdict_scope'] == 'candidate_minus_admitted_task_specification'
    assert outputs['dweight']['comparisons']['numerical']['status'] == 'not established'
    assert outputs['dweight']['comparisons']['semantic']['status'] == 'not established'
    assert outputs['dweight']['reference_scope'].startswith('not established')
    assert outputs['y']['comparisons']['semantic']['status'] == 'evaluated'


def test_black_box_statistics_are_not_presented_as_program_error(monkeypatch, capsys):
    stats = {'fixed_mean': {'summary': 'not confirmed'}, 'aligned': {'summary': 'nonzero'}}
    output = {'status': 'evaluated (black box)', 'comparison': 'K - supplied reference (no K_R decomposition)',
              'verdict_scope': 'reference_difference; not a semantic verdict', 'statistics': stats}
    report = {'levels': [{'level': {}, 'status': 'ok', 'mode': 'black-box', 'outputs': {'out': output}}],
              'failure_counts': {}}
    result = render(report, monkeypatch, capsys)['outputs']['out']['comparisons']
    assert result['numerical']['status'] == 'not established'
    assert result['semantic']['status'] == 'not established'
    assert result['total']['aligned'] == 'nonzero'
    assert result['total']['comparison'] == output['comparison']
    assert result['total']['verdict_scope'] == output['verdict_scope']


def test_supplied_reference_is_not_promoted_to_an_admitted_task(monkeypatch, capsys):
    stats = {'fixed_mean': {'summary': 'not confirmed'}, 'aligned': {'summary': 'nonzero'}}
    output = {'status': 'evaluated', 'reference': {'elements': 4, 'reference_scope': 'call-level'},
              'statistics': stats,
              'semantic': {'statistics': stats, 'verdict_scope': 'reference_difference; not a semantic verdict'},
              'total': {'statistics': stats, 'verdict_scope': 'candidate versus supplied reference; not a semantic verdict'}}
    report = {'levels': [{'level': {}, 'status': 'ok', 'outputs': {'out': output}}], 'failure_counts': {}}
    result = render(report, monkeypatch, capsys)['outputs']['out']['comparisons']
    assert result['semantic']['comparison'] == 'program reference - supplied reference'
    assert result['semantic']['verdict_scope'] == output['semantic']['verdict_scope']
    assert result['total']['comparison'] == 'K - supplied reference'
    assert result['total']['verdict_scope'] == output['total']['verdict_scope']


def test_missing_task_reference_and_special_value_result_remain_explicit(monkeypatch, capsys):
    output = {'status': 'evaluated', 'reference': {'elements': 4, 'complete_rate': 1., 'reference_scope': 'call-level'},
              'statistics': {'fixed_mean': {'summary': 'cannot judge / not established'}},
              'task_special_value_verdict': {'judgment': 'contradiction', 'mismatched_coordinates': 4}}
    report = {'levels': [{'level': {}, 'status': 'ok', 'outputs': {'out': output}}], 'failure_counts': {}}
    result = render(report, monkeypatch, capsys)['outputs']['out']
    assert result['comparisons']['numerical']['fixed_mean'] == 'cannot judge / not established'
    assert result['comparisons']['semantic']['status'] == 'not established'
    assert result['comparisons']['total']['status'] == 'not established'
    assert result['task_special_value_verdict'] == output['task_special_value_verdict']
