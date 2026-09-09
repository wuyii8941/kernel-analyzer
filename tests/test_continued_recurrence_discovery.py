import pytest
from scripts.scan_continued_recurrence_sources import checked_candidate
from scripts.scan_recurrence_sources import scan
from kernel_analyzer.continued_recurrence_source import check_source
from test_continued_recurrence_source import literal_source


def source():
    return literal_source().replace('kernel', 'triton_test')


def test_origin_discovered_without_manual_symbol_choice():
    text = source()
    rows = scan(text, checker=checked_candidate)
    assert rows[0]['contract'] == check_source(text, 'triton_test', time_start=8)


@pytest.mark.parametrize('old,new', [('tmp3 = -tmp2','tmp3 = tmp2'),
    ('10752 + x1', '9216 + x1'), ('*fp32','*bf16')])
def test_inferred_origin_does_not_bypass_full_check(old,new):
    text = source()
    assert old in text
    assert scan(text.replace(old,new), checker=checked_candidate)[0]['status']=='REJECTED'


def test_nonfamily_and_duplicate_symbols_remain_visible():
    text=source()+'\n'+source()+'\ntriton_other = factory()\n'
    rows=scan(text, checker=checked_candidate)
    assert len(rows)==3 and all(r['status']=='REJECTED' for r in rows)
