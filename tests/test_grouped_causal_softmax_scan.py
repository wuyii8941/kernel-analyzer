from pathlib import Path
from scripts.scan_grouped_causal_softmax_sources import scan


def test_discovers_without_a_requested_symbol():
    root = Path(__file__).resolve().parents[1]
    path = root / 'results/coverage/runtime_releases/qwen_seq128_r1/trace/model__0_forward_segment0_executed/output_code.py'
    rows = scan(path.read_text())
    checked = [r for r in rows if r['status'] == 'SOURCE_CHECKED']
    assert len(checked) == 1
    assert checked[0]['contract']['width'] == 128
    assert len(rows) > len(checked)
