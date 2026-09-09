import hashlib
from types import SimpleNamespace
import pytest
from scripts.build_indexed_call_contracts import check_call
from scripts.indexed_runtime_binding import bind_live_call


def test_live_call_is_unique_and_saved_source_immutable(tmp_path):
    source='def call():\n    aten.index_put_(buffer, [ids], values, True)\n'
    saved=tmp_path/'saved.py'; saved.write_text(source)
    live=tmp_path/'live.py'; live.write_text('\n'+source)
    contract=dict(check_call('aten.index_put_(buffer, [ids], values, True)'),
                  source_path=str(saved),source_sha256=hashlib.sha256(source.encode()).hexdigest())
    module=SimpleNamespace(__file__=str(live))
    match=bind_live_call(contract,[module,module])
    assert match['executing_line']==3
    assert match['executing_filename']==str(live)
    live.write_text(source+'def duplicate():\n    aten.index_put_(buffer, [ids], values, True)\n')
    with pytest.raises(ValueError,match='exactly one'): bind_live_call(contract,[module])
    live.write_text(source.replace('True','False'))
    with pytest.raises(ValueError,match='exactly one'): bind_live_call(contract,[module])
    saved.write_text(source+'\n')
    with pytest.raises(ValueError,match='changed'): bind_live_call(contract,[module])
